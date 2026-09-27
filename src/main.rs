//! IFAN: image filtering via an iterative filter-adaptive network.
//!
//! A single-image, 8-bit restoration engine in the lightgpu family: PNG in, PNG
//! out, `-m` for the weights. The GPU path is the default when the crate is built
//! with its `cuda` feature; `--cpu` forces the CPU path.
//!
//! Two things about this engine are unusual and deliberate:
//!
//!   * The input is CROPPED, not padded, to a multiple of 8 - upstream's
//!     `refine_image`. The crop is part of the method rather than a limitation:
//!     the network runs at 1/8 resolution to predict its filters, so a
//!     non-multiple-of-8 input has no well-defined filter grid.
//!   * There is no tiling. A whole 1080p frame needs 3.08 GiB of device memory -
//!     the filter tensor is 1.84 GiB of it - and the memory plan says so, with the
//!     free VRAM it compared against, before a byte is allocated. See `gpu::plan`.
use std::io::{Read, Write};
use std::process::ExitCode;

use ifan::{image, memguard, net, weights};
#[cfg(feature = "cuda")]
use ifan::gpu;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ifan: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    weights: Option<String>,
    input: Option<String>,
    output: Option<String>,
    cpu: bool,
    /// `--dump <file>`, a development build only: the raw f32 output planes,
    /// which is the format `tools/gate.py` compares in.
    #[cfg(feature = "dev")]
    dump: Option<String>,
    /// `--dump-op <name>`, a development build only: stop at the op that writes
    /// `<name>` and dump THAT, which is how a divergence between this engine and
    /// `tools/reference.py` is located to a single op instead of inferred from
    /// the final image.
    #[cfg(feature = "dev")]
    dump_op: Option<String>,
}

/// The flags a `--features dev` build adds, and nothing else does.
#[cfg(feature = "dev")]
const DEV_USAGE: &str = "
  development build only:
    --dump <file>      write the output planes as raw f32 (tools/gate.py reads
                       this format: a 12-byte u32 c,h,w header then the planes)
    --dump-op <name>   stop at the op that writes <name> and dump IT instead of
                       running to the end; writes no PNG unless <name> is `out`,
                       because an activation is not an image
";
#[cfg(not(feature = "dev"))]
const DEV_USAGE: &str = "";

fn usage() -> String {
    let base = "\
ifan - IFAN image restoration (single image, 8-bit)

    ifan -m <weights.safetensors> [-i <in.png>] [-o <out.png>] [--cpu]

    -m, --model <path>  the converted weights (models/IFAN.safetensors)
    -i, --input <path>  input PNG (default: stdin)
    -o, --output <path> output PNG (default: stdout)
        --cpu           force the CPU path instead of CUDA
    -h, --help

  environment:
    IFAN_TIME=1         CUDA only: print the per-kernel device time of a pass
                        (launches, milliseconds, GFLOP/s). It changes no
                        arithmetic - a timed run and an untimed one write
                        identical output - and is off by default because it
                        synchronises after every launch.
    IFAN_CPU_TIME=1     the same for the CPU path, by op, shape and name."
        .to_string();
    base + DEV_USAGE
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        weights: None,
        input: None,
        output: None,
        cpu: false,
        #[cfg(feature = "dev")]
        dump: None,
        #[cfg(feature = "dev")]
        dump_op: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--model" => a.weights = Some(it.next().ok_or("-m needs a path")?),
            "-i" | "--input" => a.input = Some(it.next().ok_or("-i needs a path")?),
            "-o" | "--output" => a.output = Some(it.next().ok_or("-o needs a path")?),
            "--cpu" => a.cpu = true,
            // A RELEASE BUILD REFUSES THEM BY NAME rather than ignoring them: a
            // script that asks for a dump and silently gets none would be worse
            // off than one that is told to rebuild.
            #[cfg(feature = "dev")]
            "--dump" => a.dump = Some(it.next().ok_or("--dump needs a path")?),
            #[cfg(feature = "dev")]
            "--dump-op" => a.dump_op = Some(it.next().ok_or("--dump-op needs a name")?),
            #[cfg(not(feature = "dev"))]
            "--dump" | "--dump-op" => {
                return Err(format!(
                    "`{arg}` is a development flag; rebuild with --features dev\n\n{}",
                    usage()
                ))
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
        }
    }
    if a.weights.is_none() {
        return Err(format!("no weights: pass -m <path>\n\n{}", usage()));
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let wpath = args.weights.clone().unwrap();
    let ws = weights::Weights::load(&wpath)?;

    // The input, cropped to a multiple of 8. `refine` returns whether it changed
    // anything, which is worth printing: a user who sees a crop silently happen
    // has no way to tell whether their image was resized.
    let (mut img, _from_stdin) = match &args.input {
        Some(p) => (image::load_rgb(p)?, false),
        None => {
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| format!("read stdin: {e}"))?;
            (image::load_rgb_stream(&buf[..])?, true)
        }
    };
    let (h0, w0) = (img.h, img.w);
    img.refine(8)?;
    if (img.h, img.w) != (h0, w0) {
        eprintln!(
            "ifan: cropped {}x{} to {}x{} (the network predicts filters at 1/8 \
             resolution, so the input must be a multiple of 8)",
            w0, h0, img.w, img.h
        );
    }

    let blob = net::Blob { c: 3, h: img.h, w: img.w, d: img.data.clone() };
    // `stop` is an activation to halt at, and only a development build can set
    // one - so in a release build this is `None` because no flag can make it
    // anything else, not because different code was compiled.
    #[cfg(feature = "dev")]
    let stop = args.dump_op.as_deref();
    #[cfg(not(feature = "dev"))]
    let stop: Option<&str> = None;

    let out = if args.cpu {
        cpu_path(&ws, &blob, stop)?
    } else {
        #[cfg(feature = "cuda")]
        {
            gpu_path(&ws, &blob, stop)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            // A build without the `cuda` feature has no GPU path at all, so
            // asking for the default is asking for something this binary cannot
            // do. Saying so beats silently running the CPU twin: the two agree
            // numerically, but a user who built without `cuda` should learn that
            // from the command line, not from a benchmark.
            return Err(
                "this build has no CUDA backend (built without the `cuda` feature); \
                 pass --cpu or rebuild with the default features"
                    .to_string(),
            );
        }
    };

    // A development build only: the same numbers the PNG is made from, before
    // 8-bit quantisation, which is the format `tools/gate.py` compares in.
    #[cfg(feature = "dev")]
    if let Some(p) = &args.dump {
        dump_f32(p, &out)?;
    }
    // `--dump-op` RETURNS AN ACTIVATION, NOT AN IMAGE, so writing a PNG from it
    // is only meaningful when the name asked for IS the network's output. Every
    // other name is a feature map - 1 channel at 1/8 resolution for the DME map,
    // 128 for most of the network - and handing one to the PNG writer either
    // writes a picture that means nothing or, when it has fewer than three
    // planes, indexes past the end of it. Both are worse than saying so.
    #[cfg(feature = "dev")]
    if let Some(name) = &args.dump_op {
        if name != "out" {
            eprintln!(
                "ifan: stopped at `{name}` ({}x{}x{}); no image written - --dump-op is \
                 a bisection tool, not a filter, so use --dump <file> for the numbers",
                out.c, out.h, out.w
            );
            return Ok(());
        }
    }
    let rgb = image::Image { w: out.w, h: out.h, data: out.d }.to_rgb8();
    match &args.output {
        Some(p) => image::save_rgb(p, out.w, out.h, &rgb)?,
        None => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            image::save_rgb_stream(&mut lock, out.w, out.h, &rgb)?;
            lock.flush().map_err(|e| format!("flush stdout: {e}"))?;
        }
    }
    Ok(())
}

/// The same `Vec<Op>` through the CPU interpreter.
fn cpu_path(
    ws: &weights::Weights,
    blob: &net::Blob,
    stop: Option<&str>,
) -> Result<net::Blob, String> {
    let ops = net::graph(&ws.arch);
    // The CPU plan is a REFUSAL GUARD, checked before the first blob is
    // allocated, and it is printed like the GPU's plan line: a pass that is about
    // to use 11 GiB of host memory should say so. It is arithmetic rather than a
    // measurement (see src/memguard.rs), which is why the line says "modelled".
    let plan = memguard::plan_cpu(ws, blob.h, blob.w)?;
    // The modelled PEAK, from a simulation of this interpreter's own pooled
    // allocator - not a sum over the graph's names (see src/memguard.rs).
    eprintln!(
        "ifan: cpu: {} graph buffers, {} modelled peak, {} with allocator slack",
        plan.names,
        memguard::fmt_bytes(plan.peak_live),
        memguard::fmt_bytes(plan.peak)
    );
    let mut cpu = net::Cpu::new(ws);
    #[cfg(feature = "dev")]
    return match stop {
        Some(name) => Ok(cpu.run_until(&ops, blob, name)),
        None => Ok(cpu.run(&ops, blob)),
    };
    #[cfg(not(feature = "dev"))]
    {
        let _ = stop;
        Ok(cpu.run(&ops, blob))
    }
}

#[cfg(feature = "cuda")]
fn gpu_path(
    ws: &weights::Weights,
    blob: &net::Blob,
    stop: Option<&str>,
) -> Result<net::Blob, String> {
    let mut g = gpu::Gpu::new(ws)?;
    let plan = g.plan(blob.h, blob.w)?;
    // The plan line goes to stderr because stdout is the image.
    eprintln!("ifan: {}", g.describe(&plan));
    let ops = net::graph(&ws.arch);
    #[cfg(feature = "dev")]
    let out = match stop {
        Some(name) => g.run_until(&ops, &plan, blob, name),
        None => g.run(&ops, &plan, blob),
    };
    #[cfg(not(feature = "dev"))]
    let out = {
        let _ = stop;
        g.run(&ops, &plan, blob)
    };
    // The model and the allocator, printed together: `describe` is the arithmetic
    // the plan was made on and this is what the allocator did with it. A large gap
    // between them is a bug in one of the two rather than a fact about the image.
    eprintln!("ifan: actual {}", g.describe_actual());
    out
}

/// Raw planar f32, the format the golden comparison reads - a development build
/// only.
///
/// A separate path from the PNG so a comparison reads the numbers the network
/// produced rather than the numbers after 8-bit quantisation: 1/255 is small but
/// it is exactly the scale of the differences a real bug produces.
#[cfg(feature = "dev")]
fn dump_f32(path: &str, b: &net::Blob) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    f.write_all(&(b.c as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(b.h as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(b.w as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    for v in &b.d {
        f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}
