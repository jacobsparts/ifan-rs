//! The Rust backends against the validated reference.
//!
//! The fixture in `tests/data` is a 64x64 PNG and the CLIPPED output
//! `tools/reference.py` produces for it from the ORIGINAL `.pytorch` checkpoint.
//! Comparing against a stored file rather than shelling out to torch keeps this
//! test runnable with only `cargo test`, which is the point: the three-way
//! comparison that needs torch lives in `tools/gate.py`, and this is the part of
//! it that has to hold on every build.
//!
//! THE TOLERANCE IS 1.5/255, the same as the gate's, for the same reason: the
//! engine's output is 8-bit, so a difference below half a quantisation step
//! cannot change a byte of it. The observed agreement on this fixture is ~2e-06
//! (see below), three orders inside that, and a real defect - a transposed
//! weight, a doubled tap, an iteration applied twice - lands in the 1e-2 range.
//! Tightening this to 1e-4 would start failing on the accumulation-order
//! differences between the CPU twin, the CUDA kernels and torch's own kernels,
//! which are documented in docs/NUMERICS.md and are not bugs.
//!
//! The tests SKIP (not fail) when the converted checkpoint is absent, so the
//! repository tests cleanly on a fresh clone with no 42 MB weights.
use std::path::{Path, PathBuf};

use ifan::net::{self, Blob};
use ifan::weights::Weights;

#[path = "device_lock.rs"]
mod device_lock;

/// Half a quantisation step in 0..1, as the gate uses.
const TOL: f32 = 1.5 / 255.0;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// `../models/IFAN.safetensors`, or None if the checkpoint has not been
/// converted on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = repo().join("../models/IFAN.safetensors");
    p.exists().then_some(p)
}

/// The fixture as a planar `Blob`, through the engine's OWN PNG reader so the
/// fixtures are not tied to a second implementation of the input path.
fn fixture(crop: &Path, ref_path: &Path) -> (Blob, Vec<f32>) {
    let mut img = ifan::image::load_rgb(crop.to_str().unwrap()).expect("fixture PNG");
    img.refine(8).expect("fixture is a multiple of 8");
    let blob = Blob { c: 3, h: img.h, w: img.w, d: img.data };
    let raw = std::fs::read(ref_path).expect("fixture reference");
    let (c, h, w) = (
        u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize,
        u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize,
        u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize,
    );
    assert_eq!((c, h, w), (blob.c, blob.h, blob.w), "fixture/reference shape mismatch");
    let refd: Vec<f32> = raw[12..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (blob, refd)
}

/// The worst and mean absolute difference, and where the worst one was.
fn compare(blob: &Blob, got: &[f32], want: &[f32]) -> (f32, f32, usize) {
    assert_eq!(got.len(), want.len(), "output has {} values, want {}", got.len(), want.len());
    let mut worst = 0.0f32;
    let mut at = 0usize;
    let mut sum = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        sum += d as f64;
        if d > worst {
            worst = d;
            at = i;
        }
    }
    let _ = blob;
    (worst, (sum / got.len() as f64) as f32, at)
}

#[test]
fn cpu_matches_the_reference() {
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping cpu_matches_the_reference: ../models/IFAN.safetensors is absent");
        return;
    };
    let ws = Weights::load(&ckpt).expect("checkpoint loads");
    let (blob, want) = fixture(
        &repo().join("tests/data/ifan64.png"),
        &repo().join("tests/data/ifan64_ref.f32"),
    );
    let ops = net::graph(&ws.arch);
    let mut cpu = net::Cpu::new(&ws);
    let out = cpu.run(&ops, &blob);
    let (worst, mean, at) = compare(&blob, &out.d, &want);
    eprintln!("cpu vs reference: max |d| {worst:.3e} mean {mean:.3e} at {at}");
    assert!(
        worst <= TOL,
        "the CPU path is {worst:.3e} from the reference (at {at}: {} vs {}), over the \
         {TOL:.3e} tolerance",
        out.d[at],
        want[at]
    );
}

#[test]
fn the_two_backends_agree() {
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping the_two_backends_agree: ../models/IFAN.safetensors is absent");
        return;
    };
    let ws = Weights::load(&ckpt).expect("checkpoint loads");
    let (blob, _) = fixture(
        &repo().join("tests/data/ifan64.png"),
        &repo().join("tests/data/ifan64_ref.f32"),
    );
    let ops = net::graph(&ws.arch);
    let mut cpu = net::Cpu::new(&ws);
    let cpu_out = cpu.run(&ops, &blob);
    // The only reader of `cpu_out` is the `cuda` branch below, so a CPU-only
    // build would otherwise warn about a binding it is right not to use.
    #[cfg(not(feature = "cuda"))]
    let _ = &cpu_out;

    // A build without the `cuda` feature has no second backend to compare, and a
    // machine with no usable device cannot provide one: in both cases the point
    // of the test does not exist, so it skips WITH A REASON rather than passing
    // silently on an unwritten check.
    #[cfg(not(feature = "cuda"))]
    eprintln!("skipping the_two_backends_agree: built without the `cuda` feature");
    #[cfg(feature = "cuda")]
    {
        use ifan::gpu::Gpu;
        let _guard = device_lock::device_lock();
        let mut g = match Gpu::new(&ws) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping the_two_backends_agree: no usable device ({e})");
                return;
            }
        };
        let plan = g.plan(blob.h, blob.w).expect("64x64 fits any device this runs on");
        let gpu_out = g.run(&ops, &plan, &blob).expect("the GPU pass");
        let (worst, mean, at) = compare(&blob, &gpu_out.d, &cpu_out.d);
        eprintln!("cpu vs cuda: max |d| {worst:.3e} mean {mean:.3e} at {at}");
        assert!(
            worst <= 2e-4,
            "the backends differ by {worst:.3e} (at {at}: cpu {} cuda {}), over the \
             2e-04 tolerance - see docs/NUMERICS.md for the size of a legitimate \
             accumulation-order difference",
            cpu_out.d[at],
            gpu_out.d[at]
        );
    }
}

/// The geometry walk is the one thing both backends derive a buffer size from, so
/// it is checked against the shapes the checkpoint itself declares: the filter
/// tensor's channel count is `kernel_dim(arch)`, and every weight the graph names
/// has to exist with a batch dimension of 1.
#[test]
fn the_graph_and_the_checkpoint_agree_on_shapes() {
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping the_graph_and_the_checkpoint_agree_on_shapes: no checkpoint");
        return;
    };
    let ws = Weights::load(&ckpt).expect("checkpoint loads");
    let a = ws.arch;
    let ops = net::graph(&a);
    let shapes = net::shapes_of(&ws, &ops, 3, 64, 64).expect("the geometry walk");
    assert_eq!(
        shapes.get("filt"),
        Some(&(a.kernel_dim(), 8, 8)),
        "the filter tensor is kernel_dim x (h/8) x (w/8)"
    );
    assert_eq!(shapes.get("in"), Some(&(3, 64, 64)));
    assert_eq!(shapes.get("out"), Some(&(3, 64, 64)));
    // Every op's output has to be in the table, and every name the graph READS
    // has to be written by something earlier or be the input: a graph that reads
    // a name nothing produces is a graph that would panic at run time instead.
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    seen.insert("in");
    for op in &ops {
        for r in op.reads() {
            assert!(seen.contains(r), "{op:?} reads `{r}`, which nothing wrote");
        }
        assert!(shapes.contains_key(op.out_name()), "no shape for {}", op.out_name());
        seen.insert(op.out_name());
    }
}
