//! IFAN: what a CPU pass will need, and what there is.
//!
//! IFAN'S MEMORY SHAPE IS UNUSUAL AND THIS MODULE EXISTS BECAUSE OF IT. Every
//! other engine in the family has a footprint that grows the way an image does:
//! activations, at most a few planes per level. IFAN has the F head, whose
//! output is `kernel_dim` values PER 1/8-RESOLUTION PIXEL - 15232 of them - and
//! that tensor alone (1.84 GiB at 1080p in f32) is bigger than everything else
//! in the pass put together.
//!
//! WHAT IS MODELED, AND HOW. `CpuPlan::of` SIMULATES the pooled allocator that
//! `net::Cpu` runs: it walks the same op list, keeps the same live set from
//! `Liveness`, hands out the same best-fit recycled buffer, and takes the peak of
//! `live + pool` over the pass. A hand-written formula beside the interpreter
//! drifts from it by a factor of three in either direction without anything
//! noticing - optimistic while the interpreter holds every buffer to the end,
//! pessimistic by the same factor once it recycles - and each drift is a refusal
//! that is wrong or a pass that gets killed.
//!
//! The pool is a first-fit-by-best-fit list of whole `Blob`s on the host side, so
//! the simulation tracks SIZES rather than devices: no allocator internals are
//! involved, and the only rounding left is the host allocator's own (see
//! `MEASURED_RATIO`).
//!
//! WHAT THIS IS NOT. It is arithmetic, not a measurement: a `Vec<f32>`'s real
//! cost depends on the allocator's rounding and on how much of it the kernel has
//! touched, so the peak here is a model scaled by `SLACK` and compared against
//! `MemAvailable`. That is a REFUSAL GUARD, not a memory report: it exists so a
//! pass that cannot fit says so in a sentence instead of being killed by the
//! kernel. The GPU side can afford to be exact - every byte there is one the
//! engine allocated - and it is; this side cannot, and says so.
//!
//! THERE IS ONE LAYOUT. A banded strategy - the filter encoder and IAC over one
//! band of 1/8-scale rows at a time - is the obvious way to fit an image this
//! engine otherwise refuses, and it is NOT offered here, because `net::Cpu` has no
//! band loop to run it: a strategy field that only prints which strategy it chose
//! would let a caller believe a pass had been made to fit. Model only what runs,
//! and refuse rather than promise a layout that does not exist.

use crate::weights::Weights;

/// The factor between the modelled peak and the real peak RSS.
///
/// A `Vec<f32>`'s cost is the allocator's, not the arithmetic's: it rounds every
/// allocation up, and a page that is written but never read still has to be
/// mapped. MEASURED HERE, not inherited: fitting `rss = a * modelled_peak + b`
/// over 64x64 (modelled 8.5 MiB, measured 47.6 MiB RSS), 256x256 (135.8 MiB,
/// 181.5 MiB) and 720x720 (1.05 GiB, 1.14 GiB) gives a = 1.073 and b = 37 MiB. The
/// ratio is set at 1.10 - the measured 1.073 rounded up, since a ratio that is too
/// small refuses a pass that fits while one that is too large only makes the
/// refusal conservative in the direction that costs nothing but a smaller image.
/// The guard lands 3%, 9% and 21% above the measurement at those three sizes.
pub const MEASURED_RATIO: f64 = 1.10;

/// The fixed part of the footprint: the mmapped checkpoint (42 MiB, touched by
/// both backends), the process, and the weight arena the interpreter copies out of
/// it. Added AFTER the ratio, because it does not scale with the image. Measured
/// 37 MiB at the three sizes above - it is what lets the 64x64 model, whose
/// activations are 8.5 MiB, predict 47.6 MiB of RSS. Rounded up, since being
/// generous here costs accuracy of the refusal and never correctness.
pub const BASE: usize = 48 * 1024 * 1024;

/// What a CPU pass will need, as arithmetic, before it allocates any of it.
///
/// THE PEAK OF `live + pool` OVER THE PASS, from a simulation of `net::Cpu`'s own
/// allocator (see the module doc). The sizes come from `net::shapes_of`, the same
/// walk `gpu::plan_shapes` uses, so a name's footprint and the buffer an
/// interpreter makes for it cannot disagree.
pub struct CpuPlan {
    /// How many buffers the graph names.
    pub names: usize,
    /// The modelled peak resident size of the ACTIVATIONS, before the ratio and
    /// the base: the high-water mark of the live set plus whatever the pool was
    /// still holding at that moment.
    pub peak_live: usize,
    /// `peak_live * MEASURED_RATIO + BASE`: what the guard compares.
    pub peak: usize,
}

impl CpuPlan {
    /// The plan for a pass over `h`x`wd` CROPPED pixels.
    ///
    /// THE SIMULATION IS THE INTERPRETER'S OWN RULE, deliberately: the same op
    /// list, the same `Liveness`, the same best-fit recycle, the same release at
    /// the end of the op that reads a name last, and the same exclusion of the
    /// op's own output (a name an op both reads and writes - a ResBlock's `m` -
    /// dies at that op and must not be recycled in it). Anything simpler has
    /// already been wrong twice; see the module doc.
    pub fn of(w: &Weights, h: usize, wd: usize) -> Result<CpuPlan, String> {
        let ops = crate::net::graph(&w.arch);
        let shapes = crate::net::shapes_of(w, &ops, 3, h, wd)?;
        let live = crate::net::Liveness::of(&ops);
        let bytes = |n: &str| -> usize {
            shapes
                .get(n)
                .map(|(c, h, wd)| c * h * wd * 4)
                .unwrap_or(0)
        };
        // name -> the size of the buffer currently held for it.
        let mut held: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        // Sizes of the buffers waiting to be recycled, best-fit by size.
        let mut pool: Vec<usize> = Vec::new();
        let mut live_bytes = 0usize;
        let mut pool_bytes = 0usize;
        let mut peak = 0usize;
        // `in` is the caller's buffer, present from the start.
        held.insert("in", bytes("in"));
        live_bytes += bytes("in");
        peak = peak.max(live_bytes + pool_bytes);
        for (i, op) in ops.iter().enumerate() {
            let name = op.out_name();
            let need = bytes(name);
            // The interpreter takes a recycled buffer when one fits, otherwise it
            // allocates exactly the name's geometry. Either way the size that ends
            // up held for this name is what it will occupy.
            let take = pool
                .iter()
                .enumerate()
                .filter(|(_, s)| **s >= need)
                .min_by_key(|(_, s)| **s)
                .map(|(j, s)| (j, *s));
            match take {
                Some((j, s)) => {
                    pool.swap_remove(j);
                    pool_bytes -= s;
                    held.insert(name, s);
                    live_bytes += s;
                }
                None => {
                    held.insert(name, need);
                    live_bytes += need;
                }
            }
            peak = peak.max(live_bytes + pool_bytes);
            // Release every name whose last reader was this op or earlier, except
            // this op's own output - the rule `net::Cpu::run_to` applies.
            let dead: Vec<&str> = held
                .keys()
                .filter(|n| **n != name && live.dies_after(n) <= i)
                .copied()
                .collect();
            for n in dead {
                if let Some(s) = held.remove(n) {
                    live_bytes -= s;
                    pool_bytes += s;
                    pool.push(s);
                }
            }
            peak = peak.max(live_bytes + pool_bytes);
        }
        Ok(CpuPlan {
            names: shapes.len(),
            peak_live: peak,
            peak: (peak as f64 * MEASURED_RATIO) as usize + BASE,
        })
    }
}

/// What the kernel says is available for a new allocation, in bytes.
///
/// `MemAvailable` AND NOT `MemFree`: available counts reclaimable page cache,
/// which on a machine that has just read a 42 MiB checkpoint and mapped it is a
/// large fraction of the difference - guarding on `MemFree` would refuse passes
/// that fit comfortably.
///
/// `None` when the file is unreadable, and the caller then does not guard at
/// all: refusing every pass because a `/proc` line moved is worse than running
/// one that might swap.
pub fn host_available() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        // A `?` here would return from the whole function on the first
        // non-matching line, and `/proc/meminfo`'s first line is `MemTotal:` -
        // so the guard would silently never fire. A missing key is a
        // `continue`, and this loop is the reason that distinction is written
        // down.
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let kb: usize = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        return Some(kb * 1024);
    }
    None
}

/// Plan a CPU pass, or refuse with both numbers.
///
/// ONE LAYOUT, and it is the one `net::Cpu` runs. There is no banded rung to fall
/// back to: it was never implemented, and offering a layout the interpreter
/// cannot execute is how a refusal turns into a run that swaps.
pub fn plan_cpu(w: &Weights, h: usize, wd: usize) -> Result<CpuPlan, String> {
    let plan = CpuPlan::of(w, h, wd)?;
    // With no readable `MemAvailable` there is nothing to compare against, so
    // run: guarding on a number we could not read would refuse passes on
    // machines whose /proc is odd.
    let Some(avail) = host_available() else {
        return Ok(plan);
    };
    if plan.peak <= avail {
        return Ok(plan);
    }
    Err(format!(
        "{}x{} does not fit in host memory: the pass peaks at {} of graph buffers, which is {} \
         with the measured {:.2}x allocator factor and a {} base; MemAvailable is {}",
        wd,
        h,
        fmt_bytes(plan.peak_live),
        fmt_bytes(plan.peak),
        MEASURED_RATIO,
        fmt_bytes(BASE),
        fmt_bytes(avail)
    ))
}

/// Bytes as a human would say them, for the plan line and refusals. Binary units
/// with one decimal, which is what the family's other engines print.
pub fn fmt_bytes(n: usize) -> String {
    const KIB: f64 = 1024.0;
    let f = n as f64;
    if f < KIB * KIB {
        format!("{:.1} KiB", f / KIB)
    } else if f < KIB * KIB * KIB {
        format!("{:.1} MiB", f / (KIB * KIB))
    } else {
        format!("{:.2} GiB", f / (KIB * KIB * KIB))
    }
}
