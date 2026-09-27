//! The GPU path: one weight arena, activation buffers taken from a pool, and the
//! graph from `net.rs` expressed as kernel launches.
//!
//! WHY THE PLANNER IS SEPARATE FROM THE ALLOCATOR. `PlanShape` is pure
//! arithmetic over the cropped geometry, checked against free VRAM before a single
//! byte is allocated, so an impossible pass is refused with exact numbers instead
//! of exhausting the card halfway through. The number the planner models is the
//! LIVE activation set, and the allocator now matches it: buffers are created on
//! first use and returned to a pool when their last reader has run, so the peak
//! allocation is that same set rather than the sum over every name in the graph.
//! `describe` and `describe_actual` print the two. For a whole pass they are the
//! SAME number, and a gap between them there is a bug in one of them; a
//! `--dump-op` run stops at the named activation, so its real allocation is
//! legitimately smaller than the plan for the full pass.
//!
//! THE WEIGHT ARENA. All 158 tensors are uploaded once into a single `DevBuf`
//! and addressed by name through a table, rather than one `DevBuf` per tensor.
//! One allocation of 42 MiB is a single driver call instead of 158, and - more
//! importantly - the whole set is contiguous and immutable, so no kernel launch
//! can invalidate another's pointer. The safetensors reader reports ABSOLUTE
//! file offsets, so each upload is one `copy_htod` of `raw(name)`.
#![cfg(feature = "cuda")]

use std::cell::RefCell;
use std::collections::HashMap;

use lightgpu::ffi::CUdeviceptr;
use lightgpu::vm::{self, Args, DevBuf, Event, Launch};

use crate::cuda::Kernels;
use crate::memguard::fmt_bytes;
use crate::net::{Blob, Op};
use crate::weights::Weights;

/// Threads per block for the elementwise/1-per-output kernels. 256 is the
/// family's default: measured on the same class of card, 128 leaves the SM
/// under-occupied on the convolutional grid and 512 starts losing to occupancy
/// limits on the 3x3 kernels (32 registers x 512 threads is at the edge).
const BLOCK: u32 = 256;

/// The activation slope every conv and every IAC iteration uses.
///
/// A named constant rather than a literal at six call sites: it is a property of
/// the MODEL (upstream hardcodes `LeakyReLU(0.1)` everywhere), and `Op::LeakyRelu`
/// exists precisely so the one place that is not a conv can carry its own value.
const ACT_SLOPE: f32 = 0.1;

/// A device buffer, or - during a DRY pass - the record of the buffer that would
/// have been allocated.
///
/// WHY A DRY PASS MUST NOT ALLOCATE. `plan` measures the pass by running it with
/// the launches suppressed, so that the admission decision is made on the same
/// arithmetic that will run rather than on a formula kept in step by hand. If the
/// measuring pass also ALLOCATES, three things go wrong at once: the measurement
/// itself can fail with a raw driver error on a card that is busy (so `plan`'s own
/// refusal message never gets to print), the measurement's buffers stay on the card
/// and fragment it for the real pass (a 1.84 GiB `filt` at 1080p then fails to find
/// room), and `describe`/`describe_actual` cannot be the same number. With `real`
/// false the walk allocates nothing, holds nothing and frees nothing, while taking
/// exactly the same decision path with exactly the same sizes.
struct Buf {
    /// `None` in a dry pass. Holding it in an `Option` is what makes the real
    /// case free on drop - `DevBuf` releases in its `Drop` impl - without a second
    /// code path for the dry one.
    dev: Option<DevBuf>,
    /// The buffer's size, which in a dry pass is the ONLY thing that exists and is
    /// therefore the only thing the accounting can use.
    bytes: usize,
}

impl Buf {
    /// A real, ZEROED device buffer. Zeroing matters: the elementwise kernels read
    /// exactly what they are given, and a launch that reads a plane through a
    /// padded or partial path would otherwise see a previous activation's values.
    fn zeros(bytes: usize) -> Result<Buf, String> {
        let b = DevBuf::zeros(bytes.max(1))?;
        Ok(Buf { bytes: b.bytes, dev: Some(b) })
    }

    /// The same buffer, as arithmetic only.
    fn dry(bytes: usize) -> Buf {
        Buf { dev: None, bytes: bytes.max(1) }
    }

    /// An UNINITIALISED real buffer, for scratch that the launch fills before
    /// anything reads it.
    fn alloc(bytes: usize) -> Result<Buf, String> {
        let b = DevBuf::alloc(bytes.max(1))?;
        Ok(Buf { bytes: b.bytes, dev: Some(b) })
    }

    fn ptr(&self) -> CUdeviceptr {
        match &self.dev {
            Some(b) => b.ptr,
            None => 0,
        }
    }

    /// Zero it in place. A no-op in a dry pass, where there is nothing to zero.
    fn zero(&self) -> Result<(), String> {
        match &self.dev {
            Some(b) => vm::memset_d8(b.ptr, b.bytes),
            None => Ok(()),
        }
    }

    /// Upload host floats, ignoring the request in a dry pass: the pass computes
    /// nothing, so nothing needs to be there.
    fn upload(&self, v: &[f32]) -> Result<(), String> {
        match &self.dev {
            Some(b) => b.upload(v),
            None => Ok(()),
        }
    }

    fn download(&self, out: &mut [f32]) -> Result<(), String> {
        match &self.dev {
            Some(b) => b.download(out),
            None => Ok(()),
        }
    }
}

/// One device buffer that other structures borrow by name.
struct Slot {
    buf: Buf,
}

/// The device side of one pass: the weight arena and the activation arena.
pub struct Gpu<'a> {
    k: Kernels,
    ws: &'a Weights,
    /// Name -> slot, for the buffers live at THIS point in the pass. Entries
    /// appear as `alloc` creates them and disappear when their last reader has run,
    /// so this is the live set itself rather than a record of everything the graph
    /// mentions.
    act: HashMap<String, Slot>,
    /// Bytes of device buffer created over this pass, counted when a `DevBuf` is
    /// made and never decremented.
    pub total_alloc: usize,
    /// NOT "bytes live right now". The pool keeps every buffer it has been given
    /// (it only ever hands one out again, and best-fit may hand a big one to a
    /// small request), so a buffer that a name has stopped using is still occupying
    /// the card and the footprint of a pass is the sum over every allocation it
    /// makes. That is the number `plan` measures, by running the pass with the
    /// launches suppressed (see `plan`), so the plan and the run are the same
    /// arithmetic rather than two models that have to be kept in step.
    /// Shapes and allocations only, no kernel launches: how `plan` measures the
    /// pass it is about to admit, using the very code that will run it.
    dry: bool,
    /// Every weight tensor, in one contiguous device region.
    wbuf: Option<Buf>,
    /// Tensor name -> (offset in ELEMENTS into `wbuf`, element count).
    wptr: HashMap<String, (usize, usize)>,
    /// Output shape (c, h, w) of every name in the graph, filled by `plan_shapes`.
    /// The interpreter derives these from the GRAPH rather than from the plan, and
    /// exec reads them back so a launch's dims and the buffer it writes can never
    /// disagree.
    shapes: HashMap<String, (usize, usize, usize)>,
    /// Released buffers, waiting to be reused. A field rather than a local of the
    /// interpreter's loop because `alloc` - which creates named activations - has
    /// to draw from it: if only scratch requests could reuse a freed buffer, every
    /// named activation would allocate afresh and the footprint would be the SUM
    /// over every name in the graph, which is what the plan exists to avoid.
    pool: Pool,
    /// PER-KERNEL DEVICE TIME, when `IFAN_TIME` is set in the environment.
    ///
    /// WHY THIS EXISTS. `--dump-op <name>` answers "how far into the pass is this
    /// activation" but its own figure is polluted twice over: every run carries
    /// the same ~0.2 s of fixed startup (driver init, both fatbins, the 42 MiB
    /// weight upload), and the run STOPPING at a name copies that tensor back to
    /// the host - which for `filt` at 1080p is 1.97 GB and therefore most of what
    /// its figure measures. Differencing two `--dump-op` figures to get one op's
    /// cost is thus arithmetic on two different quantities.
    ///
    /// `IFAN_TIME=1` instead wraps every launch in a pair of events and reports
    /// the device time of each KERNEL, summed over the pass, plus the D2D copies.
    /// It changes no arithmetic - `go` still launches exactly what it launched
    /// before, in the same order - so a run with it enabled and a run without it
    /// produce identical output, which is what makes it safe to leave in the
    /// binary. The cost is a synchronisation per launch, which is why it is off
    /// unless asked for: it serialises the queue, so the SUM of the per-kernel
    /// times is slightly larger than the same pass's wall time.
    timing: bool,
    /// kernel name (or "d2d copy") -> (launches, milliseconds).
    ///
    /// Behind a `RefCell` because `go` takes `&self` and has to. Making it
    /// `&mut self` would ripple out through every helper that launches a kernel
    /// (`lrelu_into`, `copy_region`, and the elementwise ops that borrow two
    /// buffers at once), for a field that is written only when `IFAN_TIME` is set
    /// and that no caller ever reads while a launch is in flight.
    times: RefCell<HashMap<String, (usize, f64)>>,
    /// The event pair `go` reuses, created on the first timed launch. Kept in the
    /// struct rather than made per launch because `cuEventCreate` is a driver call
    /// and there are 191 ops per pass; two events, recorded and re-recorded, are
    /// the usual way to time a stream.
    ev: RefCell<Option<(Event, Event)>>,
}

/// The shapes a pass will use, before any of them exist.
///
/// ONE LAYOUT, NOT SEVERAL. The only layout the interpreter can run is the
/// whole-image f32 one, so that is the only one modelled: `plan` either fits it or
/// refuses with numbers. Offering a smaller layout - banding, or an f16 filter
/// tensor - means either an interpreter that can execute it or a plan line that is
/// a promise the kernels do not keep; a modelled layout that no kernel implements
/// is worse than no layout at all.
#[derive(Debug, Clone)]
pub struct PlanShape {
    pub h: usize,
    pub w: usize,
    /// The live activation set plus the filter tensor, in bytes.
    pub bytes: usize,
}

impl<'a> Gpu<'a> {
    /// Bring up the driver, load both fatbins and pre-flight every kernel. All
    /// three failures are reported with the reason rather than a bare driver
    /// error, because on this family a missing driver is the common case and
    /// `--device cpu` is the documented answer.
    pub fn new(ws: &'a Weights) -> Result<Gpu<'a>, String> {
        vm::init().map_err(|e| format!("CUDA driver: {e}"))?;
        let k = Kernels::load()?;
        k.check_all()?;
        Ok(Gpu {
            k,
            ws,
            act: HashMap::new(),
            total_alloc: 0,
            dry: false,
            wbuf: None,
            wptr: HashMap::new(),
            shapes: HashMap::new(),
            pool: Pool::new(),
            timing: std::env::var("IFAN_TIME").is_ok(),
            times: RefCell::new(HashMap::new()),
            ev: RefCell::new(None),
        })
    }

    /// How much VRAM a new allocation has to work with.
    pub fn free_vram(&self) -> Result<usize, String> {
        vm::free_vram()
    }

    /// Fit the whole-image pass against MEASURED free VRAM, or refuse with the
    /// two numbers.
    ///
    /// The cost it checks is not a formula: it is the pass itself, walked with the
    /// launches suppressed (`dry`), so a refusal and a successful run are decided by
    /// the same arithmetic - the pass either has room for the buffers it will really
    /// create, or it does not start. `&mut self` because the walk records the
    /// graph's shapes and uploads the weight arena, both of which the run then uses.
    pub fn plan(&mut self, h: usize, w: usize) -> Result<PlanShape, String> {
        let ops = crate::net::graph(&self.ws.arch);
        // The weight arena is uploaded first: the dry pass resolves weight POINTERS
        // the same way the real one does, and a plan that skipped this would be
        // measuring a pass that could not have run.
        self.upload_weights()?;
        // A shape-only input: the dry pass reads the geometry and launches nothing.
        let input = Blob { c: 3, h, w, d: Vec::new() };
        // MEASURED, NOT MODELLED. The pass is walked for real - the same `exec`,
        // the same `alloc`, the same pool and the same release rule - with only the
        // kernel launches suppressed. A formula for the footprint has to be kept in
        // step with the interpreter by hand and is wrong the moment either side
        // changes; this cannot disagree with the run, because it IS the run minus
        // the arithmetic. It costs one walk of the graph's 191 ops and allocates
        // NOTHING: a dry pass's `Buf`s are size records with a null device
        // pointer, which is what makes the measurement safe to take on a card
        // another tenant is using (see the `Buf` doc).
        self.dry = true;
        let measured =
            self.run_ops(&ops, &PlanShape { h, w, bytes: 0 }, &input, None).map(|_| self.total_alloc);
        self.dry = false;
        // RELEASE WHAT THE MEASUREMENT ITSELF ALLOCATED, before the real pass runs.
        // The dry pass allocates real device buffers (that is what makes it a
        // measurement rather than a model), and if they stayed in the pool they
        // would both inflate the footprint by the measured amount and FRAGMENT it,
        // so a large single buffer (`filt`) could fail to find room even though the
        // plan said the pass fits. Emptying the pool here means the real pass
        // repeats the dry pass's allocation sequence exactly, on a clean card, so
        // its footprint IS the number the plan admitted it on.
        self.act.clear();
        self.pool = Pool::new();
        self.total_alloc = 0;
        let peak = measured?;
        let p = PlanShape { h, w, bytes: peak };
        let avail = vm::free_vram()?;
        // Leave room for the driver's own allocations and the output copy. This
        // is a flat reserve rather than a fraction: the driver's overhead does
        // not scale with the image, so a fraction would under-reserve on small
        // inputs and over-reserve on large ones.
        const RESERVE: usize = 192 << 20;
        let budget = avail.saturating_sub(RESERVE);
        if p.bytes > budget {
            return Err(format!(
                "{}x{} does not fit in VRAM: the pass allocates {} of device buffer and {} \
                 is free ({} usable after a {} reserve). A smaller image is the only way \
                 through - this engine has no smaller layout to fall back on.",
                w,
                h,
                fmt_bytes(p.bytes),
                fmt_bytes(avail),
                fmt_bytes(budget),
                fmt_bytes(RESERVE)
            ));
        }
        Ok(p)
    }

    /// The plan, as one line for the run log.
    pub fn describe(&self, p: &PlanShape) -> String {
        format!(
            "whole-image: cropped {}x{}, f32 filter tensor over the whole 1/8 grid, {} of \
             device buffer measured",
            p.w,
            p.h,
            fmt_bytes(p.bytes)
        )
    }

    /// What the pass actually allocated, as one line, after it has run.
    ///
    /// Printed next to `describe` because the two numbers answer the same question
    /// twice: `describe` is what the dry pass measured before any kernel ran
    /// (against a clean card, since `plan` releases the measurement's own buffers)
    /// and this is what the real pass created afterwards. They are the same walk
    /// over the same graph from the same starting state, so the two should be equal
    /// and a gap means they diverged - the dry pass skipped or reordered something
    /// the real one does.
    pub fn describe_actual(&self) -> String {
        format!("{} of device buffer allocated", fmt_bytes(self.total_alloc))
    }
}


/// The device interpreter: the same `Vec<Op>` as the CPU twin, one launch per op.
///
/// LIVENESS-BASED BUFFER POOL. Allocating a `DevBuf` per graph name would hold the
/// graph's 171 buffers at once - several times what a pass actually needs, because
/// most of them die early (every encoder activation is consumed by the decoder
/// within a few dozen ops). Instead, `Liveness` walks the op list once to find each name's
/// birth (the index of the op that writes it) and its last use, and the
/// interpreter returns a buffer to a free list as soon as no later op can read
/// it. The filter tensor and `f1` are both large enough that this is a
/// requirement rather than a refinement: at 1080p the f32 filter tensor alone is
/// 1.9 GiB of an 8 GiB card.
pub struct Liveness {
    /// name -> (birth op index, last op index that reads it). A name never read is
    /// `usize::MAX`, i.e. live to the end (the graph's `out`).
    pub live: HashMap<String, (usize, usize)>,
}

impl Liveness {
    pub fn of(ops: &[Op]) -> Liveness {
        let mut live: HashMap<String, (usize, usize)> = HashMap::new();
        // `in` is the caller's buffer, alive from before op 0.
        live.insert("in".to_string(), (0, usize::MAX));
        for (i, op) in ops.iter().enumerate() {
            for r in op.reads() {
                // The birth of a read name is the current index if it has not been
                // written yet and this is the graph's input; otherwise leave it.
                live.entry(r.to_string())
                    .and_modify(|e| {
                        // Last use means the last op that reads it, but a name read
                        // by an op that also writes it (`Add { a, b, out }` where
                        // a == out) must stay alive through that op - hence `i + 1`,
                        // which frees it just after the op rather than during it.
                        e.1 = e.1.max(i);
                    })
                    .or_insert((i, i));
            }
            live.insert(op.out_name().to_string(), (i, usize::MAX));
        }
        // Second pass: fill in the true last use before any name is freed.
        let mut last: HashMap<&str, usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            for r in op.reads() {
                *last.entry(r).or_insert(0) = i;
            }
        }
        for (name, e) in live.iter_mut() {
            if let Some(l) = last.get(name.as_str()) {
                e.1 = *l;
            }
        }
        Liveness { live }
    }

    /// When a name may be released: after its last reading op.
    pub fn dies_after(&self, name: &str) -> usize {
        self.live.get(name).map(|e| e.1).unwrap_or(usize::MAX)
    }
}

/// A free-list of device buffers, best-fit by byte size.
///
/// Best-fit because the size classes here are wildly different (`f1` is 265 MiB
/// and `iac.o.0` is 16 MiB at 1080p): first-fit would happily put a 16 MiB
/// request into a 265 MiB hole and then fail the next 265 MiB request that could
/// have used it. `devbuf_range` is the underlying allocator; this only decides
/// which buffer to hand out.
struct Pool {
    free: Vec<Buf>,
}

impl Pool {
    fn new() -> Pool {
        Pool { free: Vec::new() }
    }

    /// The smallest free buffer that is at least `bytes` large, or `None`.
    fn take(&mut self, bytes: usize) -> Option<Buf> {
        let mut best: Option<usize> = None;
        for (i, b) in self.free.iter().enumerate() {
            if b.bytes >= bytes {
                match best {
                    Some(j) if self.free[j].bytes <= b.bytes => {}
                    _ => best = Some(i),
                }
            }
        }
        match best {
            Some(i) => Some(self.free.swap_remove(i)),
            None => None,
        }
    }

    fn give(&mut self, b: Buf) {
        self.free.push(b);
    }

}

/// The name of a 4-byte alignment helper: the toolkit's `lg_*` kernels address
/// `float*`, so every buffer handed to them must be 256-byte aligned at least.
/// `DevBuf::alloc` goes through `cuMemAlloc`, whose base alignment is 256 bytes,
/// which is why no manual padding appears anywhere below.
const _ALIGNMENT_FROM_CU_MEMALLOC: () = ();

impl<'a> Gpu<'a> {
    /// Run the graph on the device and return the final blob.
    ///
    /// `ops` must be the same vector the CPU interpreter runs. The returned `Blob`
    /// is a host copy; the device buffers are released as the pass proceeds and
    /// reused within it.
    pub fn run(&mut self, ops: &[Op], plan: &PlanShape, input: &Blob) -> Result<Blob, String> {
        self.run_ops(ops, plan, input, None)
    }

    /// Run the graph but STOP after the op that writes `stop`, returning that
    /// buffer's contents.
    ///
    /// The GPU counterpart of `net::Cpu::run_until`, and it exists for the same
    /// reason: when the two backends disagree at the output, the useful question
    /// is which op introduced the difference. Without it, localizing a GPU bug
    /// means reading kernels and guessing; with it, the same ladder of names the
    /// CPU twin is checked against (`e1a`, `f_c`, `filt`, `iac.o.0`, ...) works on
    /// the device. It stops at the FIRST op producing the name, which is what the
    /// CPU's `run_until` does too.
    pub fn run_until(
        &mut self,
        ops: &[Op],
        plan: &PlanShape,
        input: &Blob,
        stop: &str,
    ) -> Result<Blob, String> {
        self.run_ops(ops, plan, input, Some(stop))
    }

    /// The interpreter's loop, shared by `run` and `run_until`.
    ///
    /// ONE LOOP, because the two differ only in where they stop, and a second copy
    /// of the freeing rule is exactly how `--dump-op` and a normal run would come
    /// to disagree about when a buffer dies - and the dump is the tool used to
    /// debug the normal run.
    fn run_ops(
        &mut self,
        ops: &[Op],
        plan: &PlanShape,
        input: &Blob,
        stop: Option<&str>,
    ) -> Result<Blob, String> {
        // A fresh pass: the previous run's live names and the count start at zero.
        // The POOL is deliberately NOT cleared - its buffers are still allocated on
        // the device and are exactly what the next pass should reuse.
        self.act.clear();
        self.total_alloc = 0;
        if !self.dry {
            self.upload_weights()?;
        }
        // Shapes first (from the graph, never from the plan), so `alloc` knows what
        // to allocate and `check_elems` has something to compare against.
        self.plan_shapes(ops, input)?;
        let live = Liveness::of(ops);
        // `in` is the caller's buffer: a graph name like any other, allocated on
        // first use like any other, uploaded once.
        self.alloc("in")?;
        if !self.dry {
            let slot = self.act.get("in").ok_or("no `in` buffer")?;
            slot.buf.upload(&input.d)?;
        }
        for (i, op) in ops.iter().enumerate() {
            // A READER must find its buffer: `alloc` creates buffers on demand,
            // which is right for a write target and would silently hand a reader a
            // zeroed buffer for a name nothing ever wrote. The CPU twin panics on
            // that ("graph reads `x` before it is written") and this is the same
            // check, made before the launch rather than after the fact.
            for r in op.reads() {
                if self.act.get(r).is_none() {
                    return Err(format!(
                        "op {i} {op:?} reads `{r}` before it is written"
                    ));
                }
            }
            self.exec(op, i)?;
            // What the caller asked to look at, before anything is released.
            if stop == Some(op.out_name()) {
                vm::sync()?;
                let s = self
                    .act
                    .get(op.out_name())
                    .ok_or_else(|| format!("no device buffer `{}`", op.out_name()))?;
                let (c, h, w) = self.shape(op.out_name())?;
                let mut host = vec![0.0f32; c * h * w];
                s.buf.download(&mut host)?;
                vm::sync()?;
                return Ok(Blob { c, h, w, d: host });
            }
            // Release everything whose last reader was this op or earlier. `<=`
            // rather than `<`: `dies_after` is the index of the op that READS the
            // name last, and once that op has run the buffer is dead. A name that
            // is never read keeps `usize::MAX` and is never released - that is the
            // graph's `out`.
            let done: Vec<String> = self
                .act
                .keys()
                .filter(|n| live.dies_after(n) <= i)
                .cloned()
                .collect();
            for n in done {
                if let Some(s) = self.act.remove(&n) {
                    // The buffer goes back to the pool, not back to the driver: it
                    // is still allocated, and the next name of that size or smaller
                    // will reuse it.
                    self.pool.give(s.buf);
                }
            }
        }
        if let Some(name) = stop {
            return Err(format!("graph has no op producing `{name}`"));
        }
        if self.dry {
            // Nothing was computed, so there is nothing to return; the caller is
            // `plan`, which only wants the byte count.
            return Ok(Blob { c: 3, h: plan.h, w: plan.w, d: Vec::new() });
        }
        let out = self.act.remove("out").ok_or("graph produced no `out`")?;
        let mut host = vec![0.0f32; input.len()];
        out.buf.download(&mut host)?;
        vm::sync()?;
        self.report_times();
        Ok(Blob { c: 3, h: plan.h, w: plan.w, d: host })
    }

    /// Every weight tensor into one contiguous device buffer, once.
    ///
    /// One `cuMemAlloc` of 42 MiB instead of 158, and - the reason it matters
    /// more than the syscall count - the whole set is a single immutable region,
    /// so no later allocation can invalidate a pointer a kernel is using. The
    /// name -> offset table is built here and used by `wptr`.
    fn upload_weights(&mut self) -> Result<(), String> {
        if self.wbuf.is_some() {
            return Ok(());
        }
        let mut total = 0usize;
        for (name, shape) in crate::weights::SHAPES {
            let n: usize = shape.iter().product();
            total += n;
            self.wptr.insert(name.to_string(), (total - n, n));
        }
        // A dry pass builds the SAME ARENA as arithmetic. It has to: the helpers
        // resolve weight pointers on the way past (`wptr`), and a plan that could
        // not resolve them would be measuring a pass that could not have run.
        self.wbuf = Some(if self.dry { Buf::dry(total * 4) } else { Buf::zeros(total * 4)? });
        if self.dry {
            return Ok(());
        }
        for (name, (off, n)) in self.wptr.iter() {
            let src = self.ws.f32(name);
            debug_assert_eq!(src.len(), *n, "shape table disagrees with the file for {name}");
            let dst = self.wbuf.as_ref().expect("the arena was just built");
            // The arena is one allocation, so a tensor is an offset into it.
            // `devbuf_range` is only a query (it reports a pointer's size), so the
            // sub-pointer is arithmetic: base + byte offset.
            let at = dst.ptr() + (off * 4) as CUdeviceptr;
            // `copy_htod` is a raw byte copy, so the f32 slice is viewed as
            // bytes rather than converted: the source of truth is the
            // safetensors payload, and any conversion here would silently
            // change the weights.
            let bytes = unsafe {
                std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4)
            };
            vm::copy_htod(at, bytes)?;
        }
        Ok(())
    }

    /// A device pointer to a filter-tensor channel block by graph name.
    ///
    /// The filter tensor is `[15232][h8][w8]` contiguous, so a block's pointer is
    /// `base + block_start * h8 * w8` floats. The rule for WHICH block lives in
    /// `net::tap_range` and is shared with the CPU twin - the ordering is the
    /// checkpoint's `torch.split` order and is the single most likely way for the
    /// two backends to quietly disagree.
    fn tap_slice(&mut self, name: &str) -> Result<CUdeviceptr, String> {
        let (base, _count) = crate::net::tap_range(&self.ws.arch, name)?;
        let (_, fh, fw) = self.shape("filt")?;
        let step = (fh * fw * 4) as CUdeviceptr;
        // `filt` is an ACTIVATION (the F.3 convolution's output), not a weight, so
        // its pointer comes from the activation arena.
        let p = self.alloc("filt")?;
        let want = base as CUdeviceptr * step;
        // A filter block past the end would read whatever follows the tensor, so
        // it is refused here rather than trusted: `tap_range` and the checkpoint's
        // tensor are two separate sources for the same count.
        let total = self.shape("filt")?.0 as CUdeviceptr * step;
        let (_b, count) = crate::net::tap_range(&self.ws.arch, name)?;
        if want + count as CUdeviceptr * step > total {
            return Err(format!(
                "filter block `{name}` runs past the filter tensor: {} + {} planes beyond {}",
                base, count, total / step
            ));
        }
        Ok(p + want)
    }

    /// A device pointer to a weight by tensor name.
    fn wptr(&self, name: &str) -> Result<CUdeviceptr, String> {
        let (off, _) = self
            .wptr
            .get(name)
            .ok_or_else(|| format!("no weight tensor `{name}`"))?;
        let base = self.wbuf.as_ref().ok_or("weight arena not uploaded")?;
        Ok(base.ptr() + (off * 4) as CUdeviceptr)
    }
}


impl<'a> Gpu<'a> {
    /// The output shape of every name in the graph, from the graph itself.
    ///
    /// NOTHING IS ALLOCATED HERE, and that is the point. Shapes are derived from
    /// the graph because the interpreter knows each output's (c, h, w) from its
    /// inputs' - the same walk the kernels do, so a buffer can never disagree with
    /// the launch that fills it. Buffers are then allocated on FIRST USE by
    /// `alloc`, because the graph's 171 names exist over a pass while only a
    /// handful are live at once; allocating them all up front makes the real
    /// footprint the SUM over every name and turns the plan line into a promise
    /// the allocator does not keep. `PlanShape.bytes` is the live-set model, and
    /// with first-use allocation the allocator's peak is that same set, which is
    /// why `describe_actual` can be checked against it.
    fn plan_shapes(&mut self, ops: &[Op], input: &Blob) -> Result<(), String> {
        // A second run of the same `Gpu` reuses the driver and the weight arena but
        // not the activations: the shapes are per-image, and the buffers are gone.
        // THE WALK IS SHARED WITH THE CPU SIDE (`net::shapes_of`), which sizes the
        // host plan's buffers from the same function - so the two readings of the
        // geometry cannot drift apart.
        self.shapes = crate::net::shapes_of(self.ws, ops, input.c, input.h, input.w)?;
        Ok(())
    }

    /// A device pointer to a graph buffer, ALLOCATING IT ON FIRST USE.
    ///
    /// A write target that does not exist yet is exactly the graph's normal state,
    /// so this creates it; a READER always finds its buffer already there, because
    /// something wrote it earlier (the CPU twin has the same distinction in its
    /// `get`/`put`, and it panics on a read-before-write for the same reason).
    fn alloc(&mut self, name: &str) -> Result<CUdeviceptr, String> {
        if let Some(s) = self.act.get(name) {
            return Ok(s.buf.ptr());
        }
        let (c, h, w) = self.shape(name)?;
        let want = c * h * w * 4;
        // From the pool when a released buffer is big enough, otherwise a fresh
        // allocation. Recycling is what keeps the footprint near the LIVE set: the
        // interpreter releases a name once its last reader has run (`run_ops`), and
        // without reuse that release would save nothing at all.
        //
        // A RECYCLED BUFFER IS ZEROED. It holds a previous activation's values, and
        // the elementwise kernels (`lg_add`, `lg_lrelu`, `if_clip`) read exactly what
        // they are given - but a launch that reads a whole plane through a padded or
        // partial path would see stale data, so the zeroing is what makes "the same
        // buffer, a different name" safe rather than lucky. The memset is skipped in
        // a dry pass, which has nothing to zero (`Buf::zero`).
        //
        // COUNTED ONCE, WHEN IT IS CREATED. A buffer handed back by the pool was
        // already counted the first time it was made, and adding its size again on
        // every reuse would inflate the total by the number of times it circulates -
        // which is what this did until the trace showed 171 "allocations" for a pass
        // that creates a few dozen distinct buffers.
        let b = match self.pool.take(want) {
            Some(b) => b,
            None => {
                let b = if self.dry { Buf::dry(want) } else { Buf::zeros(want)? };
                self.total_alloc += b.bytes;
                if std::env::var("IFAN_TRACE_ALLOC").is_ok() {
                    eprintln!(
                        "alloc {name} {} (total {}, free vram {:?})",
                        b.bytes,
                        self.total_alloc,
                        vm::free_vram().map(|b| b >> 20)
                    );
                }
                b
            }
        };
        b.zero()?;
        let p = b.ptr();
        self.act.insert(name.to_string(), Slot { buf: b });
        Ok(p)
    }
}

/// One launch of a 1-D elementwise kernel over `n` floats.
fn grid1d(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(BLOCK as usize) as u32, 1, 1)
}

/// The launch geometry of a tiled convolution: the output tile one block writes,
/// and the block that writes it.
///
/// THIS IS HALF OF THE KERNEL'S CONTRACT, with the other half in `cuda/ifan.cu`.
/// A mismatch does not fault and does not warn: every block that is not launched
/// simply never runs and the pixels it owned stay at whatever the buffer held, so
/// the two sides have to agree exactly.
///
/// The numbers were measured, not chosen; the sweeps that produced them, and the
/// one that measured faster than what is shipped, are in docs/INTERNALS.md.
#[derive(Clone, Copy)]
pub struct TileShape {
    /// Output columns per block (threads x times columns per thread).
    pub tw: usize,
    /// Output rows per block.
    pub th: usize,
    /// Output channels per block.
    pub oc: usize,
    /// The block: (x, y, 1).
    pub block: (u32, u32, u32),
}

/// `if_conv3x3_tile`: a 128x8 pixel tile of 8 channels, from a 32x8 block.
///
/// `pub` so the geometry can be checked against `cuda/ifan.cu` from outside this
/// module as well: the numbers below have to agree with that file exactly, and an
/// exported constant is one fewer place for them to disagree.
pub const G3_TILE: TileShape = TileShape { tw: 128, th: 8, oc: 8, block: (32, 8, 1) };
/// `if_conv1x1_tile`: the same pixel tile, 32 channels at a time, from a 32x4
/// block. Fewer rows per block leaves the room in the register file that the
/// wider channel tile needs.
pub const G1_TILE: TileShape = TileShape { tw: 128, th: 4, oc: 32, block: (32, 4, 1) };
/// `if_conv3x3s2_tile`: a 64x8 pixel tile of 16 channels, from a 32x8 block - the
/// same block as the stride-1 3x3, but half as many columns per thread because the
/// staged input region is TWICE the radius, and ONE input channel staged at a time
/// because that is what the sweep in `cuda/ifan.cu` picked: 9 KiB of shared memory
/// per block against 36 KiB for a wider tile, and 15.1 ms against 45.7 for the
/// same 19.1 GFLOP of work.
///
/// (`tw`/`th`/`oc` below are in OUTPUT coordinates, as the grid is.)
pub const G2_TILE: TileShape = TileShape { tw: 64, th: 8, oc: 16, block: (32, 8, 1) };
/// `if_convtranspose4x4s2p1_tile`: a 64x8 output tile of 8 channels from a 32x8
/// block. The tile is `tw`/`th` in OUTPUT coordinates, as every entry here is -
/// which for this kernel means the staged input region is only `TW/2 + 2` wide,
/// not the `(TW-1)*STRIDE + KW` a stride-2 convolution would need for the same
/// tile: a transposed convolution's output is twice as dense as its input. See
/// the kernel's own header for why its four taps cannot be expressed as one halo
/// origin the way the others' can.
pub const GT_TILE: TileShape = TileShape { tw: 64, th: 8, oc: 8, block: (32, 8, 1) };

/// The grid for `TileShape`: one block per output tile, in (x, y, channel) order.
fn tile_grid(h: usize, wd: usize, c_out: usize, t: TileShape) -> (u32, u32, u32) {
    (
        wd.div_ceil(t.tw) as u32,
        h.div_ceil(t.th) as u32,
        c_out.div_ceil(t.oc) as u32,
    )
}

impl<'a> Gpu<'a> {
    /// Refuse an op whose geometry disagrees with the graph's own.
    ///
    /// A launch reads and writes by POINTER and never sees a length, so a shape
    /// that drifted (an off-by-one in a level, a crop that did not happen) would
    /// corrupt memory rather than produce a wrong picture. Each helper computes the
    /// dims it is about to pass to a kernel and calls this with the name it is
    /// writing, so the two readings of the geometry - the graph's, from
    /// `plan_shapes`, and the helper's - have to agree before anything launches.
    ///
    /// This is compared against the SHAPE TABLE and not against the allocation
    /// size, because a buffer taken from the pool may be larger than the region it
    /// holds; and it is deliberately not checked for `name` being allocated yet,
    /// since a write target is created by `alloc` at the end of the same helper.
    fn check_elems(&self, name: &str, c: usize, h: usize, wd: usize) -> Result<(), String> {
        let (sc, sh, sw) = self.shape(name)?;
        if (sc, sh, sw) != (c, h, wd) {
            return Err(format!(
                "`{name}` is {sc}x{sh}x{sw} in the graph but this op would write {}x{}x{}",
                c, h, wd
            ));
        }
        Ok(())
    }

    fn shape(&self, name: &str) -> Result<(usize, usize, usize), String> {
        self.shapes
            .get(name)
            .copied()
            .ok_or_else(|| format!("no shape recorded for `{name}`"))
    }

    /// Launch a kernel, unless this is a dry pass.
    ///
    /// THE ONE PLACE `dry` AFFECTS ARITHMETIC. Every kernel in the interpreter goes
    /// through here, so a dry pass walks the graph, allocates the same buffers and
    /// applies the same release rule as a real one while computing nothing. That is
    /// what lets `plan` report a MEASUREMENT of the pass it is about to admit
    /// instead of a formula about it.
    fn go(
        &self,
        a: &mut Args,
        m: &lightgpu::vm::Module,
        kernel: &str,
        l: Launch,
    ) -> Result<(), String> {
        if self.dry {
            return Ok(());
        }
        if !self.timing {
            return a.launch(m, kernel, l);
        }
        // Timed: the whole point is to see the DEVICE time of this kernel without
        // the run's own fixed cost, so the events go around the launch and nothing
        // else. `t1.synchronize` is what makes the queue serial here, which is the
        // price of the measurement - see the `timing` field.
        let (t0, t1) = self.timing_events()?;
        t0.record()?;
        let r = a.launch(m, kernel, l);
        t1.record()?;
        t1.synchronize()?;
        self.add_time(kernel, t0.elapsed_ms(&t1)? as f64);
        r
    }

    /// The per-kernel device times, when `IFAN_TIME` was set, as one table.
    ///
    /// Printed from `run_ops` at the end of a REAL pass so the numbers describe the
    /// pass that just ran rather than a second one. Sorted by total time, because
    /// the question this exists to answer is "which kernel is the pass spending
    /// its time in" and an alphabetised table makes that the reader's job.
    fn report_times(&self) {
        let times = self.times.borrow();
        if !self.timing || times.is_empty() {
            return;
        }
        let mut v: Vec<(&String, &(usize, f64))> = times.iter().collect();
        v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = v.iter().map(|(_, (_, ms))| ms).sum();
        eprintln!("ifan: per-kernel device time (IFAN_TIME), {} launches, {total:.1} ms total", v.iter().map(|(_, (n, _))| n).sum::<usize>());
        eprintln!("ifan: {:<28} {:>6} {:>10} {:>8}", "kernel", "launch", "ms", "%");
        for (name, (n, ms)) in v {
            eprintln!(
                "ifan: {:<28} {:>6} {:>10.3} {:>7.1}%",
                name,
                n,
                ms,
                100.0 * ms / total
            );
        }
    }

    /// A scratch buffer with no graph name of its own, from the pool when one is
    /// available and fresh when not.
    ///
    /// Scratch is returned to the pool by the op that borrowed it, so it occupies
    /// the pool - and therefore the measured peak - for as long as the launch needs
    /// it, which is what makes the peak a real number rather than an estimate.
    fn scratch(&mut self, bytes: usize) -> Result<Buf, String> {
        match self.pool.take(bytes) {
            Some(b) => Ok(b),
            None => {
                let b = if self.dry { Buf::dry(bytes) } else { Buf::alloc(bytes)? };
                self.total_alloc += b.bytes;
                if std::env::var("IFAN_TRACE_ALLOC").is_ok() {
                    eprintln!("alloc <scratch {bytes}> (total {})", self.total_alloc);
                }
                Ok(b)
            }
        }
    }

    /// `out = leaky_relu(src, slope)` over `n` floats. `src` may be a scratch.
    ///
    /// `slope` is a parameter rather than a constant because `Op::LeakyRelu`
    /// carries one: taking 0.1 on faith here would compute a DIFFERENT function
    /// from the one the graph asked for, with no error anywhere. IFAN itself only
    /// ever uses 0.1, and that is a fact about the model, not about this kernel.
    fn lrelu_into(
        &self,
        src: CUdeviceptr,
        out: CUdeviceptr,
        n: usize,
        slope: f32,
    ) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(src).ptr(out).f32(slope).i64(n as i64);
        self.go(&mut a, &self.k.toolkit, "lg_lrelu", Launch::new(grid1d(n), (BLOCK, 1, 1)))
    }

    /// 3x3 stride 1 pad 1, optionally activated, by IFAN's TILED kernel.
    ///
    /// WHY NOT THE TOOLKIT'S `lg_conv3x3s1p1`. That kernel gives each thread one
    /// output element accumulated in one register, so its FMA chain is strictly
    /// serial and every link depends on a fresh load; measured on the 1080 it runs
    /// at 106 GFLOP/s on a 128-channel layer at 1/8 scale, against 2002 GFLOP/s
    /// for the same convolution through `if_conv3x3_tile` - and 88 against 857 on
    /// the 3->32 encoder convolution at 1080p, which is the graph's most expensive
    /// layer. The tiled kernel stages the input in shared memory, computes eight
    /// output channels per thread, and FUSES the activation, which removes a
    /// second launch and its scratch buffer for the 68 of 69 layers that have one
    /// (see `cuda/ifan.cu`).
    ///
    /// The output is written, not accumulated into. The ACCUMULATION ORDER is
    /// `(ci, ky, kx)` - the tiled kernel stages the input in shared memory and
    /// walks the channel tile outside the kernel window, because that is what lets
    /// a thread hold OC_TILE independent accumulators - which is NOT the CPU
    /// twin's `(ky, kx, ci)`. It also sums the zero-filled halo, so a boundary
    /// output adds terms the CPU twin skips. Both differences are in the last
    /// bits, bounded, and measured: docs/NUMERICS.md records the numbers, and a
    /// change that moves them is a change to be explained there rather than
    /// absorbed here.
    fn conv3x3(&mut self, x: &str, layer: &str, out: &str, act: bool) -> Result<(), String> {
        let (c_out, h, wd) = self.shape(out)?;
        let c_in = self.shape(x)?.0;
        // The graph's own geometry is the contract; the launch below is derived
        // from it and the kernel writes whatever the grid covers, so a mismatch
        // between the two would leave part of the plane unwritten rather than
        // fault.
        self.check_elems(out, c_out, h, wd)?;
        let xp = self.alloc(x)?;
        let wp = self.wptr(&format!("{layer}.weight"))?;
        let bp = self.wptr(&format!("{layer}.bias"))?;
        let outp = self.alloc(out)?;
        let mut a = Args::new();
        a.ptr(xp).ptr(wp).ptr(bp).ptr(outp)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(h as i32)
            .i32(wd as i32)
            .f32(ACT_SLOPE)
            .i32(act as i32);
        self.go(
            &mut a,
            &self.k.project,
            "if_conv3x3_tile",
            Launch::new(tile_grid(h, wd, c_out, G3_TILE), G3_TILE.block),
        )
    }

    /// 1x1, by IFAN's TILED kernel. Never activated: F's head is the only user.
    ///
    /// The tiled form matters more here than in the 3x3, in a different way. The
    /// arithmetic is one multiply-add per input channel per output element, so the
    /// cost is not the FMAs but the INPUT TRAFFIC: F.3 produces 15232 channels at
    /// 1/8 scale and every output-channel tile re-reads the whole input plane set,
    /// so an eight-channel tile reads the input 1904 times. Measured on the 1080,
    /// the toolkit's `lg_conv1x1` takes 1170 ms for that layer and the 32-channel
    /// tile takes 214 ms.
    fn conv1x1(&mut self, x: &str, layer: &str, out: &str) -> Result<(), String> {
        let (c_out, h, wd) = self.shape(out)?;
        let c_in = self.shape(x)?.0;
        let xp = self.alloc(x)?;
        let wp = self.wptr(&format!("{layer}.weight"))?;
        let bp = self.wptr(&format!("{layer}.bias"))?;
        let outp = self.alloc(out)?;
        self.check_elems(out, c_out, h, wd)?;
        let mut a = Args::new();
        a.ptr(xp).ptr(wp).ptr(bp).ptr(outp)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(h as i32)
            .i32(wd as i32)
            // The same kernel carries the activation, and this op has none.
            .f32(0.0)
            .i32(0);
        self.go(
            &mut a,
            &self.k.project,
            "if_conv1x1_tile",
            Launch::new(tile_grid(h, wd, c_out, G1_TILE), G1_TILE.block),
        )
    }

    /// 3x3 stride 2 pad 1, optionally activated, by the SAME tiled kernel as the
    /// stride-1 case with `STRIDE = 2`.
    ///
    /// THIS IS THE ENCODER'S DOWNSAMPLING: six of these run per pass, 95.6 GFLOP
    /// of the pass's 1881.6. One output element per thread means one accumulator
    /// and a strictly serial FMA chain, so what this needs - as much as the
    /// stride-1 convolutions do, and with a different answer - is a shared-memory
    /// input tile and independent accumulators per thread.
    ///
    /// The kernel takes the INPUT dims and derives the output grid as
    /// ceil(h/2) x ceil(w/2), so `oh`/`ow` from the shape table are used for the
    /// activation's element count and the two agreeing is checked below. As with
    /// the other tiled kernels, the output is written rather than accumulated
    /// into, and the activation is fused so no scratch buffer is needed.
    fn conv3x3s2(
        &mut self,
        x: &str,
        layer: &str,
        out: &str,
        act: bool,
    ) -> Result<(), String> {
        let (c_out, oh, ow) = self.shape(out)?;
        let (c_in, h, wd) = self.shape(x)?;
        if (h.div_ceil(2), wd.div_ceil(2)) != (oh, ow) {
            return Err(format!(
                "if_conv3x3s2_tile would produce {}x{} but `{out}` is {}x{}: the two \
                 geometries have drifted apart, which is a graph bug rather than a \
                 data condition",
                wd.div_ceil(2),
                h.div_ceil(2),
                ow,
                oh
            ));
        }
        self.check_elems(out, c_out, oh, ow)?;
        let xp = self.alloc(x)?;
        let wp = self.wptr(&format!("{layer}.weight"))?;
        let bp = self.wptr(&format!("{layer}.bias"))?;
        let outp = self.alloc(out)?;
        let mut a = Args::new();
        a.ptr(xp).ptr(wp).ptr(bp).ptr(outp)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(h as i32)
            .i32(wd as i32)
            .f32(ACT_SLOPE)
            .i32(act as i32);
        self.go(
            &mut a,
            &self.k.project,
            "if_conv3x3s2_tile",
            Launch::new(tile_grid(oh, ow, c_out, G2_TILE), G2_TILE.block),
        )
    }

    /// 4x4 stride 2 pad 1, ALWAYS activated (the upsamplers are), by IFAN's TILED
    /// transposed-convolution kernel.
    ///
    /// WHY IT WRITES STRAIGHT INTO `out` AND NOT INTO A SCRATCH. A scratch and a
    /// separate activation launch is what this op would look like if the
    /// activation were not fused into the kernel's epilogue, and the two buffers
    /// would be the SAME SIZE - so the pooled best-fit allocator can hand the
    /// scratch the `out` buffer it allocated one op earlier. `out` has no other
    /// reader, so such an alias is invisible to every check here, and the launch
    /// plus a separate `lg_lrelu` would deliver `act(act(conv))` where the graph
    /// asked for `act(conv)`: a power of the slope, elementwise, which is a
    /// plausible-looking image. Fusing the activation into the kernel removes the
    /// scratch and the second launch, and with them the correctness dependency on
    /// the pool not doing that. The arithmetic is the same `lrelu(conv + bias)`
    /// the separate pass would have produced.
    ///
    /// The grid is one block per output tile, so the kernel needs the INPUT dims
    /// (it derives `2h x 2wd` itself) and the two geometries are checked against
    /// each other here.
    fn convt4x4s2(&mut self, x: &str, layer: &str, out: &str) -> Result<(), String> {
        let (c_out, oh, ow) = self.shape(out)?;
        let (c_in, h, wd) = self.shape(x)?;
        if (h * 2, wd * 2) != (oh, ow) {
            return Err(format!(
                "if_convtranspose4x4s2p1_tile would produce {}x{} but `{out}` is {}x{}",
                wd * 2,
                h * 2,
                ow,
                oh
            ));
        }
        let xp = self.alloc(x)?;
        let wp = self.wptr(&format!("{layer}.weight"))?;
        let bp = self.wptr(&format!("{layer}.bias"))?;
        // The check that would have caught a transposed conv declaring the wrong
        // output channels (its weight is [c_in, c_out, 4, 4], not [c_out, ...]):
        // without it, the kernel indexes the weights with a wrong `c_out` and
        // produces finite nonsense instead of an error.
        self.check_elems(out, c_out, oh, ow)?;
        let outp = self.alloc(out)?;
        let mut a = Args::new();
        a.ptr(xp).ptr(wp).ptr(bp).ptr(outp)
            .i32(c_in as i32)
            .i32(c_out as i32)
            .i32(h as i32)
            .i32(wd as i32)
            .f32(ACT_SLOPE)
            .i32(1);
        self.go(
            &mut a,
            &self.k.project,
            "if_convtranspose4x4s2p1_tile",
            Launch::new(tile_grid(oh, ow, c_out, GT_TILE), GT_TILE.block),
        )
    }

    fn add(&mut self, a: &str, b: &str, out: &str) -> Result<(), String> {
        let (c, h, wd) = self.shape(out)?;
        let n = c * h * wd;
        let (ap, bp, op) = (self.alloc(a)?, self.alloc(b)?, self.alloc(out)?);
        let mut args = Args::new();
        args.ptr(ap).ptr(bp).ptr(op).i32(n as i32);
        self.go(&mut args, &self.k.toolkit, "lg_add", Launch::new(grid1d(n), (BLOCK, 1, 1)))
    }

    /// `out = src`, by device-to-device copy. Used by `Copy` and by `Cat2`.
    fn copy_region(
        &self,
        src: CUdeviceptr,
        dst: CUdeviceptr,
        bytes: usize,
    ) -> Result<(), String> {
        // One `cuMemcpyDtoD` rather than a kernel: it is the same transfer, and
        // it does not need a grid sized to the region. For the large stages this
        // is D2D DMA at full bandwidth - but it is DEVICE TIME like any kernel, so
        // the `IFAN_TIME` table accounts for it under one name. Leaving it out
        // would make the table's total short of the pass by however much the graph
        // spends in `Cat2` and `Copy`, and a table that does not add up is a table
        // nobody can use to decide where the time is.
        if self.dry {
            return Ok(());
        }
        if !self.timing {
            return vm::copy_d2d(dst, src, bytes);
        }
        let (t0, t1) = self.timing_events()?;
        t0.record()?;
        let r = vm::copy_d2d(dst, src, bytes);
        t1.record()?;
        t1.synchronize()?;
        self.add_time("d2d copy", t0.elapsed_ms(&t1)? as f64);
        r
    }

    /// The event pair every timed launch records around itself.
    ///
    /// Created on first use and kept: `cuEventCreate` is a driver call and there are
    /// 191 launches in a pass. The returned references borrow the `RefCell`, so the
    /// caller must not call this twice before using the result.
    fn timing_events(
        &self,
    ) -> Result<(std::cell::Ref<'_ , Event>, std::cell::Ref<'_, Event>), String> {
        let mut ev = self.ev.borrow_mut();
        if ev.is_none() {
            *ev = Some((Event::new()?, Event::new()?));
        }
        drop(ev);
        let a = std::cell::Ref::map(self.ev.borrow(), |e| &e.as_ref().expect("just created").0);
        let b = std::cell::Ref::map(self.ev.borrow(), |e| &e.as_ref().expect("just created").1);
        Ok((a, b))
    }

    /// Add one measurement to the table.
    fn add_time(&self, name: &str, ms: f64) {
        let mut times = self.times.borrow_mut();
        let e = times.entry(name.to_string()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ms;
    }

    /// `out = clip(x, 0, 1)`. The toolkit has no clip and the CPU twin clamps in
    /// floats, so `cuda/ifan.cu` carries `if_clip`: a 6 MiB round trip to the host
    /// and back would cost more than the kernel.
    fn clip(&mut self, x: &str, out: &str) -> Result<(), String> {
        let (c, h, wd) = self.shape(out)?;
        let n = c * h * wd;
        let (xp, op) = (self.alloc(x)?, self.alloc(out)?);
        let mut a = Args::new();
        a.ptr(xp).ptr(op).i64(n as i64);
        self.go(&mut a, &self.k.project, "if_clip", Launch::new(grid1d(n), (BLOCK, 1, 1)))
    }
}


impl<'a> Gpu<'a> {
    /// One op, one or more launches. `pool` supplies scratch that has no graph
    /// name of its own.
    fn exec(&mut self, op: &Op, i: usize) -> Result<(), String> {
        let r = match op {
            Op::Conv3x3 { x, w, out, act } => self.conv3x3(x, w, out, *act),
            Op::Conv1x1 { x, w, out } => self.conv1x1(x, w, out),
            Op::Conv3x3S2 { x, w, out, act } => self.conv3x3s2(x, w, out, *act),
            Op::ConvT4x4S2 { x, w, out } => self.convt4x4s2(x, w, out),
            Op::LeakyRelu { x, out, slope } => self.leaky_relu(x, out, *slope),
            Op::Sac1D { x, tap, out, ksize, vertical } => {
                self.sac1d(x, tap, out, *ksize, *vertical)
            }
            Op::SacBias { x, bias, out } => self.sac_bias(x, bias, out),
            Op::Add { a, b, out } => self.add(a, b, out),
            Op::Cat2 { a, b, out } => self.cat2(a, b, out),
            Op::Copy { x, out } => {
                // The byte count comes from the GRAPH's geometry, not from the
                // allocation's size: a buffer taken from the pool may be larger
                // than the region it holds (best-fit takes the first buffer that
                // is big enough), so `s.buf.bytes` would over-copy into `out`.
                let (c, h, wd) = self.shape(out)?;
                let (xc, xh, xw) = self.shape(x)?;
                if (c, h, wd) != (xc, xh, xw) {
                    return Err(format!(
                        "Copy: `{x}` is {xc}x{xh}x{xw} but `{out}` is {c}x{h}x{wd}"
                    ));
                }
                let (xp, op_) = (self.alloc(x)?, self.alloc(out)?);
                self.copy_region(xp, op_, c * h * wd * 4)
            }
            Op::Clip { x, out } => self.clip(x, out),
        };
        r.map_err(|e| format!("op {i} {op:?}: {e}"))
    }

    /// `out = leaky_relu(x, slope)`, tolerating `out == x`.
    ///
    /// `lg_lrelu` marks both pointers `__restrict__`, so passing it one buffer
    /// twice is undefined behaviour even though the elementwise update would be
    /// in-place-safe in plain C. When the names coincide the activation runs into
    /// a pool scratch and is copied back; otherwise it goes straight to `out`.
    fn leaky_relu(
        &mut self,
        x: &str,
        out: &str,
        slope: f32,
    ) -> Result<(), String> {
        let (c, h, wd) = self.shape(x)?;
        let n = c * h * wd;
        if x == out {
            let tmp = self.scratch(n * 4)?;
            let (xp, tp) = (self.alloc(x)?, tmp.ptr());
            let r = self.lrelu_into(xp, tp, n, slope).and_then(|()| {
                let op = self.alloc(out)?;
                self.copy_region(tp, op, n * 4)
            });
            self.pool.give(tmp);
            r
        } else {
            let (xp, op) = (self.alloc(x)?, self.alloc(out)?);
            self.lrelu_into(xp, op, n, slope)
        }
    }

    /// One direction of a SAC pass.
    ///
    /// `tap` names a slice of the filter tensor: `c * ksize` planes, channel-major
    /// (channel `ch`'s tap `k` at plane `ch * ksize + k`), which is exactly what
    /// `if_sac_vertical`/`if_sac_horizontal` index. The slice's base comes from
    /// `tap_range`, which encodes the checkpoint's `torch.split` order, and the
    /// kernel is handed the base pointer and its own eager indexing.
    fn sac1d(
        &mut self,
        x: &str,
        tap: &str,
        out: &str,
        ksize: usize,
        vertical: bool,
    ) -> Result<(), String> {
        let (c, h, wd) = self.shape(x)?;
        let n = c * h * wd;
        let tap_base = self.tap_slice(tap)?;
        let (xp, op) = (self.alloc(x)?, self.alloc(out)?);
        let kernel = if vertical { "if_sac_vertical" } else { "if_sac_horizontal" };
        let mut a = Args::new();
        a.ptr(xp).ptr(tap_base).ptr(op)
            .i32(c as i32).i32(h as i32).i32(wd as i32).i32(ksize as i32);
        self.go(
            &mut a,
            &self.k.project,
            kernel,
            Launch::new(grid1d(n), (BLOCK, 1, 1)),
        )
    }

    /// `f = f + f_bs[i]` and the activation that follows every IAC iteration.
    ///
    /// One fused kernel (`if_sac_bias`) rather than `lg_add` + `lg_lrelu`: the two
    /// always run back to back over the largest plane in the model, and this is
    /// the same fusion upstream's `f = f + f_bs[i]` then `leaky_relu` performs.
    /// The activation is applied for EVERY iteration, including the last, because
    /// that is what the released checkpoints do (`is_act_last=True`).
    fn sac_bias(&mut self, x: &str, bias: &str, out: &str) -> Result<(), String> {
        let (c, h, wd) = self.shape(x)?;
        let n = c * h * wd;
        // `bias` is a ch4-plane block of the filter tensor, one plane per channel
        // at the same h/w as the feature map - the same layout `if_sac_bias`
        // indexes (`b[ch * plane + ...]`).
        let bbase = self.tap_slice(bias)?;
        // `out` may be `x` (upstream's `f = f + f_bs[i]` is in-place): run into a
        // scratch and copy back, because the kernel would otherwise read and write
        // the same buffer with both pointers restricted.
        let direct = x != out;
        // The scratch is held as a `DevBuf` for the kernel's whole lifetime, not
        // reduced to a bare pointer: `DevBuf` FREES in `Drop`, so a pointer taken
        // from a dropped owner is a pointer into freed device memory that the pool
        // will hand out again. The owner must outlive the copy back. Note that this
        // branch is UNREACHABLE in the current graph (`x` is `iac.h.i` and `out` is
        // `iac.o.i`), so nothing here is covered by a run.
        let mut scratch = if direct { None } else { Some(self.scratch(n * 4)?) };
        let dst = match &scratch {
            Some(s) => s.ptr(),
            None => self.alloc(out)?,
        };
        let xp = self.alloc(x)?;
        let mut a = Args::new();
        // act = 1: every iteration is activated in this engine (see `Op::SacBias`),
        // and the slope is the model's (ACT_SLOPE), not a literal at this site.
        a.ptr(xp).ptr(bbase).ptr(dst).f32(ACT_SLOPE)
            .i32(c as i32).i32(h as i32).i32(wd as i32).i32(1);
        let r = self.go(&mut a, &self.k.project, "if_sac_bias", Launch::new(grid1d(n), (BLOCK, 1, 1)));
        match (r, scratch.take()) {
            (Ok(()), Some(s)) => {
                let op = self.alloc(out)?;
                let r = self.copy_region(s.ptr(), op, n * 4);
                self.pool.give(s);
                r
            }
            (r, None) => r,
            (Err(e), Some(s)) => {
                self.pool.give(s);
                Err(e)
            }
        }
    }

    /// `torch.cat([a, b], 1)` at the same h/w: two D2D copies into `out`.
    ///
    /// A copy rather than a kernel because the destination is contiguous and each
    /// source is already contiguous in `[c][h][w]`; `cuMemcpyDtoD` is the same
    /// cost as a strided kernel launch and has no indexing to get wrong.
    fn cat2(&mut self, a: &str, b: &str, out: &str) -> Result<(), String> {
        let (ac, h, wd) = self.shape(a)?;
        let (bc, bh, bw) = self.shape(b)?;
        if (bh, bw) != (h, wd) {
            return Err(format!(
                "Cat2 needs matching spatial dims, got {}x{} and {}x{}",
                h, wd, bh, bw
            ));
        }
        let (oc, oh, ow) = self.shape(out)?;
        if (oh, ow) != (h, wd) || oc != ac + bc {
            return Err(format!(
                "Cat2 output must be {}x{}x{}, allocated {}x{}x{}/{}",
                ac + bc, h, wd, oc, oh, ow, ac
            ));
        }
        let plane = h * wd * 4;
        let (ap, bp, op) = (self.alloc(a)?, self.alloc(b)?, self.alloc(out)?);
        self.copy_region(ap, op, ac * plane)?;
        self.copy_region(bp, op + (ac * plane) as CUdeviceptr, bc * plane)
    }
}
