//! The graph, recorded once and interpreted by both backends.
//!
//! WHY THE GRAPH IS DATA AND NOT CONTROL FLOW. The CPU twin and the CUDA path
//! have to walk the same operations in the same order. Written as two hand
//! -transcribed programs they would drift, and the only guard would be a numeric
//! comparison that says "something differs" without saying where. Written as one
//! `Vec<Op>` produced by `graph(arch)` and executed by two interpreters, a
//! GRAPH mistake is a mistake in both backends at once (so the golden comparison
//! against tools/reference.py catches it immediately, with the op index printed)
//! and only an INTERPRETER mistake is backend-specific.
//!
//! The op set is deliberately small and each op maps onto one loop here and onto
//! one kernel on the GPU side (`if_conv3x3_tile`, `if_conv1x1_tile`,
//! `if_conv3x3s2_tile`, `if_convtranspose4x4s2p1_tile`, the three SAC kernels,
//! `if_clip`, plus the toolkit's `lg_lrelu`, `lg_add` and `lg_copy`). Anything
//! more expressive - a fused op, a conditional - would give the two backends a way
//! to disagree about what it means, which is the thing this design exists to
//! prevent.
//!
//! THREE THINGS IN HERE LOOK LIKE MISTAKES AND ARE FAITHFUL:
//!
//!   1. SAC ignores `kernel2`. Both 1-D passes read the FIRST half of the filter
//!      tensor. Upstream `models/IAC.py:37` multiplies by `kernel1` twice where
//!      its own comment says `kernel2`; every released checkpoint was trained
//!      through that path, so a "fixed" engine would not match its weights.
//!   2. The LAST IAC iteration is activated (`is_act_last=True` upstream).
//!   3. `ResnetBlock` with `res_num > 1` adds a skip of the block's INPUT after
//!      the stem loop, not of the running value.
//!
//! All three are also written up in tools/reference.py, which is the authority.
use std::collections::HashMap;


use crate::kernels::{
    add_into_par, clip_into_par, conv1x1_into, conv3x3_into, conv3x3s2_into,
    convtranspose4x4s2_into, lrelu_in_place_par, lrelu_into_par, sac_bias_into_par,
    sac_pass_into_par,
};
use crate::weights::{Arch, Weights};

/// One NCHW activation: `c` planes of `h * w`.
#[derive(Clone)]
pub struct Blob {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub d: Vec<f32>,
}

impl Blob {
    pub fn zeros(c: usize, h: usize, w: usize) -> Blob {
        Blob { c, h, w, d: vec![0.0; c * h * w] }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }

    #[inline]
    pub fn plane(&self, c: usize) -> &[f32] {
        let hw = self.h * self.w;
        &self.d[c * hw..(c + 1) * hw]
    }

    #[inline]
    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let hw = self.h * self.w;
        &mut self.d[c * hw..(c + 1) * hw]
    }
}

/// The recorded graph.
///
/// Names are strings rather than indices because the two interpreters address
/// their buffers differently - the CPU one by a HashMap of `Blob`s, the CUDA one
/// by a HashMap of device pointers with the same keys - and because a name is
/// what a divergence report should print. The cost is a hash lookup per op
/// against several million FLOPs, which is not a thing to optimise.
/// A short label for an op's KIND, for the `IFAN_CPU_TIME` table.
///
/// The table aggregates by this, so it answers "how much of the pass is
/// convolutions, and how much is the SAC iteration" without the reader having to
/// know the 171 names in the graph.
fn op_kind(op: &Op) -> &'static str {
    match op {
        Op::Conv3x3 { .. } => "conv3x3",
        Op::Conv1x1 { .. } => "conv1x1",
        Op::Conv3x3S2 { .. } => "conv3x3s2",
        Op::ConvT4x4S2 { .. } => "convT4x4s2",
        Op::LeakyRelu { .. } => "lrelu",
        Op::Sac1D { vertical: true, .. } => "sac-vertical",
        Op::Sac1D { vertical: false, .. } => "sac-horizontal",
        Op::SacBias { .. } => "sac-bias",
        Op::Add { .. } => "add",
        Op::Cat2 { .. } => "cat2",
        Op::Copy { .. } => "copy",
        Op::Clip { .. } => "clip",
    }
}

/// The floating-point work of one op as `2 * MACs`, or zero for the elementwise
/// ops - which is what makes the `IFAN_CPU_TIME` table readable as a RATE and not
/// just a duration.
///
/// The convolution counts are from the op's own output shape and its input's
/// channel count, both taken from `shapes_of`'s table rather than from a
/// second opinion about the graph. The transposed convolution is counted with
/// all 16 kernel taps even though only four of them are ever active (its output
/// is twice as dense as its input), which OVERSTATES it by 4x - kept that way so
/// this table's totals can be compared with README.md's arithmetic breakdown,
/// which uses the same convention.
/// `(c_in, c_out, h, w)` of a convolution op, for the per-shape table.
fn op_geom(op: &Op, shapes: &HashMap<String, (usize, usize, usize)>) -> (usize, usize, usize, usize) {
    let x = match op {
        Op::Conv3x3 { x, .. }
        | Op::Conv3x3S2 { x, .. }
        | Op::Conv1x1 { x, .. }
        | Op::ConvT4x4S2 { x, .. } => x,
        _ => return (0, 0, 0, 0),
    };
    let c_in = shapes.get(x).map(|s| s.0).unwrap_or(0);
    let (c_out, h, wd) = shapes.get(op.out_name()).copied().unwrap_or((0, 0, 0));
    (c_in, c_out, h, wd)
}

fn op_gflop(op: &Op, shapes: &HashMap<String, (usize, usize, usize)>) -> f64 {
    let sh = |n: &str| shapes.get(n).copied().unwrap_or((0, 0, 0));
    let (x, taps) = match op {
        Op::Conv3x3 { x, .. } => (x, 9),
        Op::Conv3x3S2 { x, .. } => (x, 9),
        Op::Conv1x1 { x, .. } => (x, 1),
        Op::ConvT4x4S2 { x, .. } => (x, 16),
        _ => return 0.0,
    };
    let c_in = sh(x).0 as f64;
    let (c_out, h, wd) = sh(op.out_name());
    let (c_out, h, wd) = (c_out as f64, h as f64, wd as f64);
    2.0 * c_in * c_out * h * wd * (taps as f64) / 1e9
}

#[derive(Debug, Clone)]
pub enum Op {
    /// 3x3, stride 1, pad 1, `act` = leaky_relu(0.1) after.
    Conv3x3 { x: String, w: String, out: String, act: bool },
    /// 1x1. F's head only.
    Conv1x1 { x: String, w: String, out: String },
    /// 3x3, stride 2, pad 1.
    Conv3x3S2 { x: String, w: String, out: String, act: bool },
    /// 4x4, stride 2, pad 1, always activated.
    ConvT4x4S2 { x: String, w: String, out: String },
    /// `out = leaky_relu(x, slope)`.
    ///
    /// A separate op rather than a flag on the convs, because the toolkit's convs
    /// have no activation and the model applies leaky_relu in places a conv does
    /// not produce (after `x + stem(x)`, and after the SAC bias add). `out` MAY
    /// equal `x`: the CPU twin clones, and the GPU's dispatcher activates through
    /// a pool scratch and copies back, because `lg_lrelu` marks both pointers
    /// `__restrict__` and passing it one buffer twice would be undefined.
    LeakyRelu { x: String, out: String, slope: f32 },
    /// One 1-D SAC pass. `tap` is the filter blob name; `vertical` selects the
    /// axis; the tap index is the OUTPUT position and only the sample index is
    /// clamped. See the comment in cuda/ifan.cu for why this is two passes.
    Sac1D { x: String, tap: String, out: String, ksize: usize, vertical: bool },
    /// `f = f + bias_block` per channel, then leaky_relu(0.1). The two are one op
    /// because upstream applies the activation after every iteration with no
    /// exception, and splitting them would invite a backend to skip one.
    SacBias { x: String, bias: String, out: String },
    /// `out = a + b`.
    Add { a: String, b: String, out: String },
    /// Channel concat of exactly two inputs, each `ch4` channels, into `2*ch4`.
    /// The only concat this graph needs is `torch.cat([f, f_dm], 1)` in conv4_4;
    /// making it a general op would be one more place for the backends to
    /// disagree about a memory layout for no benefit.
    Cat2 { a: String, b: String, out: String },
    /// `out = x` (a copy through a named buffer). The graph needed a name for
    /// "the input again" - `out = clip(0.5*in + 0.5*res)` - and the alternative
    /// was a special case in each interpreter.
    Copy { x: String, out: String },
    /// `out = clip(x, 0, 1)`. Only the final op.
    Clip { x: String, out: String },
}

impl Op {
    /// The buffers this op READS, in order and with repeats allowed.
    ///
    /// One place, so `Liveness`, the CPU interpreter's missing-input diagnostic
    /// and any future pass agree on what an op needs. Weight names (`w`, the
    /// SAC `tap`, the `bias` block) are deliberately NOT here: they live in the
    /// immutable weight arena, outlive every activation and must never be
    /// released by a liveness pass that mistook them for intermediates.
    pub fn reads(&self) -> Vec<&str> {
        match self {
            Op::Conv3x3 { x, .. }
            | Op::Conv1x1 { x, .. }
            | Op::Conv3x3S2 { x, .. }
            | Op::ConvT4x4S2 { x, .. }
            | Op::LeakyRelu { x, .. }
            | Op::Sac1D { x, .. }
            | Op::SacBias { x, .. }
            | Op::Copy { x, .. }
            | Op::Clip { x, .. } => vec![x.as_str()],
            Op::Add { a, b, .. } | Op::Cat2 { a, b, .. } => vec![a.as_str(), b.as_str()],
        }
    }

    pub fn out_name(&self) -> &str {
        match self {
            Op::Conv3x3 { out, .. }
            | Op::Conv1x1 { out, .. }
            | Op::Conv3x3S2 { out, .. }
            | Op::ConvT4x4S2 { out, .. }
            | Op::LeakyRelu { out, .. }
            | Op::Sac1D { out, .. }
            | Op::SacBias { out, .. }
            | Op::Add { out, .. }
            | Op::Cat2 { out, .. }
            | Op::Copy { out, .. }
            | Op::Clip { out, .. } => out,
        }
    }
}

/// The graph, as `config_IFAN.py`'s `Network.forward` spells it.
///
/// The names are the checkpoint's own (`conv1_1.0.weight` is the tensor, `conv1_1`
/// is the layer), so a divergence report reads like the architecture.
pub fn graph(a: &Arch) -> Vec<Op> {
    let ch = a.ch;
    let ch4 = a.ch4;
    let mut g = Vec::with_capacity(96);
    let conv = |g: &mut Vec<Op>, x: &str, layer: &str, out: &str, act: bool| {
        g.push(Op::Conv3x3 { x: x.into(), w: layer.into(), out: out.into(), act });
    };
    let down = |g: &mut Vec<Op>, x: &str, layer: &str, out: &str| {
        g.push(Op::Conv3x3S2 { x: x.into(), w: layer.into(), out: out.into(), act: true });
    };
    // A ResnetBlock is EXPANDED here rather than kept as one op.
    //
    // `ResnetBlock.forward` is `for i in range(res_num): x = x + stem[i](x);
    // x = leaky_relu(x)`, then, when `res_num > 1`, `x = x + temp` where `temp`
    // is the block's INPUT and that last add is NOT activated. Every piece of
    // that is a primitive the graph already has, and expanding it here means:
    //
    //   * one implementation of the arithmetic, not two (a fused
    //     `Op::ResBlock` needs the CPU twin and the GPU dispatcher to agree tap
    //     for tap, and the GPU has no way to name a loop's intermediates);
    //   * the activation stays where upstream has it - inside the loop, on the
    //     residual sum, so it runs `res_num` times and the final `+ temp` is
    //     not activated;
    //   * `temp` is bound to the ORIGINAL input, not the running value.
    //
    // The expansion's intermediates carry `{layer}.{i}.{a,b,m}` names so the
    // liveness pass can size and release them like any other buffer.
    let res = |g: &mut Vec<Op>, x: &str, layer: &str, out: &str, n: usize| {
        let mut cur = x.to_string();
        for i in 0..n {
            let a = format!("{layer}.{i}.a"); // stem[i]'s first conv, activated
            let b = format!("{layer}.{i}.b"); // stem[i]'s second conv (not activated)
            let m = format!("{layer}.{i}.m"); // x + b, then activated
            // The checkpoint's names: `ResnetBlock.stem` is a ModuleList of
            // `Sequential(Conv2d, LeakyReLU, Conv2d)`, so a stem's convs are
            // `.stem.{i}.0` and `.stem.{i}.2` - the `.1` is the activation, which
            // has no tensor. These are the names in `weights::SHAPES`, i.e. the
            // checkpoint's own, and a typo here is a load-time panic rather than a
            // wrong picture.
            g.push(Op::Conv3x3 {
                x: cur.clone(),
                w: format!("{layer}.stem.{i}.0"),
                out: a.clone(),
                act: true,
            });
            g.push(Op::Conv3x3 {
                x: a,
                w: format!("{layer}.stem.{i}.2"),
                out: b.clone(),
                act: false,
            });
            g.push(Op::Add { a: cur.clone(), b, out: m.clone() });
            // leaky_relu(0.1) with no learnable parameter: the CPU twin folds it
            // into the conv/add that produced the value, and the GPU has
            // `lg_lrelu`. As a graph op it is `Op::LeakyRelu`.
            g.push(Op::LeakyRelu { x: m.clone(), out: m.clone(), slope: 0.1 });
            cur = m;
        }
        if n > 1 {
            g.push(Op::Add { a: cur, b: x.to_string(), out: out.to_string() });
        } else {
            g.push(Op::Copy { x: cur, out: out.to_string() });
        }
    };

    // ---- the image branch: encoder levels 0..3 ------------------------------
    conv(&mut g, "in", "conv1_1.0", "e1a", true);
    conv(&mut g, "e1a", "conv1_2.0", "e1b", true);
    conv(&mut g, "e1b", "conv1_3.0", "f1", true);
    down(&mut g, "f1", "conv2_1.0", "e2a");
    conv(&mut g, "e2a", "conv2_2.0", "e2b", true);
    conv(&mut g, "e2b", "conv2_3.0", "f2", true);
    down(&mut g, "f2", "conv3_1.0", "e3a");
    conv(&mut g, "e3a", "conv3_2.0", "e3b", true);
    conv(&mut g, "e3b", "conv3_3.0", "f3", true);
    down(&mut g, "f3", "conv4_1.0", "e4a");
    conv(&mut g, "e4a", "conv4_2.0", "e4b", true);
    conv(&mut g, "e4b", "conv4_3.0", "f_c", true);

    // ---- the filter branch: the same tower on its own weights, ALSO fed the
    //      image (`kconv1_1` takes 3 channels, not f1).
    conv(&mut g, "in", "kconv1_1.0", "k1a", true);
    conv(&mut g, "k1a", "kconv1_2.0", "k1b", true);
    conv(&mut g, "k1b", "kconv1_3.0", "k1c", true);
    down(&mut g, "k1c", "kconv2_1.0", "k2a");
    conv(&mut g, "k2a", "kconv2_2.0", "k2b", true);
    conv(&mut g, "k2b", "kconv2_3.0", "k2c", true);
    down(&mut g, "k2c", "kconv3_1.0", "k3a");
    conv(&mut g, "k3a", "kconv3_2.0", "k3b", true);
    conv(&mut g, "k3b", "kconv3_3.0", "k3c", true);
    down(&mut g, "k3c", "kconv4_1.0", "k4a");
    conv(&mut g, "k4a", "kconv4_2.0", "k4b", true);
    conv(&mut g, "k4b", "kconv4_3.0", "k", true);

    // ---- DME: a 1-channel map at 1/8, widened back to ch4 -------------------
    conv(&mut g, "k", "DME.0.0", "dm0", true);
    res(&mut g, "dm0", "DME.1", "dm1", a.res_num);
    res(&mut g, "dm1", "DME.2", "dm2", a.res_num);
    g.push(Op::Conv3x3 { x: "dm2".into(), w: "DME.3".into(), out: "dm".into(), act: false });
    conv(&mut g, "dm", "conv_DME.0", "f_dm", true);

    // ---- cat([k, f_dm]) and the 2*ch4 tower --------------------------------
    g.push(Op::Cat2 { a: "k".into(), b: "f_dm".into(), out: "cat".into() });
    conv(&mut g, "cat", "conv4_4.0.0", "c44a", true);
    res(&mut g, "c44a", "conv4_4.1", "c44b", a.res_num);
    res(&mut g, "c44b", "conv4_4.2", "c44c", a.res_num);
    conv(&mut g, "c44c", "conv4_4.3.0", "fc4", true);

    // ---- F: the filter tensor ----------------------------------------------
    conv(&mut g, "fc4", "F.0.0", "fa", true);
    res(&mut g, "fa", "F.1", "fb", a.res_num);
    res(&mut g, "fb", "F.2", "fc", a.res_num);
    g.push(Op::Conv1x1 { x: "fc".into(), w: "F.3".into(), out: "filt".into() });

    // ---- IAC: N iterations, each two 1-D passes + a bias + an activation ----
    // The two halves of the filter tensor are named once here; the slices are
    // what `Interp` reads. `f2` is deliberately unread (see the header).
    for i in 0..a.n {
        let k1 = format!("filt.k1.{i}");
        let _k2 = format!("filt.k2.{i}");
        let bs = format!("filt.bs.{i}");
        let sv = format!("iac.v.{i}");
        let sh = format!("iac.h.{i}");
        let so = format!("iac.o.{i}");
        let x = if i == 0 { "f_c".to_string() } else { format!("iac.o.{}", i - 1) };
        g.push(Op::Sac1D { x: x.clone(), tap: k1, out: sv.clone(), ksize: a.fs, vertical: true });
        g.push(Op::Sac1D {
            x: sv.clone(),
            tap: format!("filt.k1.{i}"),
            out: sh.clone(),
            ksize: a.fs,
            vertical: false,
        });
        g.push(Op::SacBias { x: sh, bias: bs, out: so });
    }
    let last_iac = format!("iac.o.{}", a.n - 1);

    // ---- conv_res: conv, a THREE-stem resblock, conv ------------------------
    conv(&mut g, &last_iac, "conv_res.0.0", "cr0", true);
    res(&mut g, "cr0", "conv_res.1", "cr1", 3);
    conv(&mut g, "cr1", "conv_res.2.0", "cr2", true);

    // ---- decoder: up, ADD THE SKIP, two resblocks --------------------------
    let up = |g: &mut Vec<Op>, x: &str, layer: &str, out: &str| {
        g.push(Op::ConvT4x4S2 { x: x.into(), w: layer.into(), out: out.into() });
    };
    up(&mut g, "cr2", "upconv3_u.0", "u3a");
    g.push(Op::Add { a: "u3a".into(), b: "f3".into(), out: "u3b".into() });
    res(&mut g, "u3b", "upconv3_1", "u3c", 1);
    res(&mut g, "u3c", "upconv3_2", "u3d", 1);

    up(&mut g, "u3d", "upconv2_u.0", "u2a");
    g.push(Op::Add { a: "u2a".into(), b: "f2".into(), out: "u2b".into() });
    res(&mut g, "u2b", "upconv2_1", "u2c", 1);
    res(&mut g, "u2c", "upconv2_2", "u2d", 1);

    up(&mut g, "u2d", "upconv1_u.0", "u1a");
    g.push(Op::Add { a: "u1a".into(), b: "f1".into(), out: "u1b".into() });
    res(&mut g, "u1b", "upconv1_1", "u1c", 1);
    res(&mut g, "u1c", "upconv1_2", "u1d", 1);

    conv(&mut g, "u1d", "out_res.0", "res", true);
    g.push(Op::Add { a: "res".into(), b: "in".into(), out: "raw".into() });
    g.push(Op::Clip { x: "raw".into(), out: "out".into() });
    debug_assert_eq!(ch * 4, ch4, "arch says ch={ch} ch4={ch4}");
    g
}


/// When each name in the graph is born and when it dies, from the op list.
///
/// THIS LIVES HERE, not in `gpu.rs`, because it is a property of the GRAPH and
/// both interpreters use it: the GPU releases device buffers with it and the CPU
/// interpreter (and `memguard`'s model of it) releases host ones. Two backends
/// with their own release rule is how one of them comes to hold a buffer its
/// consumer has already overwritten.
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
                live.entry(r.to_string()).or_insert((i, i));
            }
            live.insert(op.out_name().to_string(), (i, usize::MAX));
        }
        // THE LAST READ, which is what decides when a buffer may be recycled. A
        // name that is written again later (`m` in a ResBlock is written by its Add
        // and then read AND written by the LeakyRelu) keeps the OP INDEX OF ITS
        // LAST READ, not of its last write - releasing on the write would free a
        // buffer the same op is still reading.
        for (i, op) in ops.iter().enumerate() {
            for r in op.reads() {
                if let Some(e) = live.get_mut(r) {
                    e.1 = i;
                }
            }
        }
        Liveness { live }
    }

    /// When a name may be released: after its last reading op.
    pub fn dies_after(&self, name: &str) -> usize {
        self.live.get(name).map(|e| e.1).unwrap_or(usize::MAX)
    }
}

/// The output shape of every name in the graph, walked from the graph itself.
///
/// THE ONE PLACE BOTH BACKENDS LEARN A GEOMETRY, which is why it is a free
/// function here rather than a method on either interpreter: `gpu::plan_shapes`
/// calls it to size device buffers and `memguard::CpuPlan::of` calls it to size
/// host ones, so a name's footprint and the buffer an interpreter makes for it
/// cannot disagree. Deriving it from the GRAPH is the point - a table of shapes
/// written by hand has to be kept in step with `graph` and is wrong the moment
/// somebody adds an op, in a way that shows up as an out-of-bounds write rather
/// than as a compile error.
///
/// The output channel count comes from the checkpoint's own weight shape
/// (`nn.Conv2d` is `[c_out, c_in, kh, kw]`, `nn.ConvTranspose2d` is
/// `[c_in, c_out, kh, kw]` - the two are TRANSPOSED, and reading the wrong axis
/// gives a plausible power of two rather than an error), and the bias length is
/// checked against it because the bias is the one tensor whose length is
/// unambiguously the output count.
pub fn shapes_of(
    w: &Weights,
    ops: &[Op],
    c_in: usize,
    h: usize,
    wd: usize,
) -> Result<HashMap<String, (usize, usize, usize)>, String> {
    let out_ch = |layer: &str, transposed: bool| -> Result<usize, String> {
        let name = format!("{layer}.weight");
        let s = w.shape(&name)?;
        let c = if transposed { s[1] } else { s[0] };
        let bname = format!("{layer}.bias");
        let bias = w.shape(&bname)?[0];
        if c != bias {
            return Err(format!(
                "`{name}` says {c} output channels but `{bname}` has {bias}"
            ));
        }
        Ok(c)
    };
    let mut shapes: HashMap<String, (usize, usize, usize)> = HashMap::new();
    shapes.insert("in".to_string(), (c_in, h, wd));
    for op in ops.iter() {
        let get = |n: &str| -> Result<(usize, usize, usize), String> {
            shapes
                .get(n)
                .copied()
                .ok_or_else(|| format!("no shape for `{n}`"))
        };
        let out = match op {
            Op::Conv3x3 { x, w: layer, .. } | Op::Conv1x1 { x, w: layer, .. } => {
                let (_, h, wd) = get(x)?;
                (out_ch(layer, false)?, h, wd)
            }
            Op::Conv3x3S2 { x, w: layer, .. } => {
                let (_, h, wd) = get(x)?;
                (out_ch(layer, false)?, h.div_ceil(2), wd.div_ceil(2))
            }
            Op::ConvT4x4S2 { x, w: layer, .. } => {
                let (_, h, wd) = get(x)?;
                (out_ch(layer, true)?, h * 2, wd * 2)
            }
            Op::LeakyRelu { x, .. }
            | Op::Sac1D { x, .. }
            | Op::SacBias { x, .. }
            | Op::Add { a: x, .. }
            | Op::Copy { x, .. }
            | Op::Clip { x, .. } => get(x)?,
            Op::Cat2 { a, b, .. } => {
                let (ca, h, wd) = get(a)?;
                let (cb, hb, wb) = get(b)?;
                if (h, wd) != (hb, wb) {
                    return Err(format!("cat: {h}x{wd} vs {hb}x{wb}"));
                }
                (ca + cb, h, wd)
            }
        };
        shapes.insert(op.out_name().to_string(), out);
    }
    Ok(shapes)
}

/// The CPU interpreter: walk `Vec<Op>` against a set of named blobs.
///
/// This is the reference implementation of the graph, so it is written to be read
/// against tools/reference.py rather than to be fast. It is the same design as
/// `gpu::Gpu`: buffers for the graph's names are created on FIRST USE, taken from
/// a pool of recycled ones when a recycled buffer is big enough, and returned to
/// that pool once the last op that can read them has run.
///
/// WHY POOLED RATHER THAN A BLOB PER OP. Keeping every output in `blobs` for the
/// whole pass makes the footprint the SUM over all ~90 names rather than the live
/// set - 10.7 GiB against 3 GiB at 1080p, i.e. the same pass the GPU runs in a
/// third of the memory, and a CPU ceiling of 1080p on a 32 GiB machine where the
/// live set would allow 2560x1440. The buffers were already allocated; the waste
/// was entirely in holding them.
///
/// WHAT STILL WORKS. `--dump-op` runs the graph and stops at the op that produces
/// the name it was given (`run_to` returns there), so the value it prints is the
/// one that op wrote - for every name in the graph - and it needs no retention
/// rule to be correct. That is also what `Gpu::run_ops` has always done, so the
/// two backends' ladders are comparable stage for stage.
///
/// AND THE NUMBERS DO NOT MOVE, WHICH IS MEASURED RATHER THAN ARGUED. Reuse is
/// only of a buffer whose last reader has run, and a recycled buffer is ZEROED
/// (`out_buf`) because two kernels accumulate into their destination - without
/// that, the first stride-2 conv of the k-tower (`k2a`) came out 1.5e-1 from
/// torch on a 12-bit scale, which is how this was found. With it, all 169
/// activations of the 64x64 ladder and the final `out` planes are BYTE-IDENTICAL
/// to the unpooled interpreter's, and `tools/gate.py`'s CPU-vs-torch and
/// CPU-vs-CUDA agreements are unchanged to the digit.
pub struct Cpu<'a> {
    pub w: &'a Weights,
    pub blobs: HashMap<String, Blob>,
    /// Freed buffers, best-fit by element count. The CPU counterpart of
    /// `gpu::Pool`, and it exists for the same reason (see the struct doc).
    pool: Vec<Blob>,
    /// The name `run_until` stops at, or `None` for a whole pass. It is what
    /// `run_to` compares each op's output against; the release filter does not
    /// need it (the loop returns before anything is released), and `run` does not
    /// either (`out` is read by nothing, so it is never released).
    keep: Option<String>,
    /// Output geometry of every name in the graph, from `shapes_of` - the same
    /// walk the GPU sizes its device buffers from - so `out_buf` knows how big a
    /// recycled buffer has to be.
    shapes: HashMap<String, (usize, usize, usize)>,
    /// PER-OP WALL TIME, when `IFAN_CPU_TIME` is set in the environment.
    ///
    /// The CPU twin of `gpu::Gpu`'s `IFAN_TIME`, and it exists for the same
    /// reason: `--dump-op` answers "how far into the pass is this activation"
    /// with a figure that also carries the run's fixed cost (weight load, the
    /// shape walk, the plan) and, for the named tensor, a file write - so
    /// differencing two of them attributes the difference to nothing in
    /// particular. Timing each op here measures the thing being asked about.
    ///
    /// WALL time, not CPU time, and that is deliberate: the parallelism is INSIDE
    /// an op (rayon over output channels), so an op's wall time is what the pass
    /// actually spends on it, and the per-op figures add up to the pass's wall
    /// time. Each entry is (output name, kind, milliseconds, GFLOP), the last
    /// being 2 * MACs for the convolutions and 0 for everything else - so the
    /// table can be read as a rate as well as a duration, which is the question
    /// worth asking about a kernel that is 70% of the arithmetic.
    times: Vec<(String, &'static str, f64, f64)>,
}

impl<'a> Cpu<'a> {
    pub fn new(w: &'a Weights) -> Cpu<'a> {
        Cpu {
            w,
            blobs: HashMap::new(),
            pool: Vec::new(),
            keep: None,
            shapes: HashMap::new(),
            times: Vec::new(),
        }
    }

    /// The geometry of the pass, from the graph. Called once by `run`/`run_until`.
    fn shape(&self, name: &str) -> Result<(usize, usize, usize), String> {
        self.shapes
            .get(name)
            .copied()
            .ok_or_else(|| format!("no shape recorded for `{name}`"))
    }

    fn get(&self, name: &str) -> &Blob {
        self.blobs
            .get(name)
            .unwrap_or_else(|| panic!("graph reads `{name}` before it is written"))
    }

    /// Remove a name's buffer, so the caller can write into it instead of
    /// allocating a new one. `HashMap::get_mut` cannot be used for this: the op
    /// also reads OTHER names in the same map, and the borrow checker is right to
    /// refuse it.
    ///
    /// The error keeps the diagnostic `get` has - a read before a write is a graph
    /// bug, and it should say which name rather than allocate a zeroed buffer and
    /// produce a plausible wrong picture.
    fn take(&mut self, name: &str) -> Blob {
        self.blobs
            .remove(name)
            .unwrap_or_else(|| panic!("graph reads `{name}` before it is written"))
    }

    /// Store a name's buffer, CHECKING that it really holds that name's geometry.
    ///
    /// The invariant this enforces is what makes a recycled buffer safe to hand to
    /// a kernel: the kernels index their SOURCE by the source blob's own `c`,`h`,`w`
    /// and a blob whose announced shape and `d.len()` disagree is a slice index out
    /// of range or, worse, a silently wrong plane. Sizing is the pool's job; this
    /// only refuses a buffer that does not match the name it is stored under.
    fn put(&mut self, name: &str, b: Blob) {
        let (c, h, w) = self
            .shape(name)
            .unwrap_or_else(|e| panic!("put({name}): {e}"));
        assert_eq!(
            (b.c, b.h, b.w),
            (c, h, w),
            "put({name}): buffer is {}x{}x{} but the graph says {c}x{h}x{w}",
            b.c,
            b.h,
            b.w
        );
        // A blob's `c * h * w` and `d.len()` must be EQUAL at every moment it is
        // stored, because a kernel slices its source by that geometry: a blob whose
        // d is longer than its announced shape reads planes from an earlier name
        // (silently wrong arithmetic) and one that is shorter panics out of range.
        // `out_buf` is what guarantees the equality - see its `resize`.
        debug_assert_eq!(b.d.len(), c * h * w, "put({name}): d.len() != c*h*w");
        self.blobs.insert(name.to_string(), b);
    }

    /// An output buffer for `name`, from the pool when one is big enough.
    ///
    /// A RECYCLED BUFFER IS ZEROED, and that is not defensive programming: two of
    /// the kernels ACCUMULATE into their destination (`conv1x1_into` sums over
    /// input channels with `*o += k * v`, `conv3x3s2_into` sums over them with
    /// `plane[..] += v`). A recycle that left the previous occupant's values there
    /// would add them to the result - at the first stride-2 convolution of the
    /// k-tower the sum comes out ~1.5e-1 away from the reference on a 12-bit scale.
    /// The GPU side memsets on recycle for the same reason; `fill(0.0)` over a few
    /// MiB is the cheaper equivalent here.
    fn out_buf(&mut self, name: &str) -> Blob {
        let (c, h, w) = self
            .shape(name)
            .unwrap_or_else(|e| panic!("{e}"));
        let need = c * h * w;
        let mut best: Option<usize> = None;
        for (i, b) in self.pool.iter().enumerate() {
            if b.d.len() >= need {
                match best {
                    Some(j) if self.pool[j].d.len() <= b.d.len() => {}
                    _ => best = Some(i),
                }
            }
        }
        match best {
            // The recycled buffer only has to be large enough, and then it is
            // RESHAPED to the name it will hold: `resize` sets the length to
            // exactly `c * h * w` and keeps the allocation, so the pool really does
            // recycle (truncating instead would leave the Vec too short for the
            // next, larger request and the pool would never hit). A blob's
            // `d.len()` and `c * h * w` must be EQUAL whenever a kernel can see it,
            // because a kernel slices its SOURCE by that geometry - see `put`.
            Some(i) => {
                let mut b = self.pool.swap_remove(i);
                if std::env::var("IFAN_TRACE_ALLOC").is_ok() {
                    eprintln!("recycle {name}: {} elems re-used for {need}", b.d.len());
                }
                b.c = c;
                b.h = h;
                b.w = w;
                // `resize` only zeroes what it GROWS; a recycled buffer that is
                // already the right length keeps the old values, so zero it
                // explicitly (see the doc comment above for why that is required
                // rather than tidy).
                b.d.resize(need, 0.0);
                b.d.fill(0.0);
                b
            }
            // `Blob::zeros` rather than uninitialised memory: a `Vec<f32>` of
            // arbitrary bytes can be a signalling NaN, and while every kernel
            // overwrites it, `--dump` of an op whose writer was skipped would then
            // print NaN rather than zeros.
            None => Blob::zeros(c, h, w),
        }
    }

    /// Run the whole graph, returning the value of `out`. `inp` is the cropped
    /// input; the graph names it `in`.
    pub fn run(&mut self, ops: &[Op], inp: &Blob) -> Blob {
        self.blobs.clear();
        self.keep = None;
        self.run_to(ops, inp);
        self.report_times(ops);
        self.blobs.remove("out").expect("graph has no Clip op")
    }

    /// The `IFAN_CPU_TIME` table, printed after a REAL pass (never from
    /// `run_to`, so a `--dump-op` run reports only the ops it actually ran).
    ///
    /// Aggregated by KIND and then by NAME, because the two answer different
    /// questions: by kind, where does the pass's time go (convolutions against
    /// the SAC iteration against the elementwise glue); by name, which layer.
    /// Only the top few names are printed - the interesting ones are the ops
    /// that are slow for their size, which is a minority of 191.
    fn report_times(&self, ops: &[Op]) {
        if self.times.is_empty() {
            return;
        }
        let total: f64 = self.times.iter().map(|t| t.2).sum();
        let gflop: f64 = self.times.iter().map(|t| t.3).sum();
        let mut by_kind: HashMap<&str, (usize, f64, f64)> = HashMap::new();
        for (_, kind, ms, gf) in &self.times {
            let e = by_kind.entry(kind).or_insert((0, 0.0, 0.0));
            e.0 += 1;
            e.1 += ms;
            e.2 += gf;
        }
        let mut kv: Vec<_> = by_kind.into_iter().collect();
        kv.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "ifan: CPU per-op wall time (IFAN_CPU_TIME), {} ops, {:.2} s total, {gflop:.1} GFLOP reported",
            self.times.len(),
            total / 1e3
        );
        eprintln!("ifan: {:<16} {:>6} {:>10} {:>8} {:>10}", "kind", "ops", "s", "%", "GFLOP/s");
        for (kind, (n, ms, gf)) in &kv {
            let rate = if *ms > 0.0 { gf / (ms / 1e3) } else { 0.0 };
            eprintln!(
                "ifan: {kind:<16} {n:>6} {:>10.2} {:>7.1}% {rate:>10.1}",
                ms / 1e3,
                100.0 * ms / total
            );
        }
        // By SHAPE, for the convolutions only: the same kernel at a different
        // size, or a different channel count, is the cheapest way to separate
        // "this kernel is slow" from "this kernel is starved". A rate that holds
        // steady as the working set grows past the caches is an instruction-bound
        // kernel; one that collapses is a memory-bound one.
        let mut by_shape: HashMap<String, (usize, f64, f64)> = HashMap::new();
        for (name, kind, ms, gf) in &self.times {
            if *gf == 0.0 {
                continue;
            }
            if let Some(op) = ops.iter().find(|o| o.out_name() == name) {
                let (ci, co, h, wd) = op_geom(op, &self.shapes);
                let key = format!("{kind} {ci}->{co} @ {h}x{wd}");
                let e = by_shape.entry(key).or_insert((0, 0.0, 0.0));
                e.0 += 1;
                e.1 += ms;
                e.2 += gf;
            }
        }
        let mut sv: Vec<_> = by_shape.into_iter().collect();
        sv.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("ifan: {:<28} {:>4} {:>9} {:>10}", "shape", "ops", "s", "GFLOP/s");
        for (key, (n, ms, gf)) in sv.iter().take(14) {
            eprintln!("ifan: {key:<28} {n:>4} {:>9.3} {:>10.1}", ms / 1e3, gf / (ms / 1e3));
        }
        let mut nv: Vec<_> = self.times.iter().collect();
        nv.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("ifan: {:<16} {:>6} {:>10} {:>8} {:>10}", "slowest ops", "", "s", "%", "GFLOP/s");
        for (name, kind, ms, gf) in nv.iter().take(12) {
            let rate = if *ms > 0.0 { gf / (ms / 1e3) } else { 0.0 };
            eprintln!(
                "ifan: {name:<10} {kind:<14} {:>8.2} {:>7.1}% {rate:>10.1}",
                ms / 1e3,
                100.0 * ms / total
            );
        }
    }

    /// Run only up to (and including) the op producing `stop`, and return that
    /// blob. Used by `--dump-op` to compare one stage at a time.
    pub fn run_until(&mut self, ops: &[Op], inp: &Blob, stop: &str) -> Blob {
        self.blobs.clear();
        self.keep = Some(stop.to_string());
        self.run_to(ops, inp);
        self.blobs
            .remove(stop)
            .unwrap_or_else(|| panic!("graph has no op producing `{stop}`"))
    }

    /// The interpreter's loop, shared by `run` and `run_until`.
    ///
    /// ONE LOOP, because the two differ only in where they stop, and a second copy
    /// of the release rule is exactly how `--dump-op` and a normal run would come
    /// to disagree about when a buffer dies - and the dump is the tool used to
    /// debug the normal run. (This is the same argument `Gpu::run_ops` makes.)
    fn run_to(&mut self, ops: &[Op], inp: &Blob) {
        let live = Liveness::of(ops);
        // Shapes first (from the graph, never from a table), so a recycled buffer
        // can be sized against the name it will hold and a shape that drifted is
        // an error rather than a short write.
        self.shapes = shapes_of(self.w, ops, inp.c, inp.h, inp.w)
            .unwrap_or_else(|e| panic!("{e}"));
        self.put("in", inp.clone());
        for (i, op) in ops.iter().enumerate() {
            // The op index makes a divergence report point at a line of the graph
            // rather than at "somewhere in the network".
            let name = op.out_name().to_string();
            // A reader must FIND its buffer: `out_buf` creates on demand, which is
            // right for a write target and would silently hand a reader a zeroed
            // buffer for a name nothing ever wrote. Same check as the GPU's.
            for r in op.reads() {
                if self.blobs.get(r).is_none() {
                    panic!("op {i} {op:?} reads `{r}` before it is written");
                }
            }
            let t0 = std::time::Instant::now();
            self.exec_into(op).unwrap_or_else(|e| panic!("op {i} {op:?}: {e}"));
            if std::env::var("IFAN_CPU_TIME").is_ok() {
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                self.times.push((name.clone(), op_kind(op), ms, op_gflop(op, &self.shapes)));
            }
            if self.keep.as_deref() == Some(name.as_str()) {
                // STOP AT THE OP THAT PRODUCES THE NAME, exactly as `Gpu::run_ops`
                // does. `--dump-op` asks for the value at a STAGE, and a name a
                // later op rewrites in place - a ResBlock's `m`, which the
                // LeakyRelu immediately activates - would otherwise be reported as
                // its post-activation contents. The GPU path has always stopped
                // here; this is the CPU twin matching it, and it is why the two
                // backends' dumps are comparable stage for stage.
                return;
            }
            // Release everything whose last reader was this op or earlier - the
            // same rule and the same `<=` as `gpu::Gpu::run_ops`, including the
            // reason for it: `dies_after` names the op that READS the name last, and
            // once that op has run the buffer is dead. A name nothing reads keeps
            // `usize::MAX` and survives (the graph's `out`).
            //
            // The op's OWN output is excluded, and that is the `!=`: a name that an
            // op both reads and writes (a ResBlock's `m`) has `dies_after` equal to
            // this op, so without the exclusion it would go back to the pool in the
            // same op that produced it.
            let done: Vec<String> = self
                .blobs
                .keys()
                .filter(|n| n.as_str() != name && live.dies_after(n) <= i)
                .cloned()
                .collect();
            for n in done {
                if let Some(b) = self.blobs.remove(&n) {
                    self.pool.push(b);
                }
            }
        }
    }

    /// Which channels of the filter tensor a named half refers to.
    ///
    /// The filter tensor is `[N*(ch4*Fs*2) + N*ch4, h4, w4]`: the first `N*2`
    /// blocks are the tap sets of each iteration in order (k1 then k2 for
    /// iteration 0, then iteration 1, ...) and the last `N` blocks are the bias
    /// blocks. THIS ORDER IS THE CHECKPOINT'S, not a choice - `torch.split` on
    /// the channel axis in `IAC.forward` defines it, and getting it wrong
    /// produces a plausible-looking blur instead of an error.
    fn tap_range(&self, name: &str) -> Result<(usize, usize), String> {
        tap_range(&self.w.arch, name)
    }

    /// A `[channels, h, w]` copy of a channel range of the filter tensor.
    ///
    /// A COPY, not a view: the SAC passes read their taps with the same indexing
    /// as the feature map, and a strided view would need a second indexing path
    /// in `sac_pass` that the CUDA kernel does not have. 91 MiB at 1080p, once
    /// per op - and the alternative is two code paths for the same arithmetic.
    fn tap_view(&self, base: usize, channels: usize) -> Blob {
        let f = self.get("filt");
        let step = f.h * f.w;
        let mut out = Blob::zeros(channels, f.h, f.w);
        out.d.copy_from_slice(&f.d[base * step..(base + channels) * step]);
        out
    }

    /// One op, writing its result into a buffer taken from the pool.
    ///
    /// The WRITE TARGET is taken from `blobs` (so a recycled buffer is reused)
    /// and the READS stay borrowed from the same map; that is why `take` exists
    /// rather than `get_mut`. Every op below follows the same shape:
    ///
    ///   * read the source buffers that are still in `blobs`,
    ///   * `out_buf(name)` for the destination,
    ///   * write into it with the kernel's `_into` form.
    ///
    /// WHICH OPS CAN DO THIS WITHOUT CHANGING A NUMBER. A destination may be
    /// recycled only when no source of the same op still holds it, which the
    /// release rule guarantees: a name enters the pool at the end of the op that
    /// reads it last. The graph's one aliasing case is `LeakyRelu` with
    /// `out == x` (`m = leaky_relu(m)` in every ResBlock), handled in place in its
    /// arm below - elementwise, so the values are the ones the clone-then-write
    /// this replaced produced.
    ///
    /// THE ACCUMULATING KERNELS are why a recycled buffer is zeroed instead of
    /// being written over blindly: `conv1x1_into` and `conv3x3s2_into` sum into
    /// their destination (`*o += k * v`, `plane[..] += v`). See `out_buf`.
    fn exec_into(&mut self, op: &Op) -> Result<(), String> {
        let out_name = op.out_name().to_string();
        match op {
            Op::Conv3x3 { x, w, act, .. } => {
                let mut out = self.out_buf(&out_name);
                conv3x3_into(self.get(x), &mut out, self.w.weights(w), self.w.bias(w), *act);
                self.put(&out_name, out);
            }
            Op::Conv1x1 { x, w, .. } => {
                let mut out = self.out_buf(&out_name);
                conv1x1_into(self.get(x), &mut out, self.w.weights(w), self.w.bias(w));
                self.put(&out_name, out);
            }
            Op::Conv3x3S2 { x, w, act, .. } => {
                let mut out = self.out_buf(&out_name);
                conv3x3s2_into(self.get(x), &mut out, self.w.weights(w), self.w.bias(w), *act);
                self.put(&out_name, out);
            }
            Op::ConvT4x4S2 { x, w, .. } => {
                let mut out = self.out_buf(&out_name);
                convtranspose4x4s2_into(
                    self.get(x),
                    &mut out,
                    self.w.weights(w),
                    self.w.bias(w),
                    true,
                );
                self.put(&out_name, out);
            }
            Op::LeakyRelu { x, slope, .. } => {
                // `out == x` IS ALIVE IN THIS GRAPH (`m = leaky_relu(m)` inside
                // every ResBlock), and the two cases are both one buffer:
                //   * `out == x`: take the buffer and activate it in place, which
                //     is elementwise and so leaves the same values;
                //   * otherwise: a recycled destination, filled from the source.
                // Neither needs the buffer zeroed, because leaky_relu ASSIGNS every
                // element (unlike the conv kernels - see `out_buf`).
                if x == &out_name {
                    let mut b = self.take(x);
                    lrelu_in_place_par(&mut b, *slope);
                    self.put(&out_name, b);
                } else {
                    // `out_buf` returns an OWNED buffer, so the source can be
                    // borrowed from `self` right after it is taken - no clone.
                    let mut b = self.out_buf(&out_name);
                    lrelu_into_par(self.get(x), &mut b, *slope);
                    self.put(&out_name, b);
                }
            }
            Op::Sac1D { x, tap, ksize, vertical, .. } => {
                let (base, channels) = self.tap_range(tap)?;
                let tapb = self.tap_view(base, channels);
                let mut out = self.out_buf(&out_name);
                sac_pass_into_par(self.get(x), &tapb, &mut out, *ksize, *vertical);
                self.put(&out_name, out);
            }
            Op::SacBias { x, bias, .. } => {
                // NO IN-PLACE CASE HERE: the graph writes `iac.o.i` from `iac.h.i`,
                // two different names, and the bias plane comes from `filt`. The
                // kernel would tolerate `out == x` (it reads `x[ch*plane+j]` and
                // writes `out[...]` at the same index, so a single buffer would be
                // correct), but nothing asks for it and the destination is recycled
                // from the pool instead.
                let (base, channels) = self.tap_range(bias)?;
                let step = {
                    let f = self.get("filt");
                    f.h * f.w
                };
                // HOW BIG the feature map is, read now: the destination comes out
                // of `self` and needs `&mut`, so no borrow of `self` may outlive it.
                // Copying three `usize`s (not the plane) keeps the borrow graph
                // flat and the loop below allocation-free.
                let (xc, xh, xw) = {
                    let xb = self.get(x);
                    (xb.c, xb.h, xb.w)
                };
                if xc != channels {
                    return Err(format!(
                        "bias block has {channels} channels but the feature map has {xc}"
                    ));
                }
                // The bias block is a whole PLANE per channel, not a scalar:
                // upstream's `f = f + f_bs[i]` broadcasts `f_bs[i]` ([N, c, H, W])
                // over the image, and the filter tensor's spatial size is already
                // the 1/8 grid the feature map lives on, so the shapes match
                // exactly. Reading one element per plane would silently turn a
                // spatially varying bias into a constant, which shows up as a
                // per-pixel error with the bias plane's own range.
                let (fh, fw) = {
                    let f = self.get("filt");
                    (f.h, f.w)
                };
                if (fh, fw) != (xh, xw) {
                    return Err(format!(
                        "bias block is {fh}x{fw} but the feature map is {xh}x{xw}"
                    ));
                }
                // The sources are borrowed AFTER the destination is taken, for the
                // borrow checker's benefit and no other reason: `out_buf` hands back
                // an OWNED buffer, so these borrows of `self` do not conflict - and
                // they must not CLONE, because `filt` is 493 MB at 720x720 (15232
                // planes of 90x90) and cloning it once per SAC iteration cost 2.1 s
                // of the 3.6 s pass when this was first written.
                let mut out = self.out_buf(&out_name);
                sac_bias_into_par(&self.get("filt").d, self.get(x), &mut out, base, step);
                self.put(&out_name, out);
            }
            Op::Add { a, b, .. } => {
                let mut out = self.out_buf(&out_name);
                add_into_par(self.get(a), self.get(b), &mut out);
                self.put(&out_name, out);
            }
            Op::Cat2 { a, b, .. } => {
                {
                    let (x, y) = (self.get(a), self.get(b));
                    if x.c != y.c || x.h != y.h || x.w != y.w {
                        return Err(format!(
                            "cat: {} is {}x{}x{} but {} is {}x{}x{}",
                            a, x.c, x.h, x.w, b, y.c, y.h, y.w
                        ));
                    }
                }

                // Inputs first, then the second input - torch.cat's order, which
                // the weights depend on (the conv sees f then f_dm).
                let mut out = self.out_buf(&out_name);
                // `copy_from_slice` into the region the graph says this name is,
                // NOT into the whole recycled buffer - a best-fit recycle may be
                // larger than the name's geometry.
                //
                // The sources are CLONED rather than borrowed because the
                // destination came out of `self` (see `take`); two inputs of a cat
                // are a fraction of the pass's traffic, and cloning keeps the
                // interpreter's borrow graph simple enough to read.
                let x = self.get(a).clone();
                let y = self.get(b).clone();
                let (xl, yl) = (x.d.len(), y.d.len());
                out.d[..xl].copy_from_slice(&x.d);
                out.d[xl..xl + yl].copy_from_slice(&y.d);
                self.put(&out_name, out);
            }
            Op::Copy { x, .. } => {
                let mut out = self.out_buf(&out_name);
                let src = self.get(x).clone();
                out.d[..src.d.len()].copy_from_slice(&src.d);
                self.put(&out_name, out);
            }
            Op::Clip { x, .. } => {
                // The final op, and the only one whose destination is read by
                // nothing: `out` is never released (no op reads it) and the caller
                // takes it out of the map.
                let mut out = self.out_buf(&out_name);
                clip_into_par(self.get(x), &mut out);
                self.put(&out_name, out);
            }
        }
        Ok(())
    }
}

/// The channel block of the filter tensor that `name` refers to: `(base, count)`.
///
/// The order is the CHECKPOINT'S, not a choice - `torch.split` on the channel axis
/// in `IAC.forward` defines it, and getting it wrong produces a plausible-looking
/// blur instead of an error. It lives here as a free function, next to the graph
/// that names these tensors, because BOTH interpreters need it: the CPU to slice
/// its host copy and the GPU to compute a device pointer. Two copies of this rule
/// is exactly how the two backends would come to disagree.
pub fn tap_range(a: &Arch, name: &str) -> Result<(usize, usize), String> {
    let (which, i) = split_tap(name)?;
    let ck = a.ch4 * a.fs;
    Ok(match which {
        "k1" => (i * ck * 2, ck),
        "k2" => (i * ck * 2 + ck, ck),
        "bs" => (2 * a.n * ck + i * a.ch4, a.ch4),
        other => return Err(format!("unknown filter half `{other}` in `{name}`")),
    })
}

/// Split a filter-tensor name (`filt.k1.7`, `filt.bs.0`) into its half and index.
///
/// A name rather than an index because the graph is data: `--dump` and a
/// divergence report both print these strings, and `filt.k2.3` says which tensor
/// is wrong in a way that `(1, 3)` does not.
fn split_tap(name: &str) -> Result<(&str, usize), String> {
    let mut it = name.split('.');
    if it.next() != Some("filt") {
        return Err(format!("bad filter name `{name}`: expected a `filt.` prefix"));
    }
    let which = it.next().ok_or_else(|| format!("bad filter name `{name}`"))?;
    let idx = it
        .next()
        .ok_or_else(|| format!("bad filter name `{name}`"))?
        .parse::<usize>()
        .map_err(|e| format!("bad filter index in `{name}`: {e}"))?;
    Ok((which, idx))
}
