//! The CUDA side of the GPU backend: load the two fatbins and expose the
//! kernels this engine launches by symbolic name.
//!
//! TWO MODULES, NOT ONE. `build.rs` compiles the toolkit's `kernels.cu` and this
//! project's `ifan.cu` separately, each with its own `--entries` list, and they
//! are loaded as two `CUmodule`s. One combined module would be simpler but would
//! make a name collision between them a SILENT SHADOWING (whichever was
//! registered second wins, and `Module::func` would return the wrong kernel with
//! no error anywhere), which is exactly the failure mode a fatbin-per-family
//! split prevents.
//!
//! The fatbins are embedded with `include_bytes!`, so the release binary is
//! self-contained: no `cuda/` directory has to ship beside it, and a mismatch
//! between the source and the running kernels is impossible.
#![cfg(feature = "cuda")]

use lightgpu::ffi::CUfunction;
use lightgpu::vm::Module;

/// The toolkit family. Names here are asserted against `build.rs`'s
/// TOOLKIT_KERNELS list at compile time by construction - the list IS this
/// array - and against lightgpu's own kernel set by build.rs.
static TOOLKIT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ifan_toolkit.fatbin"));
/// This project's kernels, all defined in `cuda/ifan.cu`.
static PROJECT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ifan_project.fatbin"));

/// The whole GPU kernel surface, resolved once at startup rather than per launch.
/// `Module::func` is a hash lookup on the driver side, so leaving it in the hot
/// loop costs an extra driver call per tensor op - and an IFAN pass launches
/// several hundred of them: 26 encoder/filter-encoder convolutions, 2 DME, 6 F,
/// 6 conv_res, 6 out_res/decoder, 3 transposed, and 34 SAC passes for N=17.
pub struct Kernels {
    pub toolkit: Module,
    pub project: Module,
}

/// Which module a name lives in. `Fam::Toolkit` is the shared lightgpu set,
/// `Fam::Project` is `cuda/ifan.cu`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Fam {
    Toolkit,
    Project,
}

impl Kernels {
    pub fn load() -> Result<Kernels, String> {
        Ok(Kernels {
            toolkit: Module::load(TOOLKIT_FATBIN)?,
            project: Module::load(PROJECT_FATBIN)?,
        })
    }

    /// Look up a kernel in the family it is declared in. The callers pass the
    /// `Fam` explicitly rather than searching both, so a kernel that is
    /// accidentally in the wrong file is a load-time failure instead of a
    /// silently-resolved duplicate.
    pub fn func(&self, fam: Fam, name: &str) -> Result<CUfunction, String> {
        let m = match fam {
            Fam::Toolkit => &self.toolkit,
            Fam::Project => &self.project,
        };
        m.func(name).map_err(|e| format!("kernel `{name}`: {e}"))
    }

    /// Every kernel this engine intends to launch, resolved up front. A missing
    /// one is reported by name now rather than at the first frame that happens
    /// to need it, which matters because a pruned kernel otherwise surfaces as a
    /// bare driver error several hundred operations into a run.
    pub fn check_all(&self) -> Result<(), String> {
        for n in TOOLKIT_KERNELS {
            self.func(Fam::Toolkit, n)?;
        }
        for n in PROJECT_KERNELS {
            self.func(Fam::Project, n)?;
        }
        Ok(())
    }
}

/// Mirrors `build.rs`'s TOOLKIT_KERNELS. Kept as a separate literal rather than
/// shared through a `build.rs`-generated file because the two have different
/// jobs: build.rs's list decides what is EMBEDDED (and is validated against the
/// toolkit source both ways), this one decides what is PRE-FLIGHTED at runtime.
/// A name in one and not the other is a build failure or a startup error, never
/// a silent skip.
pub static TOOLKIT_KERNELS: &[&str] = &["lg_lrelu", "lg_add", "lg_copy"];

/// Mirrors `build.rs`'s PROJECT_KERNELS.
pub static PROJECT_KERNELS: &[&str] = &[
    "if_conv1x1_tile",
    "if_conv3x3s2_tile",
    "if_conv3x3_tile",
    "if_convtranspose4x4s2p1_tile",
    "if_sac_vertical",
    "if_sac_horizontal",
    "if_sac_bias",
    "if_clip",
];
