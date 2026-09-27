//! Weights: an mmap of `models/IFAN.safetensors` plus the architecture constants
//! the file declares, with every tensor shape-checked against the inventory at
//! load time.
//!
//! WHY THE METADATA IS TRUSTED AND CHECKED RATHER THAN INFERRED. `config_IFAN.py`
//! defines ch/n/fs/res_num/ks; the checkpoint does not. tools/convert.py reads
//! them out of the weights where they are recoverable (ch from conv1_1, n from
//! F.3's 15232 outputs, ch4 from conv4_4, fs from the config) and writes them
//! into `__metadata__`, so a converted file cannot misstate its own iteration
//! count. This module then requires the metadata to be present, agrees with the
//! shapes, and agrees with the engine's expectations - a file whose F.3 is
//! 15232 but whose N says 8 is rejected by name rather than run with a
//! misinterpreted filter layout.
//!
//! WHY THE SHAPE TABLE IS EXHAUSTIVE. A missing tensor surfaces as a driver
//! error or, worse, as a launch against a zero-filled buffer; a MIS-SHAPED one
//! surfaces as garbage pixels. Both are cheap to rule out once here, and the
//! table below is generated from the real file (tools/tensor_shapes.json) rather
//! than transcribed, so it cannot quietly omit the twentieth ResnetBlock stem.
//! `SHAPES.len() == NUMBER_OF_TENSORS` is asserted so a deleted entry is a
//! compile-time-visible mistake rather than a silent hole.
use std::path::{Path, PathBuf};

use lightgpu::safetensors::File;

/// How many tensors the released IFAN checkpoint has in its `module.Network.`
/// half. Pinned because the shape table is exhaustive: a table with an entry
/// missing still checks every tensor that IS listed, and the count is what makes
/// "listed" mean "all of them". The 20 `module.reblurNet.*` tensors are
/// training-only and are dropped by tools/convert.py, so they are not here.
pub const NUMBER_OF_TENSORS: usize = 158;

/// Architecture constants, as the converted file declares them.
#[derive(Debug, Clone, Copy)]
pub struct Arch {
    /// Base channel count (32).
    pub ch: usize,
    /// IAC iterations (17 in the shipped checkpoint).
    pub n: usize,
    /// SAC kernel size (3).
    pub fs: usize,
    /// ResnetBlock repeats for the blocks that use the default.
    pub res_num: usize,
    /// `ch * 4`, the channel count at 1/8 resolution (128).
    pub ch4: usize,
    /// Refine value: sides are cropped to a multiple of this (8).
    pub refine_val: usize,
}

impl Arch {
    /// `kernel_dim`, F.3's output channels: N*(ch4*Fs*2) + N*ch4.
    ///
    /// This is the number that makes the filter layout unambiguous, and it is
    /// DERIVED here rather than read, so a metadata/file disagreement about N
    /// shows up as a shape check failing at F.3 instead of as a plausible but
    /// wrong iteration count.
    pub fn kernel_dim(&self) -> usize {
        self.n * (self.ch4 * self.fs * 2) + self.n * self.ch4
    }
}

pub struct Weights {
    file: File,
    pub arch: Arch,
    /// Path, kept for error messages and the GPU plan's provenance line.
    path: PathBuf,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Weights, String> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let arch = Arch {
            ch: file.metadata_usize("ch")?,
            n: file.metadata_usize("n")?,
            fs: file.metadata_usize("fs")?,
            res_num: file.metadata_usize("res_num")?,
            ch4: file.metadata_usize("ch4")?,
            refine_val: 8,
        };
        if let Some(a) = file.metadata_get("arch") {
            if a != "ifan" {
                return Err(format!("{}: metadata arch is `{a}`, expected `ifan`", path.display()));
            }
        } else {
            return Err(format!(
                "{}: no `arch` metadata - this is not a converted IFAN checkpoint \
                 (run tools/convert.py)",
                path.display()
            ));
        }
        let w = Weights { file, arch, path };
        w.check_arch()?;
        w.check_shapes()?;
        Ok(w)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The metadata's own consistency, and the two constants the engine's graph
    /// hard-codes (`ch4 == ch * 4`, and REFINE_VAL == 8 from config_IFAN.py).
    fn check_arch(&self) -> Result<(), String> {
        let a = self.arch;
        if a.ch4 != a.ch * 4 {
            return Err(format!("metadata ch4={} is not 4*ch={}", a.ch4, a.ch));
        }
        if a.ch == 0 || a.n == 0 || a.fs == 0 || a.fs % 2 == 0 {
            return Err(format!(
                "metadata ch={} n={} fs={} is not usable (fs must be odd for a centred 1-D tap)",
                a.ch, a.n, a.fs
            ));
        }
        if a.res_num == 0 {
            return Err("metadata res_num=0".into());
        }
        // kernel_dim is the one derived number that ties F.3's shape to N. If the
        // file's F.3 does not have exactly this many output channels then one of
        // the two is wrong, and every per-iteration filter offset would be too.
        let f3 = self.shape("F.3.weight")?;
        if f3[0] != a.kernel_dim() {
            return Err(format!(
                "F.3 has {} output channels but metadata (n={}, ch4={}, fs={}) implies \
                 kernel_dim = {} - the file and its metadata disagree about the iteration count",
                f3[0], a.n, a.ch4, a.fs, a.kernel_dim()
            ));
        }
        // Fs is not in the checkpoint (F.3 is 1x1), so it is the one constant the
        // metadata could get wrong without contradicting any shape. It is
        // cross-checked against the only other place it appears: SAC splits the
        // filter tensor into N*(ch4*Fs*2) + N*ch4, so a wrong Fs still yields a
        // self-consistent kernel_dim. There is nothing else to check it against
        // short of running the model, which is what the golden suite is for; the
        // converter reads it from config_IFAN.py and says so.
        Ok(())
    }

    /// Every tensor present, every shape equal to the inventory.
    fn check_shapes(&self) -> Result<(), String> {
        let mut missing = Vec::new();
        for (name, want) in SHAPES {
            match self.file.shape(name) {
                Ok(got) => {
                    if got != *want {
                        return Err(format!(
                            "{}: tensor `{name}` has shape {:?}, expected {:?}",
                            self.path.display(),
                            got,
                            want
                        ));
                    }
                }
                Err(_) => missing.push(*name),
            }
        }
        if !missing.is_empty() {
            let shown: Vec<&str> = missing.iter().take(4).copied().collect();
            return Err(format!(
                "{}: {} of {NUMBER_OF_TENSORS} tensors are missing ({}); this is not the \
                 released IFAN checkpoint, or it is a different iteration count",
                self.path.display(),
                missing.len(),
                shown.join(", ")
            ));
        }
        // An extra tensor is not fatal - a newer converter could legitimately add
        // one - but it is worth saying, because the only way it happens today is
        // reading a file that was never pruned of `module.reblurNet.*`.
        // `file.len()` is the mapped FILE length in BYTES; the tensor count is
        // `order()`, not `len()`. Confusing the two prints millions of tensors,
        // and a wrong diagnostic is worse than none.
        let extra = self.file.order().len().saturating_sub(SHAPES.len());
        if extra > 0 {
            eprintln!(
                "ifan: note: {} carries {extra} tensor(s) beyond the {NUMBER_OF_TENSORS} this \
                 engine reads; they are ignored",
                self.path.display()
            );
        }
        Ok(())
    }

    /// Borrowed f32 view. Zero-copy: the mmap is held for the process's life, so
    /// the slice is valid as long as `self`.
    #[inline]
    pub fn f32(&self, name: &str) -> &[f32] {
        // The shape table was verified at load, so a lookup failure here is a
        // programming error rather than a bad file.
        self.file.f32(name).unwrap_or_else(|e| panic!("tensor `{name}`: {e}"))
    }

    #[inline]
    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        self.file.shape(name)
    }

    #[inline]
    pub fn weights(&self, name: &str) -> &[f32] {
        self.f32(&format!("{name}.weight"))
    }

    #[inline]
    pub fn bias(&self, name: &str) -> &[f32] {
        self.f32(&format!("{name}.bias"))
    }
}

/// The inventory, generated from the converted checkpoint
/// (tools/tensor_shapes.json) - not hand-written.
pub static SHAPES: &[(&str, &[usize])] = &[
    ("DME.0.0.bias", &[128]),
    ("DME.0.0.weight", &[128, 128, 3, 3]),
    ("DME.1.stem.0.0.bias", &[128]),
    ("DME.1.stem.0.0.weight", &[128, 128, 3, 3]),
    ("DME.1.stem.0.2.bias", &[128]),
    ("DME.1.stem.0.2.weight", &[128, 128, 3, 3]),
    ("DME.1.stem.1.0.bias", &[128]),
    ("DME.1.stem.1.0.weight", &[128, 128, 3, 3]),
    ("DME.1.stem.1.2.bias", &[128]),
    ("DME.1.stem.1.2.weight", &[128, 128, 3, 3]),
    ("DME.2.stem.0.0.bias", &[128]),
    ("DME.2.stem.0.0.weight", &[128, 128, 3, 3]),
    ("DME.2.stem.0.2.bias", &[128]),
    ("DME.2.stem.0.2.weight", &[128, 128, 3, 3]),
    ("DME.2.stem.1.0.bias", &[128]),
    ("DME.2.stem.1.0.weight", &[128, 128, 3, 3]),
    ("DME.2.stem.1.2.bias", &[128]),
    ("DME.2.stem.1.2.weight", &[128, 128, 3, 3]),
    ("DME.3.bias", &[1]),
    ("DME.3.weight", &[1, 128, 3, 3]),
    ("F.0.0.bias", &[128]),
    ("F.0.0.weight", &[128, 128, 3, 3]),
    ("F.1.stem.0.0.bias", &[128]),
    ("F.1.stem.0.0.weight", &[128, 128, 3, 3]),
    ("F.1.stem.0.2.bias", &[128]),
    ("F.1.stem.0.2.weight", &[128, 128, 3, 3]),
    ("F.1.stem.1.0.bias", &[128]),
    ("F.1.stem.1.0.weight", &[128, 128, 3, 3]),
    ("F.1.stem.1.2.bias", &[128]),
    ("F.1.stem.1.2.weight", &[128, 128, 3, 3]),
    ("F.2.stem.0.0.bias", &[128]),
    ("F.2.stem.0.0.weight", &[128, 128, 3, 3]),
    ("F.2.stem.0.2.bias", &[128]),
    ("F.2.stem.0.2.weight", &[128, 128, 3, 3]),
    ("F.2.stem.1.0.bias", &[128]),
    ("F.2.stem.1.0.weight", &[128, 128, 3, 3]),
    ("F.2.stem.1.2.bias", &[128]),
    ("F.2.stem.1.2.weight", &[128, 128, 3, 3]),
    ("F.3.bias", &[15232]),
    ("F.3.weight", &[15232, 128, 1, 1]),
    ("conv1_1.0.bias", &[32]),
    ("conv1_1.0.weight", &[32, 3, 3, 3]),
    ("conv1_2.0.bias", &[32]),
    ("conv1_2.0.weight", &[32, 32, 3, 3]),
    ("conv1_3.0.bias", &[32]),
    ("conv1_3.0.weight", &[32, 32, 3, 3]),
    ("conv2_1.0.bias", &[64]),
    ("conv2_1.0.weight", &[64, 32, 3, 3]),
    ("conv2_2.0.bias", &[64]),
    ("conv2_2.0.weight", &[64, 64, 3, 3]),
    ("conv2_3.0.bias", &[64]),
    ("conv2_3.0.weight", &[64, 64, 3, 3]),
    ("conv3_1.0.bias", &[128]),
    ("conv3_1.0.weight", &[128, 64, 3, 3]),
    ("conv3_2.0.bias", &[128]),
    ("conv3_2.0.weight", &[128, 128, 3, 3]),
    ("conv3_3.0.bias", &[128]),
    ("conv3_3.0.weight", &[128, 128, 3, 3]),
    ("conv4_1.0.bias", &[128]),
    ("conv4_1.0.weight", &[128, 128, 3, 3]),
    ("conv4_2.0.bias", &[128]),
    ("conv4_2.0.weight", &[128, 128, 3, 3]),
    ("conv4_3.0.bias", &[128]),
    ("conv4_3.0.weight", &[128, 128, 3, 3]),
    ("conv4_4.0.0.bias", &[128]),
    ("conv4_4.0.0.weight", &[128, 256, 3, 3]),
    ("conv4_4.1.stem.0.0.bias", &[128]),
    ("conv4_4.1.stem.0.0.weight", &[128, 128, 3, 3]),
    ("conv4_4.1.stem.0.2.bias", &[128]),
    ("conv4_4.1.stem.0.2.weight", &[128, 128, 3, 3]),
    ("conv4_4.1.stem.1.0.bias", &[128]),
    ("conv4_4.1.stem.1.0.weight", &[128, 128, 3, 3]),
    ("conv4_4.1.stem.1.2.bias", &[128]),
    ("conv4_4.1.stem.1.2.weight", &[128, 128, 3, 3]),
    ("conv4_4.2.stem.0.0.bias", &[128]),
    ("conv4_4.2.stem.0.0.weight", &[128, 128, 3, 3]),
    ("conv4_4.2.stem.0.2.bias", &[128]),
    ("conv4_4.2.stem.0.2.weight", &[128, 128, 3, 3]),
    ("conv4_4.2.stem.1.0.bias", &[128]),
    ("conv4_4.2.stem.1.0.weight", &[128, 128, 3, 3]),
    ("conv4_4.2.stem.1.2.bias", &[128]),
    ("conv4_4.2.stem.1.2.weight", &[128, 128, 3, 3]),
    ("conv4_4.3.0.bias", &[128]),
    ("conv4_4.3.0.weight", &[128, 128, 3, 3]),
    ("conv_DME.0.bias", &[128]),
    ("conv_DME.0.weight", &[128, 1, 3, 3]),
    ("conv_res.0.0.bias", &[128]),
    ("conv_res.0.0.weight", &[128, 128, 3, 3]),
    ("conv_res.1.stem.0.0.bias", &[128]),
    ("conv_res.1.stem.0.0.weight", &[128, 128, 3, 3]),
    ("conv_res.1.stem.0.2.bias", &[128]),
    ("conv_res.1.stem.0.2.weight", &[128, 128, 3, 3]),
    ("conv_res.1.stem.1.0.bias", &[128]),
    ("conv_res.1.stem.1.0.weight", &[128, 128, 3, 3]),
    ("conv_res.1.stem.1.2.bias", &[128]),
    ("conv_res.1.stem.1.2.weight", &[128, 128, 3, 3]),
    ("conv_res.1.stem.2.0.bias", &[128]),
    ("conv_res.1.stem.2.0.weight", &[128, 128, 3, 3]),
    ("conv_res.1.stem.2.2.bias", &[128]),
    ("conv_res.1.stem.2.2.weight", &[128, 128, 3, 3]),
    ("conv_res.2.0.bias", &[128]),
    ("conv_res.2.0.weight", &[128, 128, 3, 3]),
    ("kconv1_1.0.bias", &[32]),
    ("kconv1_1.0.weight", &[32, 3, 3, 3]),
    ("kconv1_2.0.bias", &[32]),
    ("kconv1_2.0.weight", &[32, 32, 3, 3]),
    ("kconv1_3.0.bias", &[32]),
    ("kconv1_3.0.weight", &[32, 32, 3, 3]),
    ("kconv2_1.0.bias", &[64]),
    ("kconv2_1.0.weight", &[64, 32, 3, 3]),
    ("kconv2_2.0.bias", &[64]),
    ("kconv2_2.0.weight", &[64, 64, 3, 3]),
    ("kconv2_3.0.bias", &[64]),
    ("kconv2_3.0.weight", &[64, 64, 3, 3]),
    ("kconv3_1.0.bias", &[128]),
    ("kconv3_1.0.weight", &[128, 64, 3, 3]),
    ("kconv3_2.0.bias", &[128]),
    ("kconv3_2.0.weight", &[128, 128, 3, 3]),
    ("kconv3_3.0.bias", &[128]),
    ("kconv3_3.0.weight", &[128, 128, 3, 3]),
    ("kconv4_1.0.bias", &[128]),
    ("kconv4_1.0.weight", &[128, 128, 3, 3]),
    ("kconv4_2.0.bias", &[128]),
    ("kconv4_2.0.weight", &[128, 128, 3, 3]),
    ("kconv4_3.0.bias", &[128]),
    ("kconv4_3.0.weight", &[128, 128, 3, 3]),
    ("out_res.0.bias", &[3]),
    ("out_res.0.weight", &[3, 32, 3, 3]),
    ("upconv1_1.stem.0.0.bias", &[32]),
    ("upconv1_1.stem.0.0.weight", &[32, 32, 3, 3]),
    ("upconv1_1.stem.0.2.bias", &[32]),
    ("upconv1_1.stem.0.2.weight", &[32, 32, 3, 3]),
    ("upconv1_2.stem.0.0.bias", &[32]),
    ("upconv1_2.stem.0.0.weight", &[32, 32, 3, 3]),
    ("upconv1_2.stem.0.2.bias", &[32]),
    ("upconv1_2.stem.0.2.weight", &[32, 32, 3, 3]),
    ("upconv1_u.0.bias", &[32]),
    ("upconv1_u.0.weight", &[64, 32, 4, 4]),
    ("upconv2_1.stem.0.0.bias", &[64]),
    ("upconv2_1.stem.0.0.weight", &[64, 64, 3, 3]),
    ("upconv2_1.stem.0.2.bias", &[64]),
    ("upconv2_1.stem.0.2.weight", &[64, 64, 3, 3]),
    ("upconv2_2.stem.0.0.bias", &[64]),
    ("upconv2_2.stem.0.0.weight", &[64, 64, 3, 3]),
    ("upconv2_2.stem.0.2.bias", &[64]),
    ("upconv2_2.stem.0.2.weight", &[64, 64, 3, 3]),
    ("upconv2_u.0.bias", &[64]),
    ("upconv2_u.0.weight", &[128, 64, 4, 4]),
    ("upconv3_1.stem.0.0.bias", &[128]),
    ("upconv3_1.stem.0.0.weight", &[128, 128, 3, 3]),
    ("upconv3_1.stem.0.2.bias", &[128]),
    ("upconv3_1.stem.0.2.weight", &[128, 128, 3, 3]),
    ("upconv3_2.stem.0.0.bias", &[128]),
    ("upconv3_2.stem.0.0.weight", &[128, 128, 3, 3]),
    ("upconv3_2.stem.0.2.bias", &[128]),
    ("upconv3_2.stem.0.2.weight", &[128, 128, 3, 3]),
    ("upconv3_u.0.bias", &[128]),
    ("upconv3_u.0.weight", &[128, 128, 4, 4]),
];

// The table must be exhaustive. A shortened list would still verify every tensor
// it names and would silently stop checking the rest, which is the exact failure
// this constant prevents.
const _: () = assert!(SHAPES.len() == NUMBER_OF_TENSORS);
