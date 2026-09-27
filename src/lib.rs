//! IFAN in Rust: a single-image, 8-bit restoration engine.
//!
//! The engine is a library so the binary and the golden comparison can drive the
//! same code. `main.rs` is a thin CLI over this, and `tools/gate.py` is the
//! comparison: it runs the `ifan` binary twice, once per backend, and once
//! `tools/reference.py` on the original PyTorch checkpoint, and compares the
//! three dumps with `--dump` - RAW f32 PLANES rather than PNG pixels, because
//! 8-bit quantisation hides exactly the differences a port gets wrong. The
//! comparison lives outside `tests/` because it needs torch to produce its third
//! opinion, which `cargo test` cannot assume.
pub mod image;
pub mod kernels;
pub mod memguard;
pub mod net;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod gpu;
