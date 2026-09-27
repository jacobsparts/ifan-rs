//! A process-wide lock for the device tests.

/// Tests within one binary run in PARALLEL THREADS, and `lightgpu`'s module load
/// assumes the calling thread owns the current CUDA context. Two threads loading a
/// module at once makes the second one fail with CUDA_ERROR_INVALID_CONTEXT - which
/// is why every device test takes this lock for its whole body instead of trusting
/// the scheduler not to run two of them at once.
#[cfg(feature = "cuda")]
pub fn device_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
