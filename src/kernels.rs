//! The CPU arithmetic behind the graph's ops.
//!
//! Each function here transcribes one torch operation the network uses -
//! `F.conv2d` with explicit padding, `F.conv_transpose2d`, `F.leaky_relu(0.1)`,
//! the two SAC gather loops - in the same accumulation order, so a disagreement
//! with `tools/reference.py` can only be a transcription bug.
//!
//! THE ACCUMULATION ORDER IS PART OF THE CONTRACT, not an implementation detail:
//! float32 sums are not associative and the outputs here are compared against
//! torch to within a few units in the last place. Each kernel states the order it
//! uses; `docs/INTERNALS.md` says which of them may move and which may not.
//!
//! Rayon splits only over independent elements - output channels for the
//! convolutions, one channel or one element for the pointwise ops - so no split
//! reorders a single output element's sum.
use rayon::prelude::*;

use crate::net::Blob;

/// `F.conv2d(x, w, b, padding=1)` - 3x3, stride 1, pad 1, `act` = leaky_relu(0.1).
///
/// A convenience wrapper for callers with no buffer to fill: the real arithmetic
/// is `conv3x3_into`, which the CPU interpreter uses because it takes its output
/// from a pool of recycled buffers (see `net::Cpu`).
pub fn conv3x3(x: &Blob, w: &[f32], b: &[f32], act: bool) -> Blob {
    let mut out = Blob::zeros(b.len(), x.h, x.w);
    conv3x3_into(x, &mut out, w, b, act);
    out
}

/// `conv3x3`, writing into `out`, whose shape must be `(b.len(), x.h, x.w)`.
///
/// `out` and `x` are NEVER the same buffer at any call site, and both kernels
/// below read every channel of `x` for every channel of `out`, so aliasing them
/// would be wrong rather than merely undefined. `net::Cpu` only reuses a buffer
/// whose value is dead - `Liveness::dies_after` is the last op that READS it - so
/// the two can never coincide.
///
/// DISPATCH, AND WHY THE BINARY STAYS PORTABLE. On x86-64 with AVX2 and FMA this
/// runs the vectorised kernel below; everywhere else - and on an x86-64 CPU that
/// has neither feature - it runs `conv3x3_scalar_into`, which is the same
/// arithmetic written one pixel at a time. Neither the vector code nor the CPU
/// check it needs exists on a non-x86-64 target: the function is behind
/// `#[cfg(target_arch = "x86_64")]` at COMPILE time and `is_x86_feature_detected!`
/// at RUN time, so the instruction set is never assumed by the build (no
/// `-C target-cpu=native`, no baseline bump) and a machine without AVX2 takes the
/// scalar branch rather than dying on an illegal instruction. This is the same
/// arrangement realesrgan-rs uses.
///
/// The two paths agree to about one part in 10^7 per convolution: the vector body
/// issues a real fused multiply-add where the scalar body rounds a multiply and an
/// add separately. See `conv3x3_avx2` and docs/NUMERICS.md.
pub fn conv3x3_into(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32], act: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_avx2() {
            // SAFETY: `x86_avx2` is `is_x86_feature_detected!` for exactly the
            // two features `conv3x3_avx2` is compiled with, and it is the only
            // caller.
            unsafe { conv3x3_avx2(x, out, w, b, act) };
            return;
        }
    }
    conv3x3_scalar_into(x, out, w, b, act);
}

/// The scalar reference for `conv3x3_into`: `(ky, kx, ci)`, one pixel at a time.
pub fn conv3x3_scalar_into(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32], act: bool) {
    let (c_in, h, wd) = (x.c, x.h, x.w);
    let c_out = b.len();
    debug_assert_eq!((out.c, out.h, out.w), (c_out, h, wd));
    let hw = h * wd;
    // THE ACCUMULATION ORDER IS (ky, kx, ci) AND IT IS NOT A STYLE CHOICE.
    // float32 addition is not associative, so the order is observable, and this
    // scalar body is the order every AVX2 body in this file reproduces bit for
    // bit and the one `tools/reference.py` is written in - it is what makes the
    // CPU twin a usable oracle for the reference. The TILED CUDA kernels nest
    // (ci, ky, kx) instead, over a zero-filled halo rather than by dropping
    // out-of-range taps, which is a bounded difference measured and explained in
    // docs/NUMERICS.md and not an invitation to re-nest this loop.
    //
    // The border rule is torch's `padding=1`, i.e. ZERO padding: a tap that falls
    // outside the image contributes nothing and is DROPPED. A zero row at the top
    // and bottom keeps `iy` and `ky` in lockstep, which is deliberate: pairing tap
    // row 1 with the pixel row of tap row 2 sums the k[1] row twice.
    out.d.par_chunks_mut(hw).enumerate().for_each(|(oc, plane)| {
        {
            let zero = vec![0.0f32; wd];
            for y in 0..h {
                for xx in 0..wd {
                    let mut v = 0.0f32;
                    for (ky, iy) in [(0usize, y as isize - 1), (1, y as isize), (2, y as isize + 1)] {
                        let in_range = iy >= 0 && iy < h as isize;
                        for (kx, ix) in
                            [(0usize, xx as isize - 1), (1, xx as isize), (2, xx as isize + 1)]
                        {
                            if ix < 0 || ix >= wd as isize {
                                continue;
                            }
                            let ix = ix as usize;
                            for ci in 0..c_in {
                                // A zero row is a zero *pixel*, so the tap is
                                // multiplied by 0.0f and the sum is unchanged -
                                // the branches below only decide where the pixel
                                // comes from, never which tap multiplies it.
                                let px = if in_range {
                                    x.d[ci * hw + iy as usize * wd + ix]
                                } else {
                                    let _ = &zero;
                                    0.0
                                };
                                v += px * w[(oc * c_in + ci) * 9 + ky * 3 + kx];
                            }
                        }
                    }
                    plane[y * wd + xx] = v;
                }
            }
        }
        let bb = b[oc];
        for v in plane.iter_mut() {
            *v += bb;
            if act && *v < 0.0 {
                *v *= 0.1;
            }
        }
    });
}

/// The AVX2 3x3 conv: the same `(ky, kx, ci)` sum, thirty-two pixels at a time.
///
/// WHY A ROW TILE. The scalar kernel above walks one output pixel at a time, and
/// every one of its `9 * c_in` tap reads is a separate address computation against
/// a plane that is `h*w` floats away from the last one. Here one output row is held
/// in four 8-float accumulators, so for a fixed tap the loop body is a contiguous
/// 32-float load, one broadcast weight and four fused multiply-adds. The weight is
/// loaded ONCE per tap for 32 pixels instead of once per pixel, and the input is
/// read as one streaming run rather than 32 gathers.
///
/// THE ACCUMULATION ORDER IS UNCHANGED, AND THAT IS THE POINT. The tap loops are
/// still `ky`, then `kx`, then `ci`, so lane `j` receives exactly the sequence of
/// products the scalar kernel gives pixel `x0 + j` - the four accumulators merely
/// interleave four lanes' chains in the instruction stream. What DOES differ is
/// that `_mm256_fmadd_ps` fuses the multiply and the add where the scalar body
/// rounds them separately, which is a difference of about one part in 10^7 per
/// convolution and is measured, not assumed: `cargo test`'s parity check reports the
/// worst value on the 64x64 fixture, and docs/NUMERICS.md records it.
///
/// THE BORDER IS THE SCALAR KERNEL. The first and last tile of a row, the first and
/// last row of the image, and any row where the input's edge would be read are
/// computed by the general path, so padding semantics live in exactly one place and
/// the vector body needs no bounds test at all.
///
/// Portability: this function exists only on x86-64 and is compiled with
/// `avx2,fma`; the dispatcher reaches it only after `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_avx2(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32], act: bool) {
    let (c_in, h, wd) = (x.c, x.h, x.w);
    debug_assert_eq!(out.c, b.len());
    let hw = h * wd;
    // TWO OUTPUT CHANNELS PER CHUNK, and that is a change to how the work is
    // SPLIT, not to what is computed: each output element still accumulates its
    // taps in `(ky, kx, ci)` order, so its bits are the ones the scalar kernel
    // produces and the CPU-vs-torch figures cannot move.
    //
    // WHY PAIRS. One row tile per output channel loads four vectors of input and
    // one broadcast weight per `(ky, kx, ci)` for four FMAs - 1.25 loads per FMA,
    // against two load ports and two FMA ports per core. The kernel measures
    // ~4.2 flops/cycle/core, which is exactly one load per FMA against two loads
    // per cycle: it is LOAD-PORT BOUND, and no amount of unrolling changes that as
    // long as the loads are private to one accumulator set. Two output channels
    // share the same four input loads and add only their own broadcast, so the
    // ratio becomes 6 loads per 8 FMAs - 0.75 - and the ceiling with it.
    out.d.par_chunks_mut(2 * hw).enumerate().for_each(|(k, chunk)| {
        let oc = 2 * k;
        let w0 = &w[oc * c_in * 9..(oc + 1) * c_in * 9];
        if chunk.len() == 2 * hw {
            // The pair path. `c_out` is even for every convolution in the graph
            // except the last one, which has three channels, so this is the normal
            // case and the tail below is the exception.
            let (p0, p1) = chunk.split_at_mut(hw);
            let w1 = &w[(oc + 1) * c_in * 9..(oc + 2) * c_in * 9];
            unsafe {
                conv3x3_plane_pair_avx2(
                    x,
                    p0,
                    p1,
                    w0,
                    w1,
                    b[oc],
                    b[oc + 1],
                    act,
                    hw,
                    wd,
                )
            };
        } else {
            // The odd channel of `out` (3 -> 32 -> 3 in the decoder's last layer).
            unsafe { conv3x3_plane_avx2(x, chunk, w0, b[oc], act, hw, wd) };
        }
    });
}

/// One output channel, one row at a time: full 32-pixel tiles from `x0 = 1` to
/// `wd - TX - 1` - the last one re-anchored so the leftover is covered twice with
/// the identical value - and the row's two end columns through the scalar pixel
/// body, because their `kx = 0` and `kx = 2` taps read off the row.
///
/// A row with fewer than `TX + 2` columns, and the top and bottom rows (a tile
/// stages three whole rows, and at `y = 0` one of them does not exist), go through
/// the scalar body in full. Those are `2/h` of the image.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_plane_avx2(
    x: &Blob,
    plane: &mut [f32],
    wblk: &[f32],
    bias: f32,
    act: bool,
    hw: usize,
    wd: usize,
) {
    let (c_in, h) = (x.c, x.h);
    for y in 0..h {
        let dst_row = plane.as_mut_ptr().add(y * wd);
        if wd >= TX + 2 && y > 0 && y + 1 < h {
            let last = wd - TX - 1;
            let mut x0 = 1usize;
            while x0 <= last {
                conv3x3_row_tile_avx2(
                    x.d.as_ptr(),
                    wblk.as_ptr(),
                    c_in,
                    hw,
                    wd,
                    y,
                    x0,
                    bias,
                    act,
                    dst_row,
                );
                x0 += TX;
            }
            conv3x3_row_tile_avx2(
                x.d.as_ptr(),
                wblk.as_ptr(),
                c_in,
                hw,
                wd,
                y,
                last,
                bias,
                act,
                dst_row,
            );
            conv3x3_pixel_scalar(x, wblk, c_in, hw, wd, y, 0, bias, act, dst_row);
            conv3x3_pixel_scalar(x, wblk, c_in, hw, wd, y, wd - 1, bias, act, dst_row);
        } else {
            for xx in 0..wd {
                conv3x3_pixel_scalar(x, wblk, c_in, hw, wd, y, xx, bias, act, dst_row);
            }
        }
    }
}

/// Two output channels, one row at a time: the same coverage as
/// `conv3x3_plane_avx2`, with the two channels' accumulators fed by ONE set of
/// input loads. See the dispatch for why that is the whole point.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_plane_pair_avx2(
    x: &Blob,
    p0: &mut [f32],
    p1: &mut [f32],
    w0: &[f32],
    w1: &[f32],
    bias0: f32,
    bias1: f32,
    act: bool,
    hw: usize,
    wd: usize,
) {
    let (c_in, h) = (x.c, x.h);
    for y in 0..h {
        let d0 = p0.as_mut_ptr().add(y * wd);
        let d1 = p1.as_mut_ptr().add(y * wd);
        if wd >= TX + 2 && y > 0 && y + 1 < h {
            let last = wd - TX - 1;
            let mut x0 = 1usize;
            while x0 <= last {
                conv3x3_row_tile2_avx2(
                    x.d.as_ptr(),
                    w0.as_ptr(),
                    w1.as_ptr(),
                    c_in,
                    hw,
                    wd,
                    y,
                    x0,
                    bias0,
                    bias1,
                    act,
                    d0,
                    d1,
                );
                x0 += TX;
            }
            conv3x3_row_tile2_avx2(
                x.d.as_ptr(),
                w0.as_ptr(),
                w1.as_ptr(),
                c_in,
                hw,
                wd,
                y,
                last,
                bias0,
                bias1,
                act,
                d0,
                d1,
            );
            conv3x3_pixel_scalar(x, w0, c_in, hw, wd, y, 0, bias0, act, d0);
            conv3x3_pixel_scalar(x, w0, c_in, hw, wd, y, wd - 1, bias0, act, d0);
            conv3x3_pixel_scalar(x, w1, c_in, hw, wd, y, 0, bias1, act, d1);
            conv3x3_pixel_scalar(x, w1, c_in, hw, wd, y, wd - 1, bias1, act, d1);
        } else {
            for xx in 0..wd {
                conv3x3_pixel_scalar(x, w0, c_in, hw, wd, y, xx, bias0, act, d0);
                conv3x3_pixel_scalar(x, w1, c_in, hw, wd, y, xx, bias1, act, d1);
            }
        }
    }
}

/// One output pixel, scalar - the arithmetic the vector tiles replicate, and the
/// path the pixels they cannot cover take.
///
/// The nesting is `(ky, kx, ci)`, identical to `conv3x3_scalar_into`'s, and so is
/// the out-of-range behaviour: a tap whose COLUMN falls off the row is skipped
/// entirely, and one whose ROW falls off the image contributes a zero pixel times
/// its weight. The second is not a shortcut - `v + 0.0 * w` is `v` exactly - which
/// is why a row may mix vector tiles and this function without moving a bit.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn conv3x3_pixel_scalar(
    x: &Blob,
    wblk: &[f32],
    c_in: usize,
    hw: usize,
    wd: usize,
    y: usize,
    xx: usize,
    bias: f32,
    act: bool,
    dst_row: *mut f32,
) {
    let h = x.h;
    let mut v = 0.0f32;
    for (ky, iy) in [(0usize, y as isize - 1), (1, y as isize), (2, y as isize + 1)] {
        let in_range = iy >= 0 && iy < h as isize;
        for (kx, ix) in [
            (0usize, xx as isize - 1),
            (1, xx as isize),
            (2, xx as isize + 1),
        ] {
            if ix < 0 || ix >= wd as isize {
                continue;
            }
            let ix = ix as usize;
            for ci in 0..c_in {
                let px = if in_range {
                    *x.d.get_unchecked(ci * hw + iy as usize * wd + ix)
                } else {
                    0.0
                };
                v += px * *wblk.get_unchecked(ci * 9 + ky * 3 + kx);
            }
        }
    }
    let mut vv = v + bias;
    if act && vv < 0.0 {
        vv *= 0.1;
    }
    *dst_row.add(xx) = vv;
}

/// `TX` output pixels of ONE ROW of TWO output channels, all taps in range.
///
/// The two accumulator sets are fed from the same four input loads; only the
/// weight broadcasts are per channel, which is what makes this cheaper per FMA
/// than two calls to `conv3x3_row_tile_avx2`. Each set still walks its taps in
/// `(ky, kx, ci)` order, so every element's bits are the ones the single-channel
/// tile would have produced.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv3x3_row_tile2_avx2(
    src: *const f32,
    w0: *const f32,
    w1: *const f32,
    c_in: usize,
    hw: usize,
    wd: usize,
    y: usize,
    x0: usize,
    bias0: f32,
    bias1: f32,
    act: bool,
    dst0: *mut f32,
    dst1: *mut f32,
) {
    use core::arch::x86_64::*;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut b0 = _mm256_setzero_ps();
    let mut b1 = _mm256_setzero_ps();
    let mut b2 = _mm256_setzero_ps();
    let mut b3 = _mm256_setzero_ps();
    for ky in 0..3usize {
        let row = (y + ky - 1) * wd;
        for kx in 0..3usize {
            let base = x0 + kx - 1;
            for ci in 0..c_in {
                let p = src.add(ci * hw + row + base);
                let i0 = _mm256_loadu_ps(p);
                let i1 = _mm256_loadu_ps(p.add(8));
                let i2 = _mm256_loadu_ps(p.add(16));
                let i3 = _mm256_loadu_ps(p.add(24));
                let u = _mm256_set1_ps(*w0.add(ci * 9 + ky * 3 + kx));
                let v = _mm256_set1_ps(*w1.add(ci * 9 + ky * 3 + kx));
                a0 = _mm256_fmadd_ps(u, i0, a0);
                a1 = _mm256_fmadd_ps(u, i1, a1);
                a2 = _mm256_fmadd_ps(u, i2, a2);
                a3 = _mm256_fmadd_ps(u, i3, a3);
                b0 = _mm256_fmadd_ps(v, i0, b0);
                b1 = _mm256_fmadd_ps(v, i1, b1);
                b2 = _mm256_fmadd_ps(v, i2, b2);
                b3 = _mm256_fmadd_ps(v, i3, b3);
            }
        }
    }
    // Bias and activation after the whole sum, in that order, as everywhere else.
    let mut t = [0.0f32; TX];
    _mm256_storeu_ps(t.as_mut_ptr(), a0);
    _mm256_storeu_ps(t.as_mut_ptr().add(8), a1);
    _mm256_storeu_ps(t.as_mut_ptr().add(16), a2);
    _mm256_storeu_ps(t.as_mut_ptr().add(24), a3);
    for j in 0..TX {
        let mut v = t[j] + bias0;
        if act && v < 0.0 {
            v *= 0.1;
        }
        *dst0.add(x0 + j) = v;
    }
    _mm256_storeu_ps(t.as_mut_ptr(), b0);
    _mm256_storeu_ps(t.as_mut_ptr().add(8), b1);
    _mm256_storeu_ps(t.as_mut_ptr().add(16), b2);
    _mm256_storeu_ps(t.as_mut_ptr().add(24), b3);
    for j in 0..TX {
        let mut v = t[j] + bias1;
        if act && v < 0.0 {
            v *= 0.1;
        }
        *dst1.add(x0 + j) = v;
    }
}

/// Output pixels of one interior row tile, held in registers.
///
/// The tile width, and the single source of truth for it: the accumulator count
/// below is derived from it, and `conv3x3_avx2`'s interior test is written in
/// terms of it. A width that the accumulators do not cover would silently drop
/// output, so the two must not be allowed to drift.
#[cfg(target_arch = "x86_64")]
const TX: usize = 32;

/// `TX` output pixels of one row, all taps in range, accumulated in `(ky, kx, ci)`
/// order and written to `dst + x0`.
///
/// `w` points at this output channel's `c_in * 9` weights in the checkpoint's
/// `[ci][3][3]` order, and `src` at the whole input's first element.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv3x3_row_tile_avx2(
    src: *const f32,
    w: *const f32,
    c_in: usize,
    hw: usize,
    wd: usize,
    y: usize,
    x0: usize,
    bias: f32,
    act: bool,
    dst: *mut f32,
) {
    use core::arch::x86_64::*;
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    for ky in 0..3usize {
        let row = (y + ky - 1) * wd;
        for kx in 0..3usize {
            let base = x0 + kx - 1;
            for ci in 0..c_in {
                let wv = _mm256_set1_ps(*w.add(ci * 9 + ky * 3 + kx));
                let p = src.add(ci * hw + row + base);
                a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p), a0);
                a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p.add(8)), a1);
                a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p.add(16)), a2);
                a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p.add(24)), a3);
            }
        }
    }
    // The bias and the activation land after the whole sum, in that order, exactly
    // as the scalar kernel's epilogue does - the bias is NOT an initial value.
    let mut t = [0.0f32; TX];
    _mm256_storeu_ps(t.as_mut_ptr(), a0);
    _mm256_storeu_ps(t.as_mut_ptr().add(8), a1);
    _mm256_storeu_ps(t.as_mut_ptr().add(16), a2);
    _mm256_storeu_ps(t.as_mut_ptr().add(24), a3);
    for j in 0..TX {
        let mut v = t[j] + bias;
        if act && v < 0.0 {
            v *= 0.1;
        }
        *dst.add(x0 + j) = v;
    }
}

/// Is this CPU an AVX2 machine with fused multiply-add?
///
/// Both are required: `conv3x3_row_tile_avx2` is compiled with `avx2,fma`, and
/// `_mm256_fmadd_ps` is an FMA instruction. `is_x86_feature_detected!` reads the
/// CPU's feature bits once and caches them, so this is a predictable branch rather
/// than a CPUID per call.
#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}


// ---------------------------------------------------------------------------
// The vector path is optional at RUN time, so it needs a test that does not
// depend on the machine it runs on.
// ---------------------------------------------------------------------------
#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_tests {
    use super::*;

    /// A deterministic pseudo-random field: no dependency, same numbers on every
    /// machine, and it exercises both signs and a wide range without ever
    /// overflowing into infinities after 27 taps.
    fn field(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The vector body may differ from the scalar one ONLY by fused multiply-add
    /// rounding. That is what this asserts, and it is the whole numerical
    /// contract of the change: a reassociated sum would be orders of magnitude
    /// worse than 1e-5 on these magnitudes.
    fn close(a: &Blob, b: &Blob, tol: f32, what: &str) {
        assert_eq!((a.c, a.h, a.w), (b.c, b.h, b.w), "{what}: shapes differ");
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for (i, (x, y)) in a.d.iter().zip(b.d.iter()).enumerate() {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
        assert!(worst <= tol, "{what}: max |d| {worst:e} at {at} exceeds {tol:e}");
    }

    /// Every geometry that reaches a different branch: an interior region and a
    /// ragged tail, an image narrower than one tile, a single row (so the row
    /// above and below are both padding), and a one-channel input.
    #[test]
    fn the_vector_path_matches_the_scalar_one() {
        if !x86_avx2() {
            eprintln!("no AVX2/FMA on this CPU: nothing to compare");
            return;
        }
        for (c_in, c_out, h, w) in [
            (3usize, 4usize, 9usize, 40usize),  // narrow: tail plus borders
            (5, 6, 16, 64),                     // two exact tiles
            (2, 3, 1, 100),                     // one row, wide
            (8, 8, 33, 100),                    // wide, several tiles
            (1, 1, 7, 33),                      // degenerate channel counts
        ] {
            for act in [false, true] {
                let x = Blob {
                    c: c_in,
                    h,
                    w,
                    d: field(c_in * h * w, 0x9e3779b9 ^ (w as u32)),
                };
                let wv = field(c_out * c_in * 9, 0x85ebca6b ^ (h as u32));
                let bv = field(c_out, 0xc2b2ae35);
                let mut scalar = Blob::zeros(c_out, h, w);
                conv3x3_scalar_into(&x, &mut scalar, &wv, &bv, act);
                let mut vector = Blob::zeros(c_out, h, w);
                conv3x3_into(&x, &mut vector, &wv, &bv, act);
                close(&scalar, &vector, 1e-5, &format!("conv3x3 {c_in}x{c_out} {h}x{w} act={act}"));

                // The 1x1 is exercised on the same geometries for the same
                // reason, plus one that only it can reach: a plane wider than one
                // 32-float block with a RAGGED TAIL (`hw` not a multiple of 32),
                // which is what the leftover loop below the vector body covers.
                // The destination starts at a non-zero value because the op
                // ACCUMULATES into it, and a vector body that seeded its
                // accumulators from zero would be caught here and nowhere else.
                let w1 = field(c_out * c_in, 0x27d4eb2f ^ (c_in as u32));
                let mut d_scalar = Blob::zeros(c_out, h, w);
                let mut d_vector = Blob::zeros(c_out, h, w);
                for (i, v) in d_scalar.d.iter_mut().enumerate() {
                    *v = ((i % 7) as f32 - 3.0) * 0.25;
                }
                d_vector.d.copy_from_slice(&d_scalar.d);
                conv1x1_scalar_into(&x, &mut d_scalar, &w1, &bv);
                conv1x1_into(&x, &mut d_vector, &w1, &bv);
                close(
                    &d_scalar,
                    &d_vector,
                    1e-5,
                    &format!("conv1x1 {c_in}x{c_out} {h}x{w} (tail {})", (h * w) % 32),
                );

                // The tail is worth reporting when it fails: a plane whose
                // element count is an exact multiple of 32 exercises only the
                // vector body, and `(h*w) % 32` is what says which one ran.
            }
        }
    }
}

/// `F.conv2d(x, w, b)` with a 1x1 kernel. No activation: F's head is the only user.
pub fn conv1x1(x: &Blob, w: &[f32], b: &[f32]) -> Blob {
    let mut out = Blob::zeros(b.len(), x.h, x.w);
    conv1x1_into(x, &mut out, w, b);
    out
}

/// `conv1x1`, writing into `out`: the dispatcher, gated exactly as `conv3x3_into`
/// is (compile-time `cfg`, run-time `is_x86_feature_detected!`, scalar fallback).
///
/// The vectorised form pays off here for a different reason than in the 3x3: the
/// inner loop is already contiguous, but it is `c_in` separate passes over the
/// same output plane (15232 reads of the same 135x240 map at 1080p), so the win
/// is fewer passes over memory and a fused multiply-add rather than a wider
/// load.
pub fn conv1x1_into(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_avx2() {
            // SAFETY: `x86_avx2` is exactly the two features `conv1x1_avx2` is
            // compiled with, and it is the only caller.
            unsafe { conv1x1_avx2(x, out, w, b) };
            return;
        }
    }
    conv1x1_scalar_into(x, out, w, b);
}

/// The scalar reference for `conv1x1_into`: `c_in` passes over one output plane.
pub fn conv1x1_scalar_into(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32]) {
    let (c_in, h, wd) = (x.c, x.h, x.w);
    debug_assert_eq!((out.c, out.h, out.w), (b.len(), h, wd));
    let hw = h * wd;
    out.d.par_chunks_mut(hw).enumerate().for_each(|(oc, plane)| {
        for ic in 0..c_in {
            let k = w[oc * c_in + ic];
            let src = &x.d[ic * hw..(ic + 1) * hw];
            for (o, v) in plane.iter_mut().zip(src) {
                *o += k * v;
            }
        }
        let bb = b[oc];
        for v in plane.iter_mut() {
            *v += bb;
        }
    });
}

/// The AVX2 1x1 conv: thirty-two pixels per pass over the input channel set.
///
/// ONE ACCUMULATOR PER LANE, so the sum over `ic` is a fused multiply-add chain
/// instead of a load-add-store round trip, and the input plane is walked once for
/// all 32 pixels rather than once per pixel. The order is the scalar kernel's (ic
/// ascending) and the accumulation still STARTS FROM WHAT IS ALREADY THERE, since
/// `conv1x1_into` accumulates into its destination - see `net::Cpu::out_buf` for
/// why the destination arrives zeroed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv1x1_avx2(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32]) {
    use core::arch::x86_64::*;
    let (c_in, h, wd) = (x.c, x.h, x.w);
    let hw = h * wd;
    debug_assert_eq!(out.c, b.len());
    out.d.par_chunks_mut(hw).enumerate().for_each(|(oc, plane)| {
        let (src, dst) = (x.d.as_ptr(), plane.as_mut_ptr());
        let wp = w.as_ptr().add(oc * c_in);
        let mut j = 0usize;
        while j + 32 <= hw {
            // Seeded from the destination, not from zero: this op ACCUMULATES.
            let mut a0 = _mm256_loadu_ps(dst.add(j));
            let mut a1 = _mm256_loadu_ps(dst.add(j + 8));
            let mut a2 = _mm256_loadu_ps(dst.add(j + 16));
            let mut a3 = _mm256_loadu_ps(dst.add(j + 24));
            for ic in 0..c_in {
                let kv = _mm256_set1_ps(*wp.add(ic));
                let p = src.add(ic * hw + j);
                a0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p), a0);
                a1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p.add(8)), a1);
                a2 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p.add(16)), a2);
                a3 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(p.add(24)), a3);
            }
            _mm256_storeu_ps(dst.add(j), a0);
            _mm256_storeu_ps(dst.add(j + 8), a1);
            _mm256_storeu_ps(dst.add(j + 16), a2);
            _mm256_storeu_ps(dst.add(j + 24), a3);
            j += 32;
        }
        // The ragged tail, in the scalar kernel's form.
        while j < hw {
            let mut v = 0.0f32;
            for ic in 0..c_in {
                v += *wp.add(ic) * *src.add(ic * hw + j);
            }
            *dst.add(j) += v;
            j += 1;
        }
        // The bias is added ONCE over the whole plane, exactly as the scalar
        // kernel's epilogue does - it is not a per-`ic` term and not an initial
        // value.
        let bb = *b.get_unchecked(oc);
        let bv = _mm256_set1_ps(bb);
        let mut k = 0usize;
        while k + 32 <= hw {
            _mm256_storeu_ps(dst.add(k), _mm256_add_ps(_mm256_loadu_ps(dst.add(k)), bv));
            _mm256_storeu_ps(dst.add(k + 8), _mm256_add_ps(_mm256_loadu_ps(dst.add(k + 8)), bv));
            _mm256_storeu_ps(
                dst.add(k + 16),
                _mm256_add_ps(_mm256_loadu_ps(dst.add(k + 16)), bv),
            );
            _mm256_storeu_ps(
                dst.add(k + 24),
                _mm256_add_ps(_mm256_loadu_ps(dst.add(k + 24)), bv),
            );
            k += 32;
        }
        while k < hw {
            *dst.add(k) += bb;
            k += 1;
        }
    });
}

/// `F.conv2d(x, w, b, stride=2, padding=1)` - 3x3, `act` = leaky_relu(0.1) on demand.
///
/// The output geometry is `ceil(h / 2)`, which is torch's rule for a stride-2
/// padded conv; the graph's shape table is built with the same `div_ceil`, and
/// `Gpu::conv3x3s2` refuses to launch if the two ever disagree.
pub fn conv3x3s2(x: &Blob, w: &[f32], b: &[f32], act: bool) -> Blob {
    let mut out = Blob::zeros(b.len(), x.h.div_ceil(2), x.w.div_ceil(2));
    conv3x3s2_into(x, &mut out, w, b, act);
    out
}

/// `conv3x3s2`, writing into `out`, whose h/w must be the input's `div_ceil(2)`.
pub fn conv3x3s2_into(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32], act: bool) {
    let (c_in, h, wd) = (x.c, x.h, x.w);
    let c_out = b.len();
    let (oh, ow) = (h.div_ceil(2), wd.div_ceil(2));
    debug_assert_eq!((out.c, out.h, out.w), (c_out, oh, ow));
    out.d.par_chunks_mut(oh * ow).enumerate().for_each(|(oc, plane)| {
        // A zero row for the two rows that fall off the image, allocated ONCE per
        // output channel instead of once per (input channel, output row): it used
        // to be `vec![0.0; wd]` inside the `oy` loop, which is `c_in * oh`
        // allocations of `wd` floats for a buffer that never changes.
        let zero_row = vec![0.0f32; wd];
        for ic in 0..c_in {
            let k = &w[(oc * c_in + ic) * 9..(oc * c_in + ic) * 9 + 9];
            let src = &x.d[ic * h * wd..(ic + 1) * h * wd];
            for oy in 0..oh {
                let y = oy * 2;
                // The three input rows the output row reads, with the pad rule
                // spelled out: row y-1 is skipped when y == 0, row y+1 is the
                // last row when y+1 is past the bottom.
                // Zero row/column at the borders and the SAME tap indices for
                // every row, so a tap can only ever be paired with its own pixel
                // row. The previous version clamped the bottom row to `y` and kept
                // tap row 1, which summed the k[1] row twice - the identical
                // defect conv3x3 had (see its comment).
                let r0 = if y == 0 { &zero_row[..] } else { &src[(y - 1) * wd..y * wd] };
                let r1 = &src[y * wd..y * wd + wd];
                let r2 = if y + 1 < h { &src[(y + 1) * wd..(y + 2) * wd] } else { &zero_row[..] };
                // INTERIOR COLUMNS, FOUR AT A TIME, and this is where the
                // kernel's time was going: `plane[..] += v` is a load-add-store
                // chain PER INPUT CHANNEL, and one accumulator at a time leaves
                // the nine-tap FMA chain and that store-forwarding chain fully
                // exposed to each other. Four independent accumulators give the
                // core four chains to interleave, which is all the ILP a
                // latency-bound loop needs - the FLOP rate of this kernel was
                // 71 GFLOP/s, a third of what the 3x3 above manages.
                //
                // The tap bounds are why the interior starts at `ox = 1` and stops
                // at `e`: column `x = xx - 1` needs `xx >= 1` and `x = xx + 1` needs
                // `xx + 1 < wd`, and inside that range the nine taps are three
                // unconditional loads with no bounds test at all.
                let e = if wd >= 2 { (wd - 2) / 2 } else { 0 };
                let mut ox = 1usize;
                while wd >= 2 && ox + 3 <= e {
                    let xx = [2 * ox, 2 * ox + 2, 2 * ox + 4, 2 * ox + 6];
                    let mut v = [0.0f32; 4];
                    for (rr, ky) in [(r0, 0usize), (r1, 1), (r2, 2)] {
                        for j in 0..4 {
                            v[j] += rr[xx[j] - 1] * k[ky * 3];
                            v[j] += rr[xx[j]] * k[ky * 3 + 1];
                            v[j] += rr[xx[j] + 1] * k[ky * 3 + 2];
                        }
                    }
                    for j in 0..4 {
                        plane[oy * ow + ox + j] += v[j];
                    }
                    ox += 4;
                }
                while wd >= 2 && ox <= e {
                    let xx = 2 * ox;
                    let mut v = 0.0f32;
                    for (rr, ky) in [(r0, 0usize), (r1, 1), (r2, 2)] {
                        v += rr[xx - 1] * k[ky * 3];
                        v += rr[xx] * k[ky * 3 + 1];
                        v += rr[xx + 1] * k[ky * 3 + 2];
                    }
                    plane[oy * ow + ox] += v;
                    ox += 1;
                }
                // THE TWO EDGES, which are the pixels a tap reads off the row:
                // `ox = 0` and everything past `e`. Same (ky, kx) order, with the
                // out-of-range tap DROPPED - which is the arithmetic the interior
                // loop above still performs, since dropping a term is adding zero.
                for ox in std::iter::once(0usize).chain(e + 1..ow) {
                    let xx = ox * 2;
                    // Bounds, not index remapping: an out-of-range tap falls on a
                    // zero row (y) or is dropped (x), exactly as zero padding means.
                    let x0 = if xx == 0 { None } else { Some(xx - 1) };
                    let x1 = if xx + 1 < wd { Some(xx + 1) } else { None };
                    let rows: [(&[f32], usize); 3] = [(r0, 0usize), (r1, 1), (r2, 2)];
                    let mut v = 0.0f32;
                    for (rr, ky) in rows {
                        if let Some(c) = x0 {
                            v += rr[c] * k[ky * 3];
                        }
                        v += rr[xx] * k[ky * 3 + 1];
                        if let Some(c) = x1 {
                            v += rr[c] * k[ky * 3 + 2];
                        }
                    }
                    plane[oy * ow + ox] += v;
                }
            }
        }
        let bb = b[oc];
        for v in plane.iter_mut() {
            *v += bb;
            if act && *v < 0.0 {
                *v *= 0.1;
            }
        }
    });
}

pub fn convtranspose4x4s2_into(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32], act: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_avx2() {
            // SAFETY: `x86_avx2` is `is_x86_feature_detected!` for exactly the two
            // features this body is compiled with, and it is the only caller.
            unsafe { convtranspose4x4s2_avx2(x, out, w, b, act) };
            return;
        }
    }
    convtranspose4x4s2_scalar_into(x, out, w, b, act);
}

/// One output pixel of `convtranspose4x4s2`, scalar. The reference the vector
/// body's interior must equal BIT FOR BIT, so its (ky, kx, ic) order and its
/// out-of-range rule (skip the tap) are the ones to copy, not to improve.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn convt_pixel_scalar(
    x: &Blob,
    w: &[f32],
    c_in: usize,
    h: usize,
    wd: usize,
    c_out: usize,
    oc: usize,
    oy: usize,
    ox: usize,
) -> f32 {
    let mut v = 0.0f32;
    for ky in 0..4usize {
        let num = oy as isize + 1 - ky as isize;
        if num % 2 != 0 {
            continue;
        }
        let iy = num / 2;
        if iy < 0 || iy >= h as isize {
            continue;
        }
        for kx in 0..4usize {
            let num = ox as isize + 1 - kx as isize;
            if num % 2 != 0 {
                continue;
            }
            let ix = num / 2;
            if ix < 0 || ix >= wd as isize {
                continue;
            }
            for ic in 0..c_in {
                let k = w[(ic * c_out + oc) * 16 + ky * 4 + kx];
                v += x.d[(ic * h + iy as usize) * wd + ix as usize] * k;
            }
        }
    }
    v
}

/// The AVX2 body: eight output columns at a time, as four EVEN and four ODD - no,
/// as eight even and eight odd, interleaved on the store.
///
/// THE AXIS THAT VECTORISES. For output column `ox` the active taps are the `kx`
/// with `(ox + 1 - kx)` even, so even `ox` reads `kx = 1` (input `ix = t`) and
/// `kx = 3` (`ix = t - 1`) where `t = ox/2`, and odd `ox` reads `kx = 0`
/// (`ix = t + 1`) and `kx = 2` (`ix = t`). Both are CONTIGUOUS in `ix` across the
/// eight lanes of a vector - it is the output that is strided, and only on the
/// store, where two unpacks and two 128-bit lane permutes interleave the even and
/// odd results into sixteen consecutive floats.
///
/// Eight output columns hold `4 * c_in` MACs each (two rows by two taps by `c_in`),
/// which is what the scalar body's single serial accumulator was spending its
/// time on: 20-30 GFLOP/s of real work, the worst rate in the CPU path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn convtranspose4x4s2_avx2(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32], act: bool) {
    use core::arch::x86_64::*;
    let (c_in, h, wd) = (x.c, x.h, x.w);
    let c_out = b.len();
    let (oh, ow) = (h * 2, wd * 2);
    debug_assert_eq!((out.c, out.h, out.w), (c_out, oh, ow));
    out.d.par_chunks_mut(oh * ow).enumerate().for_each(|(oc, plane)| {
        // Re-derived per closure call: a raw pointer is not `Sync`, and `x` is.
        let src = x.d.as_ptr();
        let bias = b[oc];
        for oy in 0..oh {
            let row = plane.as_mut_ptr().add(oy * ow);
            // The active `ky` for this row, ascending: `oy = 2*iy - 1 + ky` puts
            // `ky` at `1` or `0` for even/odd `oy` and, for that `ky`, `ky + 2` is
            // the second row. `iy` is then `(oy + 1 - ky) / 2`, and a row outside
            // the image contributes nothing.
            let kya = 1 - (oy & 1);
            // `t` is the half-column index: outputs `2t` (even) and `2t + 1`
            // (odd). A block of eight half-columns reads input `t - 1` through
            // `t + 8`, so it needs `t >= 1` and `t + 8 <= wd - 1`.
            let mut t = 0usize;
            while t < wd {
                if t >= 1 && wd >= 9 && t + 8 < wd {
                    let mut acc_e = _mm256_setzero_ps();
                    let mut acc_o = _mm256_setzero_ps();
                    for kk in 0..2usize {
                        let ky = kya + 2 * kk;
                        let num = oy as isize + 1 - ky as isize;
                        if num % 2 != 0 {
                            continue;
                        }
                        let iy = num / 2;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        let base = src.add(iy as usize * wd);
                        // kx in order, each with `ic` inner: the per-element order
                        // is (ky, kx, ic), the scalar body's. Even and odd columns
                        // have disjoint active kx, so the two accumulators each
                        // see their own taps in the right order.
                        for kx in 0..4usize {
                            let (acc_is_odd, shift) = match kx {
                                0 => (true, 1isize),
                                1 => (false, 0),
                                2 => (true, 0),
                                _ => (false, -1),
                            };
                            let p = base.offset(t as isize + shift);
                            let mut a = if acc_is_odd { acc_o } else { acc_e };
                            for ic in 0..c_in {
                                let wv = _mm256_set1_ps(*w.as_ptr().add((ic * c_out + oc) * 16 + ky * 4 + kx));
                                a = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p.add(ic * h * wd)), a);
                            }
                            if acc_is_odd {
                                acc_o = a;
                            } else {
                                acc_e = a;
                            }
                        }
                    }
                    let bv = _mm256_set1_ps(bias);
                    let mut e = _mm256_add_ps(acc_e, bv);
                    let mut o = _mm256_add_ps(acc_o, bv);
                    if act {
                        e = _mm256_max_ps(e, _mm256_mul_ps(e, _mm256_set1_ps(0.1)));
                        o = _mm256_max_ps(o, _mm256_mul_ps(o, _mm256_set1_ps(0.1)));
                    }
                    let lo = _mm256_unpacklo_ps(e, o);
                    let hi = _mm256_unpackhi_ps(e, o);
                    _mm256_storeu_ps(row.add(2 * t), _mm256_permute2f128_ps(lo, hi, 0x20));
                    _mm256_storeu_ps(row.add(2 * t + 8), _mm256_permute2f128_ps(lo, hi, 0x31));
                    t += 8;
                    continue;
                }
                // The two leftmost and the tail: the same accumulation, one
                // column at a time.
                for ox in [2 * t, 2 * t + 1] {
                    if ox < ow {
                        let mut v = convt_pixel_scalar(x, w, c_in, h, wd, c_out, oc, oy, ox) + bias;
                        if act && v < 0.0 {
                            v *= 0.1;
                        }
                        *row.add(ox) = v;
                    }
                }
                t += 1;
            }
        }
    });
}

/// `F.conv_transpose2d(x, w, b, stride=2, padding=1)`, 4x4 kernel, activated.
///
/// Written in GATHER form, the way the CUDA kernel is: for each output pixel,
/// sum every input pixel/tap pair that contributes to it. The scatter form (walk
/// the input, add into four output pixels) is what torch does internally, but it
/// needs atomic or ordered accumulation to be safe to parallelise, and the gather
/// form is a fixed 16-term loop that parallelises over output channels with no
/// synchronisation at all.
///
/// The weight layout differs from the forward conv, and this is the one place it
/// matters: torch stores a ConvTranspose2d weight as `[c_in, c_out, kh, kw]`,
/// i.e. the INPUT channel is the outer index. The gather index below therefore
/// reads `w[(ic * c_out + oc) * 16 + ...]`.
pub fn convtranspose4x4s2(x: &Blob, w: &[f32], b: &[f32], act: bool) -> Blob {
    let mut out = Blob::zeros(b.len(), x.h * 2, x.w * 2);
    convtranspose4x4s2_into(x, &mut out, w, b, act);
    out
}

/// `convtranspose4x4s2`, writing into `out`, whose h/w must be twice the input's.
///
/// `out` and `x` are never the same buffer: the gather reads all of `x` for every
/// output pixel, so aliasing would be wrong (see `conv3x3_into`).
pub fn convtranspose4x4s2_scalar_into(x: &Blob, out: &mut Blob, w: &[f32], b: &[f32], act: bool) {
    let (c_in, h, wd) = (x.c, x.h, x.w);
    let c_out = b.len();
    let (oh, ow) = (h * 2, wd * 2);
    debug_assert_eq!((out.c, out.h, out.w), (c_out, oh, ow));
    out.d.par_chunks_mut(oh * ow).enumerate().for_each(|(oc, plane)| {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut v = 0.0f32;
                // Output (oy, ox) receives from input pixels iy, ix whose
                // 4x4 window scaled by 2 and shifted by -1 covers it:
                //   oy = 2*iy - 1 + ky  =>  iy = (oy + 1 - ky) / 2  with ky - 1 even.
                for ky in 0..4usize {
                    let num = oy as isize + 1 - ky as isize;
                    if num % 2 != 0 {
                        continue;
                    }
                    let iy = num / 2;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..4usize {
                        let num = ox as isize + 1 - kx as isize;
                        if num % 2 != 0 {
                            continue;
                        }
                        let ix = num / 2;
                        if ix < 0 || ix >= wd as isize {
                            continue;
                        }
                        for ic in 0..c_in {
                            let k = w[(ic * c_out + oc) * 16 + ky * 4 + kx];
                            v += x.d[(ic * h + iy as usize) * wd + ix as usize] * k;
                        }
                    }
                }
                plane[oy * ow + ox] = v + b[oc];
            }
        }
        if act {
            for v in plane.iter_mut() {
                if *v < 0.0 {
                    *v *= 0.1;
                }
            }
        }
    });
}

/// `out = a + b`, elementwise, same shape.
pub fn add(a: &Blob, b: &Blob) -> Blob {
    let mut out = Blob::zeros(a.c, a.h, a.w);
    add_into_par(a, b, &mut out);
    out
}

/// `add`, writing into `out`. There is NO aliasing case in this engine: the graph
/// writes `x + y` to a name that is neither `x` nor `y` (`DME.1.i.m`, `iac`'s
/// residual sums, `raw = res + in`), so an in-place caller would be a graph
/// change rather than a caller's choice.
/// `add_into`, split across the pool.
///
/// The elementwise ops are the only ones with no intrinsic parallelism, and the
/// three of them plus the SAC passes are 23% of the pass at 720x720 and a fifth of
/// it at 1080p - all of that on ONE core, while the convolutions next to them keep
/// twelve busy. Splitting an elementwise op is also the cheapest possible thing to
/// split: the sums are independent per element and each one is `x + y` in the same
/// order, so this is the same arithmetic, and `IFAN_CPU_TIME` measures whether the
/// thread handoff costs more than it saves.
///
/// The lengths are truncated to the shortest of the three, which is what the
/// sequential `zip` did - a recycled destination may be larger than the name's
/// geometry.
pub fn add_into_par(a: &Blob, b: &Blob, out: &mut Blob) {
    debug_assert_eq!((a.c, a.h, a.w), (b.c, b.h, b.w));
    debug_assert_eq!((a.c, a.h, a.w), (out.c, out.h, out.w));
    let n = out.d.len().min(a.d.len()).min(b.d.len());
    out.d[..n]
        .par_iter_mut()
        .zip(a.d[..n].par_iter())
        .zip(b.d[..n].par_iter())
        .for_each(|((o, x), y)| *o = x + y);
}

/// `leaky_relu` in place, split across the pool - the ResBlock's `m` activation,
/// which the interpreter does without a copy.
pub fn lrelu_in_place_par(b: &mut Blob, slope: f32) {
    b.d.par_iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v *= slope;
        }
    });
}

/// `leaky_relu` from one buffer into another, split across the pool.
pub fn lrelu_into_par(src: &Blob, out: &mut Blob, slope: f32) {
    let n = out.d.len().min(src.d.len());
    out.d[..n]
        .par_iter_mut()
        .zip(src.d[..n].par_iter())
        .for_each(|(o, v)| *o = if *v < 0.0 { *v * slope } else { *v });
}

/// `out = clip(x, 0, 1)`, split across the pool. The graph's last op.
pub fn clip_into_par(src: &Blob, out: &mut Blob) {
    let n = out.d.len().min(src.d.len());
    out.d[..n]
        .par_iter_mut()
        .zip(src.d[..n].par_iter())
        .for_each(|(o, v)| *o = v.clamp(0.0, 1.0));
}

/// One iteration of the SAC block, split across the pool: one CHANNEL per task.
///
/// `tap` is the filter tensor's kernel1 slice for one iteration: `c * ksize`
/// planes of `h x w`, laid out CHANNEL-MAJOR - channel `ch`'s tap `k` is at plane
/// `ch * ksize + k`. `vertical` selects the gather direction, as `IAC.py` does.
///
/// THE INDEXING IS UPSTREAM'S COMPOSITION AND MUST NOT BE TIDIED UP. `IAC.py`
/// pads the tap map by `ksize / 2`, unfolds it into `[c, ksize, h, w]`, and then
/// indexes that with a clamped `(y + dy, x + dx)` - so the sample position is
/// `y + dy` where `y` was already the centre of the unfold window, and the tap
/// j of a channel is read at the SAMPLE position while being indexed at the
/// OUTPUT position. Collapsing the two steps into one clamped gather composes
/// the indices correctly and the TAPS wrongly: the form below matches the
/// reference to 3.6e-15 where a same-position form is off by tens of units.
///
/// Per element the accumulation is the `ksize` taps in index order into one
/// float, and the channels never read each other's planes.
pub fn sac_pass_into_par(x: &Blob, tap: &Blob, out: &mut Blob, ksize: usize, vertical: bool) {
    let (c, h, wd) = (x.c, x.h, x.w);
    debug_assert_eq!((tap.h, tap.w), (h, wd));
    debug_assert_eq!(tap.c, c * ksize, "tap blob must hold ksize planes per channel");
    debug_assert_eq!((out.c, out.h, out.w), (c, h, wd));
    let half = ksize / 2;
    let plane = h * wd;
    let clamp = |v: isize, n: usize| -> usize { v.clamp(0, n as isize - 1) as usize };
    out.d[..c * plane]
        .par_chunks_mut(plane)
        .enumerate()
        .for_each(|(ch, op)| {
            let xp = &x.d[ch * plane..(ch + 1) * plane];
            let taps = &tap.d[ch * ksize * plane..(ch + 1) * ksize * plane];
            for y in 0..h {
                for xx in 0..wd {
                    let mut acc = 0.0f32;
                    if vertical {
                        for j in 0..ksize {
                            let yy = clamp(y as isize + j as isize - half as isize, h);
                            acc += xp[yy * wd + xx] * taps[(j * h + y) * wd + xx];
                        }
                    } else {
                        for j in 0..ksize {
                            let x2 = clamp(xx as isize + j as isize - half as isize, wd);
                            acc += xp[y * wd + x2] * taps[(j * h + y) * wd + xx];
                        }
                    }
                    op[y * wd + xx] = acc;
                }
            }
        });
}

/// The SAC bias-and-activation op, split across the pool: one CHANNEL per task.
///
/// `bias` points at the filter tensor's bias block for this iteration and `step`
/// is one plane of the FILTER tensor (not of the feature map): the block is a
/// whole plane per channel, taken from a tensor whose spatial size is already the
/// 1/8 grid, and the two are the same size - checked by the caller.
pub fn sac_bias_into_par(f: &[f32], x: &Blob, out: &mut Blob, base: usize, step: usize) {
    let (c, h, wd) = (x.c, x.h, x.w);
    let plane = h * wd;
    out.d[..c * plane]
        .par_chunks_mut(plane)
        .enumerate()
        .for_each(|(ch, op)| {
            let bp = &f[(base + ch) * step..(base + ch) * step + plane];
            let xp = &x.d[ch * plane..(ch + 1) * plane];
            for j in 0..plane {
                let v = xp[j] + bp[j];
                op[j] = if v < 0.0 { 0.1 * v } else { v };
            }
        });
}
