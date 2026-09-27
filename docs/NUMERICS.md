# Numerical notes

Why the CPU and CUDA backends agree as closely as they do, the few places where
they deliberately do not, and how to check that a change has not moved either.
None of this is needed to run the engine; it is here for anyone editing a kernel.

## The two Rust backends are compared to each other, torch to both

The gate (`tools/gate.py`) compares three things on the same PNG: the CPU
interpreter in `src/net.rs` and the CUDA path `src/gpu.rs` + `cuda/ifan.cu`
against each other, and both against `tools/reference.py`, a PyTorch
transcription of the upstream network running on the original `.pytorch`
checkpoint.

Torch is the third opinion, not the oracle. Its convolution summation order is
whatever cuDNN or its CPU kernel chooses, it is free to change between releases,
and nothing in it is written to match an `acc += w * x` loop. The two Rust
backends, by contrast, ARE written to agree with each other, so the useful
question at the output is "do they match to float32 round-off", and the useful
question against torch is "is the difference at the 1/255 scale", i.e. could it
change an 8-bit pixel. Both are reported, and the tolerance on the third
comparison is `1.5/255`.

## The compiler is part of the arithmetic

`build.rs` compiles `cuda/ifan.cu` with `--fmad=true` (with `--ftz=false
--prec-div=true --prec-sqrt=true`, so denormals are kept and division and square
root are correctly rounded). `--fmad=true` means `acc += w * x` in a kernel
**contracts into a single-rounded FFMA** - the product is never rounded on its
own.

Rust does not contract float operations. There is no `fma`/`mul_add` anywhere in
`src/` or `cuda/`, so the CPU twin rounds twice where the kernel rounds once.

**This is the measured reason CPU-vs-CUDA sits at 1-3e-06 rather than at zero.**
It is deliberate: the CPU twin exists to be read next to `tools/reference.py`
and to be the simulation a graph bug shows up in, and writing it with explicit
`mul_add` would mean neither of those things while also making the CPU path
require an FMA-capable CPU. The gate's tolerance (2e-04 between the backends) is
two orders of magnitude above the observed difference, so an actual defect - the
defects found while building this engine were all in the 0.1-1.0 range - is
nowhere near the noise floor.

If the two backends must agree to the last bit, the fix is on the CUDA side:
compile the project kernels with `--fmad=false`. Nothing else changes, and the
CPU twin needs no edit. The cost is one extra rounding per multiply-add in every
convolution.

The two backends are also compiled with the same fast-math switches in one
respect that matters and one that does not: `if_sac_vertical` and
`if_sac_horizontal` normalise their weights with a plain `sum += tap`, which
`--fmad=false` would not touch, and the SAC kernels' index arithmetic is integer
throughout.

## Accumulation order is contractual

A convolution has one mathematically correct value and many correct summation
orders; in float32 they give different last bits. Since the gate compares the
two Rust backends, changing an order changes the reported difference, and it is
the kind of "tidy-up" that looks free. The orders that are fixed:

* the CPU's 3x3 convolutions accumulate `ky`, then `kx`, then `ci` - that is
  `src/kernels.rs`'s nesting, and it is the order `tools/reference.py` is written
  in;
* the CPU's transposed convolution accumulates in the same order, so the two
  lists of taps below are equivalent orderings of the same sum;
* **the CUDA side's 3x3 and stride-2 convolutions do NOT use that order.** They
  run through `if_conv_tile_kernel` (`cuda/ifan.cu`), which stages the input in
  shared memory and walks the channel tile in the OUTER loop and the kernel window
  inside it - `ci`, then `ky`, then `kx` - because that is what lets it keep eight
  independent accumulators per thread instead of one. It also ZERO-FILLS the staged
  halo rather than skipping out-of-range taps, so a boundary output sums 27 terms
  where the CPU twin sums 22 of them and adds five zeros.

  That pair of differences is worth 2-3e-07 of the CUDA-vs-torch figure at 64x64
  (1.431e-06 with the plain order against 1.729e-06 with the tiled one; 8.345e-07
  against 1.073e-06 at 256x256, 2.623e-06 against 2.801e-06 at 720x720): adding
  zeros to a sum does not change it, but summing the same terms in a different
  order does, in the last bits. It is a deliberate trade - a convolution several
  times faster for a change far below the 1/255 the output PNG can express, with
  CPU-vs-torch (the leg that does not involve the tiled kernels) unmoved;
* the transposed convolution's tiled kernel nests (ci, tap) over its four active
  taps in the order (dy, dx) = (0,0), (0,1), (1,0), (1,1), so CUDA-vs-torch at
  64x64 IMPROVED when it replaced the one-output-per-thread kernel (1.729e-06 ->
  1.609e-06) - the activation is applied once in the epilogue rather than twice
  through a scratch that aliased `out` (see `src/gpu.rs`'s `convt4x4s2`);
* the SAC convolutions on the CUDA side still nest as the CPU does;
* `src/kernels.rs`'s `convtranspose4x4s2` dispatches on the PARITY of the output
  row and column rather than filtering out-of-range taps in a loop, because that
  is what the CUDA kernel's shifted-tap formulation does;
* the two SAC passes accumulate over the kernel taps in index order, in a single
  float, matching `torch.sum(mul(x, k), -1)` in the reference.

* `src/kernels.rs`'s AVX2 bodies were rewritten for speed and every one of them
  was checked BYTE-IDENTICAL against the PNG the previous version produced, which
  is a stronger check than the gate: `conv3x3_avx2` covers each row with full
  32-pixel tiles whose accumulator set is written to the same pixels more than once
  (re-covering a partial tile at an overhang-anchored start is the same inputs in
  the same `(ky, kx, ci)` order, so the same bits), the transposed convolution's
  vector body keeps the scalar body's `kx`-ascending, `ci`-innermost order per
  output element, and running two output channels in one chunk of work changes
  only WHICH thread computes a pixel, never the order within it. So the CPU column
  of the gate is still exactly where it always was - 1.967e-06 / 8.345e-07 /
  1.550e-06 at 64/256/720 - which is the point of fixing the order in writing:
  a speed change that moves those numbers is a defect, not a trade.
* `conv3x3s2` is the one CPU kernel whose nesting is `ic`-OUTERMOST, because it
  accumulates `plane[p] += v_ic` once per input channel (that is what makes it
  cheap in memory and it is why a 4-way unroll of its interior bought only 71.7 ->
  79.0 GFLOP/s: the load-add-store chain per channel is the dependency, not the
  arithmetic).

## The two SAC passes do not collapse, and the indices do

`IAC` applies its separable adaptive convolution as two sequential 1-D passes
with an intermediate plane, both with `padding=mode="replicate"`. It is tempting
to fold them into one clamped gather, because the sample indices compose:
the horizontal pass reads the vertical pass's output at column
`clamp(x + k2 - pad)`, and the vertical pass read `clamp(y + k1 - pad)`.

What does NOT compose is the TAP. The tap the vertical pass applied to that
sample is `kv[y][clamp(x + k2 - pad)][k]`, not `kv[y][x][k]`. Measured on a
5x7x2 case in float64, the exact composed form matches this two-pass code to
3.6e-15 while a form with every tap at `(y, x)` is off by 34.5 absolute. SAC is
therefore two passes with an intermediate plane in `tools/reference.py`, in
`src/kernels.rs` and in `cuda/ifan.cu` alike.

## Two upstream divergences are reproduced on purpose

`tools/reference.py` documents both at length; they are here because a port that
"fixes" either one will not match the released weights:

1. **`IAC` multiplies by `kernel1` on BOTH passes.** Upstream
   `models/IAC.py:37` passes `kernel1` to the horizontal pass where its own
   comment says `kernel2`, and the upstream README calls this out as a bug. Every
   released checkpoint was trained through that path.
   `IFAN_REF_FIX_KERNEL2=1` switches the reference to the intended form, which
   exists only to measure what the bug costs.
2. **The last IAC iteration is activated.** `is_act_last` defaults to True
   upstream, so there are N activations and not N-1, and the weights were trained
   that way.

## How to check a change

* **The gate.** `python3 tools/gate.py --size 256` runs the engine both ways and
  the reference, and prints the three differences. It needs the torch
  environment for the reference half; `--skip-torch` compares only the two Rust
  backends, and `--skip-cuda` only CPU-vs-torch.
* **`--dump` writes the final planes** as flat little-endian f32 with a 12-byte
  `(c, h, w)` header, which is what the gate compares - the numbers the network
  produced rather than the numbers after 8-bit quantisation, because 1/255 is
  exactly the scale a real bug shows up at.
* **`--dump-op <name>` stops at a single activation** and writes THAT, on either
  backend, which is how a divergence is localised to one op instead of guessed
  from the last one. `net::Cpu::run_until` and `gpu::Gpu::run_until` share the
  activation names, so a name that works on one path works on the other.

  ```
  ifan -m IFAN.safetensors -i in.png -o out.png --cpu   --dump-op filt --dump /tmp/cpu_filt.f32
  ifan -m IFAN.safetensors -i in.png -o out.png         --dump-op filt --dump /tmp/gpu_filt.f32
  ```

  The names, in graph order: `in`, `e1a`, `e1b`, `f1`, `e2a`, `e2b`, `f2`,
  `e3a`, `e3b`, `f3`, `e4a`, `e4b`, `f_c`, `k1a`..`k1c`, `k2a`..`k2c`,
  `k3a`..`k3c`, `k4a`, `k4b`, `k`, `dm0`, `dm1`, `dm2`, `dm`, `f_dm`, `cat`,
  `c44a`..`c44c`, `fc4`, `fa`, `fb`, `fc`, `filt`, then per IAC iteration `i`
  the three `iac.v.i`, `iac.h.i`, `iac.o.i` (the last is `iac.o.16` for the
  17-iteration checkpoint), then `cr0`..`cr2`, `u3a`..`u3d`, `u2a`..`u2d`,
  `u1a`..`u1d`, `res`, `raw`, `out`.

  Plus, INSIDE every residual BLOCK, three names per block: a block is a named
  layer in its own right (`res()` is called once per iteration), so its two 3x3
  convolutions and their sum appear as `L.a`, `L.b` and `L.m`, where `L` is one of
  the 21 layer names `conv_res.1.0`, `conv_res.1.1`, `conv_res.1.2`, `F.1.0`,
  `F.1.1`, `F.2.0`, `F.2.1`, `DME.1.0`, `DME.1.1`, `DME.2.0`, `DME.2.1`,
  `conv4_4.{1,2}.{0,1}`, `upconv{1,2,3}_{1,2}.0`. That is 63 of the graph's 169
  names, and they matter when a divergence is INSIDE a block rather than between
  blocks: `m` is where the leaky ReLU happens (the only name an op both reads and
  writes), and a bisection that only had the layer outputs would stop at the
  block's edge and say nothing about which of its two convolutions moved.

  A ladder of these at 256x256 is the record this engine was debugged against.
  The worst stage, CPU twin vs CUDA, was 7.5e-06 on `dm` (a quantity reaching
  2.30) and 9.4e-06 on `cr2`; `filt` - the tensor that makes this network
  unusual - was 3.2e-06 over 15232x32x32 values, and the output was 6.3e-07.
* **`IFAN_TRACE_ALLOC=1`** traces every fresh device buffer with its size and
  the free VRAM after it, and prints the dry pass's plan. That is how the GPU
  allocator was checked; see `gpu::plan`.

A comment-only edit changes the binary's bytes (the build-id and metadata
sections are hashed from the source), so **binary byte-identity is not a valid
regression check** across source edits. The output PNG is, and so is the plan
line the engine prints on every GPU run.
