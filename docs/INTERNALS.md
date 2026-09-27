# Internals

How this engine is put together, what it measures, and what was tried and
rejected. None of it is needed to run `ifan`; it is here because the reasoning
behind a kernel's shape is worth keeping somewhere, and the README is for people
who want a deblurred PNG.

## The shape of the code

| file | what it holds |
|---|---|
| `src/net.rs` | the op list - `Op` and `graph(arch)` - plus the CPU interpreter (`Cpu`) |
| `src/gpu.rs` | the CUDA driver layer: weight arena, buffer pool, the plan, the launch gate |
| `src/kernels.rs` | the CPU kernels, AVX2 where the host has it |
| `cuda/ifan.cu` | the CUDA kernels |
| `src/weights.rs` | the safetensors mmap, the tensor names, the architecture constants |
| `src/memguard.rs` | the CPU memory plan and the refusal guard |
| `src/image.rs` | PNG in, PNG out, CRC-validated |
| `src/cuda.rs`, `build.rs` | the fatbins and the kernel-name lists |

The graph is **data**: one `Vec<Op>` produced by `graph(arch)` and executed by
two interpreters. Written as two hand-transcribed programs the CPU twin and the
CUDA path would drift, and the only guard would be a numeric comparison that says
"something differs" without saying where. As data, a graph mistake is a mistake
in both backends at once, so the comparison against `tools/reference.py` catches
it with the op index printed, and only an interpreter mistake is
backend-specific. There are 191 ops and 171 named buffers.

## Op set

`Conv3x3`, `Conv1x1`, `Conv3x3S2`, `ConvT4x4S2`, `LeakyRelu`, `Sac1D`, `SacBias`,
`Add`, `Cat2`, `Copy`, `Clip`. Each maps onto one kernel or onto one loop in the
interpreter. A more expressive op - a fused one, a conditional - would give the
two backends a way to disagree about what it means, which is the thing the design
exists to prevent.

The filter tensor's channel-block layout lives in exactly one place,
`net::tap_range`: kernel1 of iteration `i` at `i * ck * 2`, kernel2 at
`i * ck * 2 + ck`, the bias planes at `2 * N * ck + i * ch4`, with
`ck = ch4 * fs = 384`. Two copies of that rule is how the two backends would come
to disagree, so there is one.

## CUDA kernels

Four tiled kernels, instantiated from two templates:

* `if_conv_tile_kernel<KH, KW, STRIDE, TBX, TBY, TPX, CI_TILE, OC_TILE>` - stages
  an input tile in shared memory per channel tile (the halo ZERO-FILLED rather
  than skipped), gives each thread `OC_TILE` accumulators, interleaves pixels by
  `TBX`, stages the weights, and fuses bias and activation into the epilogue.
  Instantiations: `G3_TILE` 128x8/8 (32x8 tile, CI 4, OC 8), `G1_TILE` 128x4/32,
  `G2_TILE` 64x8/16 (32x8, CI 1, OC 16).
* `if_convt_tile_kernel` - the transposed 4x4 stride 2. Four taps per output,
  selected by the output's PARITY: `ry = ceil(oy/2)`, `kya = 1 - (oy & 1)`, the
  second contributing row `ry - 1` with `kya + 2`; shared tile `SW = TW + 2`,
  `SH = TH + 2`, origin `oy0/2 - 1`. `TBX` and `TBY` MUST BE EVEN or the parity
  split breaks. Instantiation `GT_TILE` 64x8/8 (32x8, CI 2, OC 8).

Plus `if_sac_vertical`, `if_sac_horizontal`, `if_sac_bias` and `if_clip`, and the
toolkit's `lg_lrelu`, `lg_add`, `lg_copy`. `build.rs` compiles the toolkit's
`cuda/kernels.cu` and this project's `cuda/ifan.cu` into two fatbins, and both
`build.rs`'s kernel lists and `src/cuda.rs`'s mirror of them are validated BOTH
ways against the files: a kernel defined but not listed fails the build, and so
does a name listed but not defined.

Why the toolkit's kernels could not be reused: `lg_conv4x4s4` is a stride-4 patch
embed with no padding and `lg_upsample2x_nearest` has no learned kernel, so
neither the stride-2 convolution nor the transposed convolution exists there, and
there is no dense tiled convolution and no clip at all.

### Measured kernel rates

The benchmark that produced these was `examples/kbench.rs`, removed with the rest
of the development tooling; it is in the pre-cleanup copy and its method was a
timed loop of launches on a fixed input, one geometry per row. Numbers are
GFLOP/s at 1080p on a GTX 1080, tiled against one-output-element-per-thread:

| op | shape | tiled | untiled |
|---|---|---|---|
| 3x3 | 3 -> 32 | 679 | 88 |
| 3x3 | 32 -> 32 | 1659 | 226 |
| 3x3 | 128 -> 128 at 1/8 | 2040 | 106 |
| 3x3 s2 | 32 -> 64 | 1294 | 58 |
| 4x4 t s2 | 128 -> 128 | 4899 | 749 |

Whole-pass efficiency is 1710 GFLOP/s of the card's ~8.9 TFLOP/s, 19%. The wall
times those rates produce are in the README's Performance section; the per-op
split of a real pass is what `IFAN_TIME=1` prints (see Instruments below).

### The geometry sweeps

Every launch geometry in `cuda/ifan.cu` came from a sweep of a few dozen
candidates for its own kernel, run through the (now deleted) benchmark. At 1080p
on the 1080, milliseconds of a single launch, `1x1 128->15232 1/8` (`F.3`) beside
`1x1 128->128 1/8`:

| TBX | TBY | TPX | CI | OC | F.3 ms | 128->128 | registers |
|---|---|---|---|---|---|---|---|
| 32 | 4 | 4 | 4 | 32 | 591 | 567 | 255 + 68 B spill |
| 32 | 4 | 4 | 4 | 16 | 383 | 375 | no spill |
| 32 | 8 | 4 | 4 | 8 | 383 | 375 | no spill |
| 32 | 4 | 8 | 4 | 16 | 383 | 375 | no spill |
| 32 | 2 | 4 | 4 | 32 | 574 | 549 | 255 + 68 B spill |
| 32 | 4 | 4 | 4 | 48 | 223 | 193 | 255 + 4648 B spill |

The winner spills and still wins: at these sizes the FMA chains are the
bottleneck, and a spilled accumulator set with sixteen independent chains beats a
clean set with four. 48 channels with FOUR pixels per thread loses because 192
accumulators go to local memory wholesale - but that is about `(TPX, OC)` together,
not about 48 channels, which a second sweep at 1080p says:

| TBX | TBY | TPX | CI | OC | F.3 ms | GFLOP/s |
|---|---|---|---|---|---|---|
| 32 | 4 | 4 | 4 | 32 | 198.4 | 637 |
| 32 | 4 | 4 | 4 | 16 | 230.2 | 549 |
| 32 | 4 | 4 | 4 | 8 | 272.3 | 464 |
| 32 | 4 | 8 | 4 | 16 | 309.9 | 408 |
| 32 | 2 | 4 | 4 | 32 | 207.2 | 610 |
| 32 | 4 | 2 | 4 | 32 | 219.9 | 574 |
| 32 | 4 | 1 | 1 | 32 | 136.5 | 925 |
| 32 | 4 | 2 | 1 | 32 | 126.8 | 996 |
| 32 | 4 | 2 | 1 | 48 | 108.5 | 1164 (best) |
| 32 | 8 | 2 | 1 | 48 | 110.5 | 1143 |
| 32 | 4 | 2 | 1 | 96 | 126.2 | 1001 |
| 32 | 4 | 2 | 1 | 64 | 149.0 | 847 |
| 32 | 4 | 2 | 1 | 128 | 170.2 | 742 |
| 32 | 2 | 2 | 1 | 32 | 128.3 | 984 |
| 64 | 4 | 2 | 4 | 16 | 208.4 | 606 |

FEWER pixels per thread and a WIDER channel tile is what this kernel wants, the
opposite of the 3x3's answer, and the reason is the same one that shows up in its
profile: its cost is the INPUT RE-READ - 15232 output channels over an `OC`-channel
tile is `c_out / OC` passes over the same input - so an extra pixel per thread buys
a shorter accumulator chain per shared load, while an extra CHANNEL buys one fewer
pass over the input.

And the stride-2 kernel, swept the same way (`s2 32->64 1080p`, 19.1 GFLOP per
launch, so 15.1 ms is 1265 GFLOP/s):

| TBX | TBY | TPX | CI | OC | ms | shared |
|---|---|---|---|---|---|---|
| 32 | 4 | 4 | 4 | 8 | 45.7 | 36 KiB |
| 16 | 16 | 2 | 2 | 16 | 26.0 | 34 KiB |
| 32 | 8 | 4 | 2 | 16 | 25.6 | 34 KiB |
| 32 | 8 | 2 | 2 | 32 | 18.8 | 19 KiB |
| 32 | 8 | 2 | 4 | 16 | 16.4 | 36 KiB |
| 32 | 16 | 2 | 2 | 16 | 15.4 | 34 KiB |
| 32 | 8 | 2 | 1 | 16 | 15.1 | 9 KiB (chosen) |

The lesson here is the opposite of the 1x1's: this kernel wants a SMALL shared
tile. Its rows with 32 accumulators are all within a factor of 1.6 of each other;
the rows with 64 accumulators, for the same total work, are not. What separates
36 KiB from 9 KiB is how many blocks an SM can hold, and the staged region here is
mostly halo - a 129x17 region for a 64x8 output tile - so a wider channel tile
pays for staging and buys nothing else.

### F.3's launch geometry is an open question

`F.3` (`fc -> filt`, 1x1, 128 -> 15232) is one launch and 18% of the pass. Its
geometry was swept twice. The second sweep found that this kernel wants FEWER
pixels per thread and a WIDER channel tile: `TPX 2, OC 48` measures 108.5 ms
standalone against the shipped geometry's 198.4 ms, 1.8x faster. The earlier
sweep's opposite conclusion was about 48 channels *with four pixels per thread*,
where 192 accumulators spill the register file.

**That geometry is not shipped, because inside this engine it makes no
difference.** F.3's launch measures 199.8 ms at 1080p and ~59 ms at 720p in every
configuration of the sweep, including `OC 8`, which needs six times as many input
passes and is 2.5x slower standalone, and the whole pass is 1.40 s with either
geometry. Two explanations were tested and refuted: the 477-block channel tail on
a 20-SM device (the figure does not move when the block count changes sixfold),
and GPU idle draining into the measured interval (synchronising before every timed
launch changed nothing). The engine's rate scales with FLOPs at ~600 GFLOP/s at
both sizes where the standalone benchmark's scales with memory traffic, so this
launch is bound by something a geometry change does not reach - the card's cache
or clock state across a stop-start pass are candidates, neither demonstrated.

### The alias a fused epilogue removed

The transposed convolution used to run its untiled kernel into a pool scratch and
then activate into `out`. The pool's best-fit returned the SAME buffer as `out`,
so the decoder was `act(act(conv))` everywhere - silent, plausible, and not caught
by the gate because it is a legal-looking accumulation-order-sized difference.
Fusing the activation into the tiled kernel's epilogue deletes the scratch
entirely, and CUDA-vs-torch at 64x64 improved slightly (1.729e-06 -> 1.609e-06) as
a side effect. The lesson is in `src/net.rs`'s `out_buf`: a recycled buffer is
RESHAPED and ZEROED, because `conv1x1_into` and `conv3x3s2_into` accumulate into
their destination.

## Memory

**GPU.** `plan` MEASURES the pass rather than modelling it: it walks the whole
graph with `dry = true` through the single `go()` launch gate, and `Buf`'s device
handle is an `Option`, so a dry pass allocates nothing. `describe` and
`describe_actual` agree at every size (64x64 6.2 MiB, 256x256 99.8 MiB, 720x720
789 MiB, 1080p 3.08 GiB, 4096x4096 24.94 GiB refused). There is no fallback
ladder and no smaller layout to fall back to.

**CPU.** `memguard::CpuPlan::of` SIMULATES the pooled allocator - the same op
list, the same liveness rule, the same best-fit recycle, the peak of
`live + pool` - because a formula beside the interpreter has to be kept in step
with it by hand. The modelled peak is scaled by a measured ratio (1.10) and a
48 MiB base, and the guard refuses with both numbers when the result exceeds
`MemAvailable`.

## CPU kernels

`conv3x3_into`, `conv1x1_into` and `convtranspose4x4s2_into` are dispatchers to
`#[cfg(target_arch = "x86_64")] #[target_feature(enable = "avx2,fma")]` bodies
behind an `x86_avx2()` check, with scalar bodies for other targets. No
`-C target-cpu=native`, no AVX-512: the same binary runs on any x86-64, which is
the point of shipping a CPU path at all.

* `conv3x3_avx2` splits the output into chunks of `2 * hw`, i.e. TWO OUTPUT
  CHANNELS per chunk, so that both channels' accumulators are fed by one set of
  input loads. The single-channel tile issues four input loads and one broadcast
  per four FMAs and measured exactly one load per FMA - the load port's limit - so
  this is a load-port fix, not a scheduling one. `conv3x3_row_tile2_avx2` is the
  paired tile; the odd last chunk (the graph's 3-channel final convolution) takes
  `conv3x3_plane_avx2`.
* Row coverage in both plane paths: full 32-pixel tiles from `x0 = 1` to
  `wd - TX - 1`, then ONE more anchored at `wd - TX - 1`, which re-covers a
  partial tile with the identical value; `conv3x3_pixel_scalar` does `x = 0` and
  `x = wd - 1` (their taps read off the row); rows narrower than `TX + 2` and the
  top and bottom rows go scalar in full, which is `2/h` of the image.
* `convtranspose4x4s2_avx2` vectorises over OUTPUT COLUMNS in the two parity
  classes its taps create (even `ox`: `kx = 1` at `ix = t`, `kx = 3` at `ix = t-1`;
  odd `ox`: `kx = 0` at `ix = t+1`, `kx = 2` at `ix = t`), all contiguous reads.
  The store interleaves with `unpacklo/hi` and two `permute2f128`, and per element
  the order stays `kx`-ascending with `ci` innermost, so it is bit-identical to
  `convt_pixel_scalar`. It needs `1 <= t` and `t + 8 < wd`.
* `conv3x3s2_into` is portable ILP only and the slowest kernel (~83 GFLOP/s):
  its `plane[p] += v_ic` accumulation fixes `(ic, ky, kx)` with `ic` OUTERMOST, so
  each input channel is a load-add-store chain no unrolling can hide. A 4-way
  unroll of the interior, with the zero row hoisted out of the `(ic, oy)` loops,
  bought 71.7 -> 79.0 GFLOP/s. Recorded as a negative result.
* The pointwise ops had no parallelism of their own; `add_into_par`,
  `lrelu_in_place_par`, `lrelu_into_par`, `clip_into_par`, `sac_pass_into_par` and
  `sac_bias_into_par` split them across the pool (per element, or per channel for
  the SAC pair). Worth 0.25 s of a 1.50 s pass at 720x720.

Trap worth remembering: `out_buf` hands back an OWNED buffer, so `self.get(..)`
may be borrowed immediately after it. Cloning a source to satisfy the borrow
checker instead made `SacBias` copy the whole 493 MB `filt` tensor once per SAC
iteration, which is 2.1 s of a 3.6 s pass - and `IFAN_CPU_TIME=1` caught it.

## Instruments

| variable | where | what it prints |
|---|---|---|
| `IFAN_TIME=1` | `src/gpu.rs` | every launch and D2D copy, timed with a reused event pair inside `go()`: launches, milliseconds, GFLOP/s |
| `IFAN_CPU_TIME=1` | `src/net.rs` | the same for the CPU path, by KIND, by SHAPE (`c_in -> c_out @ h`) and the 12 slowest NAMED ops |

Both are off by default and arithmetically neutral: `IFAN_TIME` synchronises after
every launch, so it costs wall time, but a timed and an untimed run write
byte-identical PNGs. `IFAN_CPU_TIME` measures wall time per op inside the
interpreter, which is why its per-op figures add up to the pass wall - the
parallelism is inside an op.

Two figures that look like deltas and are not: `--dump-op <name>` returns after
copying a named activation to the host, so every run carries the ~0.11 s of fixed
startup and, for `filt` at 1080p, a 1.84 GiB device-to-host copy. Use the
instruments above instead.

## Development build

`--dump` and `--dump-op` exist only with `--features dev`
(`cargo build --release --features dev --target-dir ../target/dev`). A release
build does not list them and refuses them by name. That split exists because
`tools/gate.py` reads raw f32 planes to compare pre-clip values, while the gate's
pass/fail decision reads the PNGs written by the SHIPPED binary, and the gate
asserts the two builds' PNGs are identical - a gate that needs a different build
from the one users get is a gate that can pass while the shipped binary fails.

## Verification: the four legs, and the bound they measure

    cargo test --release                     # 3 tests, GPU + CPU + shapes
    CARGO_TARGET_DIR=/tmp/cputarget cargo test --release --no-default-features
    cargo build --release --features dev --target-dir ../target/dev
    python3 tools/gate.py --size 64          # then 256 and 720

The tests skip with a printed reason rather than failing when the converted
checkpoint is absent, so a machine without it gets a green run and a note. The
gate is the one that needs a torch environment
(`/home/jacob/torchenv311/bin/python`); it builds nothing itself, so the two
binaries must already exist.

Measured on this machine (i7-13700K, GTX 1080, rustc 1.92, nvcc 12.4), maximum
absolute difference from `tools/reference.py` on the 0..1 scale:

| size | CPU vs torch | CUDA vs torch | CPU vs CUDA |
|---|---|---|---|
| 64x64 | 1.520e-06 | 1.371e-06 | 2.801e-06 |
| 256x256 | 7.451e-07 | 9.537e-07 | 1.192e-06 |
| 720x720 | 1.699e-06 | 2.980e-06 | 3.040e-06 |

**Read that as a bound of about 2e-06 for the CPU path and 3e-06 for the CUDA
path, not as a digit string to reproduce.** Three reasons it cannot be pinned
tighter. torch's own output depends on its THREAD COUNT - at 64x64 the same
comparison is 1.5199e-06 with `OMP_NUM_THREADS=1` and 1.6987e-06 with the default
16, and torch-vs-torch across thread counts differs by 2.4e-07 - which is why the
gate pins `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for the reference run. The
clipped and pre-clip rows are identical here because no value near the clip
boundary differs by enough to be clamped differently; the gate checks both
anyway, against a `--tol-raw` of 1e-5, because a defect that happens to sit under
the clip would otherwise be invisible. And the two backends sum in different
orders by design (see above), so their difference from each other is larger than
either one's difference from torch.

For scale: the graph defects found while building this engine all showed up
between 1e-02 and 1.0, four orders of magnitude above this floor.
