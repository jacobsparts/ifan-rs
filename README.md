# ifan-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [nafnet-rs](https://github.com/jacobsparts/nafnet-rs),
[rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs),
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs),
[maxim-rs](https://github.com/jacobsparts/maxim-rs),
[scunet-rs](https://github.com/jacobsparts/scunet-rs),
[adaptive-enhance](https://github.com/jacobsparts/adaptive-enhance),
[nightenh-rs](https://github.com/jacobsparts/nightenh-rs) and
[swin2sr-rs](https://github.com/jacobsparts/swin2sr-rs), all
built on the [lightgpu toolkit](https://github.com/jacobsparts/lightgpu);
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives them all.

[IFAN](https://github.com/codeslake/IFAN) defocus deblurring as a single
self-contained binary. Feed it a PNG of an out-of-focus photo (a phone shot with
a too-close or off-centre subject), get back a PNG with the defocus blur removed.
No Python, PyTorch, ONNX Runtime, or CUDA toolkit needed at runtime.

```
ifan -m IFAN.safetensors -i blurry.png -o sharp.png
```

![A defocused photo beside the same photo after this engine, from the upstream
demo image](assets/before-after.png)

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and the
  CPU path when it is not, so one binary covers a machine with no NVIDIA driver
  at all; `--cpu` selects the CPU path explicitly.
* 2.48 MiB binary, statically linked except `libc` and `libgcc_s`.
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk. (The CPU-only
  build is 1.18 MiB; most of the GPU build's extra weight is the fatbins, which
  carry sm_61, sm_75 and sm_80 plus PTX.)
* The architecture is **iterative**, and that is the interesting part: the
  network predicts a *filter tensor* - 15232 floats per 1/8-scale pixel, a
  separable 3x3 pair and a bias for each of 17 iterations - and then applies it
  with an adaptive convolution whose weights change per pixel. The filter tensor
  for a 1920x1080 input is 1.84 GiB on its own, which is why memory gets its own
  section below rather than a footnote.
* No tile size to choose and none to get wrong: the pass is measured before it
  starts, and a pass that will not fit is refused with both numbers rather than
  launched and killed.
* One checkpoint, converted from the authors' `IFAN.pytorch`: 10.477 M
  parameters, N=17 IAC iterations. See Choosing a checkpoint below.
* Both backends agree with the upstream PyTorch implementation to float32
  round-off; see [docs/NUMERICS.md](docs/NUMERICS.md) for why they are not
  bit-identical and how to check a change.

## Download

Prebuilt binaries and the converted checkpoint are attached to the
[releases](https://github.com/jacobsparts/ifan-rs/releases). Both binaries run on
the CPU; they differ only in whether CUDA support is compiled in.

| asset | contents | notes |
|---|---|---|
| `ifan-linux-x86_64` | CPU + CUDA | x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs a compute capability 6.1+ GPU, `--cpu` runs the pure-Rust path anywhere |
| `ifan-linux-x86_64-cpu-only` | CPU only | same, with nothing NVIDIA-related included - the CPU backend is its only one, and it says so if you run it without `--cpu` instead of falling back |
| `IFAN.safetensors` | the converted checkpoint | see Choosing a checkpoint |

```sh
chmod +x ifan-linux-x86_64
./ifan-linux-x86_64 -m IFAN.safetensors -i blurry.png -o sharp.png
```

The `chmod` is not decoration: a download does not carry the executable
bit through, and a binary that has lost it fails with `Permission denied`
before it can print anything.

## Build

```sh
cargo build --release
# CPU only, no CUDA toolkit or driver needed at build time either:
cargo build --release --no-default-features
```

The default build needs `nvcc` (set `NVCC=` if it is not on `PATH`) and produces
one binary with both backends. The `--no-default-features` build contains only
the CPU path: the GPU is that build's default backend, so run without `--cpu` it
reports `this build has no CUDA backend (built without the `cuda` feature); pass
--cpu or rebuild with the default features` rather than failing obscurely, and
`--cpu` runs it. The kernels cover `sm_61`, `sm_75`, `sm_80`
and compute capability 8.0 PTX, so the GPU path runs on Pascal (GTX 10-series)
through Ampere, and on anything newer via the PTX.

`lightgpu` is a normal Cargo dependency on
[its repository](https://github.com/jacobsparts/lightgpu), so a clone of this
project builds on its own.

### Tests

```sh
cargo test --release
```

Three checks, all skipped with a printed reason rather than failing when the
converted checkpoint is absent: the CPU twin against a stored reference produced
by `tools/reference.py` on the original PyTorch checkpoint (raw f32 planes, not
PNG pixels - 8-bit quantisation hides exactly the differences a port gets wrong),
the CPU and CUDA paths against each other on the same fixture, and the geometry
walk against the shapes the checkpoint itself declares. The full three-way
comparison, including the torch reference, is `python3 tools/gate.py`, which
needs a torch environment and is therefore not part of `cargo test`.

## Choosing a checkpoint

There is one model, and `-m` is how you choose it - there is no flag for the
task, because IFAN is trained for one.

| checkpoint | what it is | params | iterations |
|---|---|---|---|
| `IFAN` | defocus deblurring, the paper's final model | 10.477 M | 17 |

The upstream authors also publish `IFAN_8`, `IFAN_26`, `IFAN_35` and `IFAN_44`
(N=8, 26, 35, 44 - accuracy against speed), plus a 16-bit variant and a
dual-pixel variant. The engine reads the iteration count **out of the converted
file's metadata** rather than from a flag, and F.3's output width has to match
it, so any of them converts and runs:

```sh
python3 tools/convert.py IFAN_26.pytorch IFAN_26.safetensors
```

That is also why there is no `--iterations`: the number is a property of the
weights. `tools/convert.py` is where the metadata is written, and it refuses a
checkpoint whose tensors do not line up with the architecture it declares.

## Usage

```sh
ifan -m IFAN.safetensors -i blurry.png -o sharp.png
ifan -m IFAN.safetensors -i blurry.png -o sharp.png --cpu
cat blurry.png | ifan -m IFAN.safetensors > sharp.png
```

```
ifan - IFAN image restoration (single image, 8-bit)

    ifan -m <weights.safetensors> [-i <in.png>] [-o <out.png>] [--cpu]

    -m, --model <path>  the converted weights (models/IFAN.safetensors)
    -i, --input <path>  input PNG (default: stdin)
    -o, --output <path> output PNG (default: stdout)
        --cpu           force the CPU path instead of CUDA
    -h, --help
```

Input is 8-bit RGB or greyscale PNG and output is 8-bit RGB. Leaving `-i` out
reads the PNG from stdin and leaving `-o` out writes it to stdout - both are
whole files, read and written as bytes, so a pipe works and nothing else goes to
stdout (the plan, crop and refusal lines go to stderr). There is no `--device`
flag: the GPU is used when there is a driver and the CPU path when there is not,
and `--cpu` is for when you want the CPU path on a machine that has both.

There is no `--tile`. This engine runs a whole image at once, and the plan below
decides whether that fits; a memory-control flag that was silently ignored would
let a caller believe it had bounded this process's memory.

The `IFAN_TIME=1` and `IFAN_CPU_TIME=1` environment variables print where a pass
spent its time; see [docs/INTERNALS.md](docs/INTERNALS.md).

## Large images

The whole image is resident on the device at once, and IFAN's filter tensor is
`15232 x (h/8) x (w/8)` floats, so VRAM grows with the input faster than the
picture does. The pass is **measured** before it starts, by walking the whole
graph with the kernel launches suppressed, so the requirement is a number rather
than a matter of luck:

| input | device buffer the pass allocates |
| --- | --- |
| 64x64 | 6.2 MiB |
| 256x256 | 99.8 MiB |
| 720x720 | 789 MiB |
| 1280x853 (cropped to 1280x848) | 1.61 GiB |
| 1920x1080 | 3.08 GiB |
| 2560x1440 | 5.48 GiB |
| 2048x2048 | 6.23 GiB |
| 4096x4096 | 24.94 GiB - refused, see below |

Those are what the **pass allocates**, which is a property of the engine; free
VRAM is a property of the moment and moves by gigabytes on a machine whose
desktop compositor or other tenants are holding device memory. The engine prints
both numbers on every GPU run, so which one you are looking at is never in doubt:

```
ifan: whole-image: cropped 1920x1080, f32 filter tensor over the whole 1/8 grid, 3.08 GiB of device buffer measured
ifan: actual 3.08 GiB of device buffer allocated
```

Two things keep that number from being larger, and both are about the filter
tensor: it is predicted at **1/8 resolution** rather than full resolution, and it
is written once and then read by all 17 iterations instead of one iteration's
slice being materialised per step. Everything else is an ordinary activation, and
activations are recycled through a pool as their last reader runs, so the 171
named buffers the graph contains are a few dozen live at a time.

On an 8 GB card every size a photo editor produces fits up to 2560x1440.
2048x2048 at 6.23 GiB is close to the edge: it runs when the card is quiet and is
refused when it is not.

### A pass that will not fit is refused, with the numbers

There is no fallback. The measured requirement is compared against free VRAM
before a byte is allocated, and a pass that does not fit says what it needed and
what there was:

```
$ ifan -m IFAN.safetensors -i 4096.png -o out.png
ifan: 4096x4096 does not fit in VRAM: the pass allocates 24.94 GiB of device buffer and 7.78 GiB is free (7.59 GiB usable after a 192.0 MiB reserve). A smaller image is the only way through - this engine has no smaller layout to fall back on.
$ echo $?
1
```

The reserve is headroom for the driver's own allocations and for whatever else is
on the card between the check and the launches.

### The CPU path, where memory is the other way round

`--cpu` needs no GPU, and its footprint is bounded by HOST memory rather than
VRAM. It recycles its activations the way the GPU path does, and its plan is
arithmetic rather than a measurement - the same op list, the same liveness rule,
the same best-fit recycle, and the peak of `live + pool` over the pass - so the
line says which it is:

```
ifan: cpu: 171 graph buffers, 1.05 GiB modelled peak, 1.20 GiB with allocator slack
```

The second number is the first scaled by a measured allocator factor (1.10x,
fitted from this engine rather than assumed) plus a 48 MiB base for the process
and the mapped checkpoint. It is a REFUSAL GUARD rather than a memory report: a
pass that cannot fit says so in a sentence instead of being killed by the kernel,
comparing the modelled peak against `MemAvailable`, so that a machine which has
just mapped a 42 MiB checkpoint - most of it in reclaimable page cache - is not
refused a pass it could run:

```
$ ifan -m IFAN.safetensors -i 4096.png -o out.png --cpu
ifan: 4096x4096 does not fit in host memory: the pass peaks at 33.94 GiB of graph buffers, which is 37.38 GiB with the measured 1.10x allocator factor and a 48.0 MiB base; MemAvailable is 18.62 GiB
$ echo $?
1
```

## Differences from upstream

Two upstream behaviours look like bugs and are reproduced on purpose, because
every released checkpoint was trained through them and a "fixed" port would not
match the weights:

* `IAC` multiplies by `kernel1` on **both** of its two passes, where upstream's
  own comment says `kernel2` was meant (`models/IAC.py:37`, and the upstream
  README calls it out). `IFAN_REF_FIX_KERNEL2=1` in `tools/reference.py` switches
  to the intended form, which exists only to measure what the bug costs.
* The **last** IAC iteration is activated (`is_act_last` defaults to True
  upstream), so there are N activations and not N-1.

And one that is not reproduced:

* `predict.py` downscales an image whose longest side exceeds 1920 with
  `cv2.INTER_AREA`, which is a memory accommodation from 2021 rather than part of
  the model's contract. This engine runs at full size. On an image above 1920 the
  output will therefore differ from upstream's `predict.py` - it is a different
  (more faithful) computation, not a bug.

The crop to a multiple of 8 is reproduced, because it IS the contract:
`refine_image` crops rather than pads, and both `predict.py` and `eval.py` go
through it, so it is what every published number was produced under.

## Performance

IFAN, on an i7-class host with a GTX 1080 (Pascal, sm_61):

| input | this engine, GPU | this engine, CPU | PyTorch 2.5 (CUDA) | PyTorch 2.5 (CPU) |
|---|---|---|---|---|
| 720x720 | 0.45 s, 184 MB peak RSS | 1.27 s, 19.2 s user, 1.16 GB peak RSS | **0.13 s**, 625 MB RSS / 1035 MB VRAM | 2.26-3.21 s (fwd), 1.5 GB peak RSS |
| 1920x1080 | 1.37 s, 243 MB peak RSS | 6.43 s, 87.2 s user, 4.4 GB peak RSS | **0.53 s**, 625 MB RSS / 4009 MB VRAM | 3.49-3.59 s (fwd), 4.9 GB peak RSS |

**PyTorch on the GPU is 2.6x faster than this engine's CUDA path at 1080p and
3.5x at 720x720. Against PyTorch on the CPU the order is split by size: this
engine is FASTER at 720x720 (1.27 s against 2.26-3.21 s) and about 1.8x behind at
1080p (6.43 s against 3.49-3.59 s).** The honest reasons:

1. PyTorch's CUDA convolutions run through cuDNN, which mines every layer for a
   fast algorithm - GEMM with implicit im2col, Winograd, FFT - and picks per
   shape. The CUDA path here has one tiled direct-convolution kernel per
   convolution KIND, with a launch geometry measured by hand for IFAN's channel
   counts; its individual kernels measure 640-4900 GFLOP/s and the whole pass
   averages 1710 GFLOP/s, 19% of the GTX 1080's ~8.9 TFLOP/s, where cuDNN on a
   convolutional network of this shape reaches ~30-40%.
2. PyTorch's CPU backend is linked against Intel MKL/oneDNN, which is a
   hand-tuned SGEMM microkernel per ISA across 16 threads. The CPU twin here uses
   rayon over hand-written AVX2/FMA intrinsics where the host has them and scalar
   loops everywhere else - no `-C target-cpu=native`, no AVX-512, so the same
   binary runs on any x86-64. That constraint is deliberate, and it is the story
   of the CPU column: those kernels reach 550-840 GFLOP/s against the CUDA path's
   1710 on the same machine, so the remaining 1.8x at 1080p - where the
   convolution dominates - is what a hand-tuned GEMM microkernel buys that
   hand-written intrinsics do not.

What the engine buys instead is the deployment contract: a single 2.48 MiB static
binary (1.18 MiB CPU-only) with no Python, no PyTorch, no CUDA toolkit runtime,
243 MB of resident memory on the GPU path, and a pure-Rust CPU fallback anywhere.

Both figures are worth reading carefully. On the GPU, a 64x64 input costs 0.11 s,
which is the fixed startup - driver init, both fatbins, the 42 MiB weight upload -
and is the number to subtract before reading either row. On the CPU, 1.27 s of
wall against 19.2 s of user time at 720x720 is 15.1 cores busy: the pass saturates
the machine, so the user time is the figure that travels between hosts, and both
rows above are the minimum of several runs on a shared box.

Where the time goes, what was tried instead, and the two instruments
(`IFAN_TIME=1`, `IFAN_CPU_TIME=1`) that produce the per-kernel and per-op tables
are in [docs/INTERNALS.md](docs/INTERNALS.md).

## Accuracy

Not measured against a benchmark here, because that needs the DPDD test set and
the upstream evaluation script. What IS measured is that this engine and the
upstream network compute the same thing. `tools/gate.py` runs the CPU path, the
CUDA path and `tools/reference.py` on the original PyTorch checkpoint over the
same image and reports the largest per-pixel difference of each backend from the
reference.

At 64x64, 256x256 and 720x720 the CPU path is within 2.0e-06 and the CUDA path
within 3.0e-06 of the torch reference on the 0..1 scale, i.e. at most 0.001 of one
8-bit level - the differences are float32 round-off, not visible in an 8-bit
image, and unchanged in magnitude at every size. The full table is in
[docs/INTERNALS.md](docs/INTERNALS.md). They are not zero because the three
implementations sum a convolution in different orders and only the CUDA kernels
contract `acc += w * x` into a single-rounding FFMA; [docs/NUMERICS.md](docs/NUMERICS.md)
explains which orders are free to differ and why the numbers must be read as a
bound rather than as digits (torch's own output moves in the last bits with its
thread count).

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license;
see [LICENSE](LICENSE).

This is an independent reimplementation of the IFAN architecture, which is by
Junyong Lee, Hyeongseok Son, Jaesung Rim, Sunghyun Cho and Seungyong Lee
(POSTECH), "Iterative Filter Adaptive Network for Single Image Defocus
Deblurring", CVPR 2021, and is released by them under the **AGPL-3.0**.
`tools/reference.py` is a PyTorch transcription of their network and
`tools/convert.py` transcribes their published configuration constants: both are
derived works, covered by the AGPL and **not** by this repository's MIT licence.
The demo input in `assets/before-after.png` is their
`demo/input/input.jpg`, and the **checkpoint** is their work as well - the
converted `IFAN.safetensors` attached to the releases is a format conversion of
their official `IFAN.pytorch`, redistributed under the same terms. The original
`.pytorch` file is not redistributed here. See
[MODEL_LICENSE-IFAN.txt](MODEL_LICENSE-IFAN.txt) for the upstream licence text
and for exactly which files it covers.
