# ifan-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

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
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk.
* The architecture is **iterative**: the network predicts a *filter tensor* -
  15232 floats per 1/8-scale pixel, a separable 3x3 pair and a bias for each of
  17 iterations - and then applies it with an adaptive convolution whose weights
  change per pixel. The filter tensor for a 1920x1080 input is 1.84 GiB on its
  own, which is why memory gets its own section below rather than a footnote.
* No tile size to choose and none to get wrong: the pass is measured before it
  starts, and a pass that will not fit is refused with both numbers rather than
  launched and killed.
* Both backends agree with the upstream PyTorch implementation to float32
  round-off; see [docs/NUMERICS.md](docs/NUMERICS.md) for why they are not
  bit-identical and how to check a change.

## Download

Prebuilt binary and the converted checkpoint are attached to the
[releases](https://github.com/jacobsparts/ifan-rs/releases).

| asset | what it is |
|---|---|
| `ifan-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs a compute capability 6.1+ GPU, `--cpu` runs the pure-Rust path anywhere |
| `IFAN.safetensors` | the converted checkpoint: defocus deblurring, the paper's final model, 10.477 M parameters, 17 iterations |

```sh
chmod +x ifan-linux-x86_64
./ifan-linux-x86_64 -m IFAN.safetensors -i blurry.png -o sharp.png
```

## Models

One model, chosen with `-m`; there is no task flag because IFAN is trained for
one. The upstream authors also publish `IFAN_8`, `IFAN_26`, `IFAN_35` and
`IFAN_44` (N=8, 26, 35, 44 - accuracy against speed), plus a 16-bit variant and a
dual-pixel variant. The engine reads the iteration count out of the converted
file's metadata rather than from a flag, so any of them converts and runs:

```sh
python3 tools/convert.py IFAN_26.pytorch IFAN_26.safetensors
```

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

There is no fallback. The measured requirement is compared against free VRAM
before a byte is allocated, and a pass that does not fit says what it needed and
what there was:

```
$ ifan -m IFAN.safetensors -i 4096.png -o out.png
ifan: 4096x4096 does not fit in VRAM: the pass allocates 24.94 GiB of device buffer and 7.78 GiB is free (7.59 GiB usable after a 192.0 MiB reserve). A smaller image is the only way through - this engine has no smaller layout to fall back on.
$ echo $?
1
```

`--cpu` needs no GPU, and its footprint is bounded by HOST memory rather than
VRAM; the CPU path's own plan is guarded the same way against `MemAvailable`:

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
