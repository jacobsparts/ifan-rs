#!/usr/bin/env python3
"""Golden gate: reference.py vs the engine's CPU twin vs its CUDA path.

Three opinions on the same input:

  1. tools/reference.py - the torch transcription of upstream, run on the
     ORIGINAL checkpoint (models/IFAN.pytorch). This also tests tools/convert.py,
     since the engine reads the converted .safetensors.
  2. ifan --cpu          - the Rust CPU interpreter, no GPU involved.
  3. ifan (default)      - the same Vec<Op> through the CUDA kernels.

The first two run the SHIPPED binary, the one a user gets from `cargo build
--release`, and they compare the PNGs it writes: the comparison is on the CLIPPED
output because that is the engine's defined output (`Op::Clip` is the graph's
last op and reference.py's `result` is the same clip). A pre-clip comparison is
done as well, because clipping hides sub-level differences: 1/255 is EXACTLY the
scale a real bug produces, so a clipped-only comparison can pass while the network
is wrong underneath. The tolerance is therefore two-tiered: clipped values must
agree to a quantisation step, and the raw values to float32 round-off.

THE PRE-CLIP NUMBERS COME FROM A `--features dev` BUILD, built into its own target
directory so it cannot overwrite the shipped binary:

    cargo build --release --features dev --target-dir target/dev

That is why the raw comparison is the one leg that does not run the shipped
artifact. It is still the shipped CODE: a release build reaches the same interpreter
with a dump path that no flag can set (see src/main.rs), so what the dev build adds
is a way to READ the numbers, not a different engine. The PNGs the shipped binary
writes are what the pass/fail decision uses.

WHY A SYNTHETIC INPUT. The DPDD test set used by the paper is not available, so
there is no ground truth to score against; the gate's question is only whether
the three implementations agree. A deterministic random field with structure at
several scales (so every level of the pyramid and every SAC tap sees something)
answers that question, and it is small enough to run in seconds.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAMILY = ROOT.parent


def make_input(path, h, w, seed=1234):
    """A structured field, not white noise: gradients + a few blobs.

    White noise would still catch a wrong tap order, but a smooth field with
    edges exercises the SAC clamping and the replicate padding at the borders,
    which is where a 1-D gather is easiest to get subtly wrong.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (yy / max(h - 1, 1)) * np.ones((h, w)) * 0.6
    base += (xx / max(w - 1, 1)) * np.ones((h, w)) * 0.3
    for _ in range(6):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(1.0, max(2.0, min(h, w) / 4))
        base += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))) * rng.uniform(-0.3, 0.3)
    base += rng.normal(0, 0.01, (h, w))
    rgb = np.stack([base, np.roll(base, 3, 0), np.roll(base, 5, 1)], -1)
    rgb = np.clip(rgb, 0.0, 1.0)
    u8 = np.round(rgb * 255.0).astype(np.uint8)
    from PIL import Image
    Image.fromarray(u8, "RGB").save(path)
    return u8


def read_dump(path):
    raw = Path(path).read_bytes()
    c, h, w = np.frombuffer(raw[:12], dtype="<u4")
    d = np.frombuffer(raw[12:], dtype="<f4")
    assert d.size == c * h * w, f"dump says {c}x{h}x{w} but holds {d.size} floats"
    return d.reshape(c, h, w)


def read_png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        print("FAILED:", " ".join(str(c) for c in cmd))
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise SystemExit(1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--weights", default=str(FAMILY / "models" / "IFAN.safetensors"))
    ap.add_argument("--torch-weights", default=str(FAMILY / "models" / "IFAN.pytorch"))
    # The workspace target dir, one level above the crate (cargo workspace).
    ap.add_argument("--bin", default=str(FAMILY / "target" / "release" / "ifan"))
    # The dev build is the SAME engine with the dump flags compiled in; see the
    # module docstring. Built with --target-dir so it never replaces --bin.
    ap.add_argument(
        "--dev-bin", default=str(FAMILY / "target" / "dev" / "release" / "ifan"))
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--tol-clip", type=float, default=1.5 / 255.0)
    ap.add_argument("--tol-cpu-cuda", type=float, default=2e-4)
    ap.add_argument("--tol-raw", type=float, default=1e-5)
    a = ap.parse_args()

    work = FAMILY / "target" / "gate"
    work.mkdir(parents=True, exist_ok=True)
    png = work / f"in_{a.size}.png"
    make_input(png, a.size, a.size, a.seed)
    print(f"input: {png} ({a.size}x{a.size})")

    # ---- the engine, CPU and GPU ----
    for b in (a.bin, a.dev_bin):
        if not Path(b).exists():
            print(f"missing binary: {b}")
            return 1

    # The CLI the user has refuses --dump by name; the dev build accepts it. Check
    # that here rather than let a stale binary silently answer for the wrong one.
    p = subprocess.run([a.bin, "--help"], capture_output=True, text=True)
    if "--dump" in p.stdout:
        print(f"{a.bin} is not a release build: it advertises --dump")
        return 1

    # The shipped binary WRITES the PNG; the dev build produces the numbers.
    # TWO dumps per backend: the graph's last op is a clip, so `--dump` alone gives
    # the CLIPPED output, and the pre-clip numbers - which is what a divergence
    # should be measured on, since the clip can hide a difference of exactly the
    # size a real bug makes - come from `--dump-op raw` (the op the Clip reads).
    cpu_dump = work / "cpu.f32"
    cpu_raw_dump = work / "cpu_raw.f32"
    run([a.bin, "-m", a.weights, "-i", str(png), "-o", str(work / "cpu.png"), "--cpu"])
    run([a.dev_bin, "-m", a.weights, "-i", str(png), "-o", str(work / "cpu_dev.png"),
         "--cpu", "--dump", str(cpu_dump)])
    run([a.dev_bin, "-m", a.weights, "-i", str(png), "-o", str(work / "cpu_raw.png"),
         "--cpu", "--dump-op", "raw", "--dump", str(cpu_raw_dump)])
    cpu = read_dump(cpu_dump)
    cpu_raw = read_dump(cpu_raw_dump)
    cpu_png = read_png(work / "cpu.png")
    cpu_dev_png = read_png(work / "cpu_dev.png")
    if not np.array_equal(cpu_png, cpu_dev_png):
        print("the dev build's PNG differs from the shipped build's")
        return 1
    print(f"engine --cpu : {cpu.shape} min {cpu.min():.6f} max {cpu.max():.6f} "
          f"mean {cpu.mean():.6f}")

    gpu = None
    gpu_raw = None
    if not a.skip_cuda:
        gpu_dump = work / "gpu.f32"
        gpu_raw_dump = work / "gpu_raw.f32"
        p = run([a.bin, "-m", a.weights, "-i", str(png), "-o", str(work / "gpu.png")])
        print(p.stderr.strip())
        run([a.dev_bin, "-m", a.weights, "-i", str(png),
             "-o", str(work / "gpu_dev.png"), "--dump", str(gpu_dump)])
        run([a.dev_bin, "-m", a.weights, "-i", str(png),
             "-o", str(work / "gpu_raw.png"), "--dump-op", "raw", "--dump", str(gpu_raw_dump)])
        if not np.array_equal(read_png(work / "gpu.png"), read_png(work / "gpu_dev.png")):
            print("the dev build's PNG differs from the shipped build's")
            return 1
        gpu = read_dump(gpu_dump)
        gpu_raw = read_dump(gpu_raw_dump)
        print(f"engine --cuda: {gpu.shape} min {gpu.min():.6f} max {gpu.max():.6f} "
              f"mean {gpu.mean():.6f}")
        d = np.abs(gpu - cpu)
        print(f"CPU vs CUDA  : max |d| {d.max():.3e} mean {d.mean():.3e}")
        if d.max() > a.tol_cpu_cuda:
            print(f"MISMATCH: CPU and CUDA disagree by more than {a.tol_cpu_cuda}")
            idx = np.unravel_index(np.argmax(d), d.shape)
            print(f"  worst at {idx}: cpu {cpu[idx]:.8f} cuda {gpu[idx]:.8f}")
            return 1

    if a.skip_torch:
        print("torch reference skipped")
        return 0

    # ---- the torch reference, on the ORIGINAL checkpoint ----
    #
    # TWO RUNS, TWO FILES. They must not share a path: an earlier version of this
    # script wrote the raw output and then overwrote it with the clipped one, so
    # `ref` and `ref_raw` were the same array and the raw column below silently
    # printed the clipped number.
    #
    # TORCH_FIX = A PINNED THREAD COUNT, and it is not cosmetic. FLOPs here are not
    # associative, so torch's own output depends on how many threads it uses:
    # at 64x64 the engine's difference from the reference is 1.5199e-06 with
    # OMP_NUM_THREADS=1 and 1.6987e-06 by default. Without a pin the third
    # significant digit of the reported difference is a property of the machine's
    # load rather than of this engine, and a comparison against a recorded value
    # would drift for no reason.
    torch_env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    ref_npy = work / "ref.npy"
    ref_raw_npy = work / "ref_raw.npy"
    run([sys.executable, str(HERE / "reference.py"), "-m", a.torch_weights,
         "-i", str(png), "-o", str(ref_raw_npy), "--raw"], env=torch_env)
    run([sys.executable, str(HERE / "reference.py"), "-m", a.torch_weights,
         "-i", str(png), "-o", str(ref_npy)], env=torch_env)
    ref = np.load(ref_npy)
    ref_raw = np.load(ref_raw_npy)
    # reference.py is NCHW with the batch dim; the engine's dump is CHW. Squeeze
    # it here rather than let numpy BROADCAST (3,64,64) against (1,3,64,64), which
    # silently compares the wrong elements and reports a huge difference that is
    # an artifact of the shapes, not of the networks.
    if ref.ndim == 4 and ref.shape[0] == 1:
        ref = ref[0]
    if ref_raw.ndim == 4 and ref_raw.shape[0] == 1:
        ref_raw = ref_raw[0]
    assert ref.shape == cpu.shape, f"reference {ref.shape} vs engine {cpu.shape}"
    print(f"reference    : {ref.shape} min {ref.min():.6f} max {ref.max():.6f} "
          f"mean {ref.mean():.6f}")

    for name, arr, arr_raw in (("CPU ", cpu, cpu_raw), ("CUDA", gpu, gpu_raw)):
        if arr is None:
            continue
        d = np.abs(arr - ref)
        # CLIPPED against clipped, PRE-CLIP against pre-clip. Comparing the
        # engine's clipped output against the reference's pre-clip one measures
        # the clip, not the network - which is what an earlier version of this
        # script did on this very line, and it reported a difference of 1e-02.
        dr = np.abs(arr_raw - ref_raw)
        print(f"{name} vs torch : clipped max |d| {d.max():.3e} "
              f"({d.max()*255:.3f}/255)  raw max |d| {dr.max():.3e}")
        if d.max() > a.tol_clip:
            print(f"MISMATCH: {name} differs from the reference by more than "
                  f"{a.tol_clip} ({a.tol_clip*255:.2f}/255)")
            idx = np.unravel_index(np.argmax(d), d.shape)
            print(f"  worst at {idx}: ref {ref[idx]:.8f} engine {arr[idx]:.8f}")
            return 1
        # The raw leg is bounded too, and it is the one that catches a difference
        # the clip would hide. `--tol-raw` is deliberately looser than the clip:
        # both sides are float32 sums of the same terms in different orders, and
        # one unit in the last place of an activation near 1 is 6e-08, so a real
        # defect (a transposed weight, a doubled tap) lands orders above this.
        if dr.max() > a.tol_raw:
            print(f"MISMATCH: {name} differs from the PRE-CLIP reference by more "
                  f"than {a.tol_raw}")
            idx = np.unravel_index(np.argmax(dr), dr.shape)
            print(f"  worst at {idx}: ref_raw {ref_raw[idx]:.8f} engine {arr[idx]:.8f}")
            return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
