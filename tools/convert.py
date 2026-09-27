#!/usr/bin/env python3
"""Convert IFAN.pytorch into the .safetensors this engine loads.

IFAN is by [codeslake](https://github.com/codeslake) and released under the
**AGPL-3.0** licence (CVPR 2021, "Iterative Filter Adaptive Network for
Single Image Defocus Deblurring"). The checkpoint tensors are the IFAN
authors' work and stay under those terms; see MODEL_LICENSE-IFAN.txt. This
script is a format conversion and is covered by that licence as a derivative,
not by the engine's MIT licence.

    python3 tools/convert.py IFAN.pytorch IFAN.safetensors

The released checkpoint has two halves:

* `module.Network.*` - the model. 158 tensors, the only ones this engine reads.
* `module.reblurNet.*` - a training-time re-blur branch (17 filters, its own
  encoder and a 357-channel head). It exists to synthesise training pairs and is
  never used for inference, so it is dropped here rather than shipped as dead
  weight. The conversion says how many tensors it dropped.

The Rust side reads the architecture constants out of `__metadata__` instead of
inferring them, for the same reason nafnet-rs does: `configs/config_IFAN.py`
defines them, the checkpoint does not, and a converted file that does not say
which iteration count it is cannot be shape-checked against its own weights.
Writing `N` here is what lets the engine have no `--iterations` flag: F.3's
15232 outputs ARE N (15232 = 17 * (128 * 6) + 17 * 128), and the loop is over
the same number.

Nothing is transposed: torch stores conv weights [c_out][c_in][kh][kw], which is
what both backends read.
"""
import argparse
import json
import struct
import sys

import numpy as np
import torch

NET_PREFIX = "module.Network."
REBLUR_PREFIX = "module.reblurNet."


def infer(state: dict, explicit: dict):
    """Read the architecture out of the weights, so the metadata cannot lie."""
    ch = state["conv1_1.0.weight"].shape[0]
    res_num = 2  # from config_IFAN.py; not recoverable from shapes, checked below
    ks = state["conv1_1.0.weight"].shape[2]

    # F.3 is a bare Conv2d: [N*(ch4*Fs*2) + N*ch4, ch4, 1, 1].
    f_out = state["F.3.weight"].shape[0]
    ch4 = state["F.3.weight"].shape[1]
    fs = state["F.3.weight"].shape[2]  # 1x1 head, Fs comes from the SAC config
    fs = explicit.get("fs", 3)
    n = f_out // (ch4 * (2 * fs + 1))
    if n * (ch4 * (2 * fs + 1)) != f_out:
        raise SystemExit(
            f"F.3 has {f_out} outputs, which is not N*ch4*(2*Fs+1) for ch4={ch4}, Fs={fs}; "
            "pass --fs if this is not a 3-tap IFAN"
        )
    if explicit.get("n") is not None and explicit["n"] != n:
        raise SystemExit(f"--n {explicit['n']} but the weights say N={n}")
    if explicit.get("ch") is not None and explicit["ch"] != ch:
        raise SystemExit(f"--ch {explicit['ch']} but the weights say ch={ch}")
    return dict(ch=ch, n=n, fs=fs, ks=ks, res_num=res_num, ch4=ch4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n", type=int, default=None, help="IAC iteration count (default: from F.3)")
    ap.add_argument("--ch", type=int, default=None, help="base channels (default: from conv1_1)")
    ap.add_argument("--fs", type=int, default=3, help="SAC filter size")
    ap.add_argument("--keep-reblur", action="store_true",
                    help="also convert the training-only reblur branch (not used by the engine)")
    args = ap.parse_args()

    raw = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        print(f"{args.src}: not a state dict ({type(raw)})", file=sys.stderr)
        return 1

    net = {k[len(NET_PREFIX):]: v for k, v in raw.items()
           if k.startswith(NET_PREFIX) and torch.is_tensor(v)}
    if not net:
        print(f"{args.src}: no {NET_PREFIX}* tensors - not an IFAN checkpoint?", file=sys.stderr)
        return 1
    dropped = [k for k in raw if k.startswith(REBLUR_PREFIX)]
    other = [k for k in raw if not k.startswith(NET_PREFIX) and not k.startswith(REBLUR_PREFIX)]

    a = infer(net, vars(args))
    if not args.keep_reblur and other:
        print(f"warning: {len(other)} tensors are under neither prefix, dropping: {other[:4]}",
              file=sys.stderr)

    # The one architecture constant shapes cannot recover, so it is checked
    # instead: res_num=2 means each ResnetBlock has exactly stem.0 and stem.1,
    # and conv4_4's resnet blocks are the only ResnetBlock outside conv_res.
    # "conv4_4.1.stem.1.0.weight" -> block "conv4_4.1", stem "1".
    def stems(block):
        pre = block + ".stem."
        return len({k[len(pre):].split(".")[0] for k in net if k.startswith(pre)})

    for block in ("conv4_4.1", "conv4_4.2", "DME.1", "DME.2", "F.1", "F.2"):
        got = stems(block)
        if got != a["res_num"]:
            print(f"{args.src}: {block} has {got} stems, expected res_num={a['res_num']}",
                  file=sys.stderr)
            return 1
    if stems("conv_res.1") != 3:
        print(f"{args.src}: conv_res.1 is not the res_num=3 block this port assumes", file=sys.stderr)
        return 1

    metadata = {
        "format": "pt",
        "arch": "ifan",
        "ch": str(a["ch"]),
        "n": str(a["n"]),
        "fs": str(a["fs"]),
        "ks": str(a["ks"]),
        "res_num": str(a["res_num"]),
        "ch4": str(a["ch4"]),
        # Which published figure this checkpoint is the source of, if any. The
        # engine prints it and the README's accuracy table cites it - as the
        # AUTHORS' number, never as one measured here (the DPDD test set is not
        # obtainable any more, see the README).
        "source": "IFAN.pytorch (codeslake/IFAN, CVPR 2021)",
    }

    items = sorted(net.items())
    if args.keep_reblur:
        items += sorted((k[len(REBLUR_PREFIX):], v) for k, v in raw.items()
                        if k.startswith(REBLUR_PREFIX) and torch.is_tensor(v))

    offset = 0
    header = {}
    for name, t in items:
        arr = t.detach().to(torch.float32).contiguous().numpy()
        nbytes = arr.size * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        # Pad each tensor to a 4-byte boundary: the engine mmaps the file and
        # hands out `&[f32]` slices, so a tensor starting mid-word is an
        # unaligned read, which the loader refuses rather than misreading.
        offset += (nbytes + 3) & ~3
    header["__metadata__"] = metadata
    hjson = json.dumps(header, separators=(",", ":")).encode()

    head = len(struct.pack("<Q", 0)) + len(hjson)
    pad = (-head) & 7
    with open(args.dst, "wb") as f:
        f.write(struct.pack("<Q", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        written = 0
        for name, t in items:
            arr = t.detach().to(torch.float32).contiguous().numpy()
            raw_bytes = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            want = header[name]["data_offsets"][0]
            if written < want:
                f.write(b"\0" * (want - written))
                written = want
            f.write(raw_bytes)
            written += len(raw_bytes)

    values = sum(t.numel() for _, t in items)
    print(f"{args.dst}: {len(items)} tensors, {values} values "
          f"({values * 4 / 2**20:.1f} MiB)")
    print(f"  ch={a['ch']} N={a['n']} Fs={a['fs']} ks={a['ks']} res_num={a['res_num']} ch4={a['ch4']}")
    print(f"  dropped {len(dropped)} training-only reblurNet tensors"
          + ("" if not dropped else f" (e.g. {dropped[0]})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
