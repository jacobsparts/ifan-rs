#!/usr/bin/env python3
"""IFAN reference: the graph as the released checkpoint defines it.

This is a transcription of codeslake/IFAN's `models/archs/IFAN.py`, `models/IAC.py`
and `models/utils.py` (AGPLv3 - see LICENSE and MODEL_LICENSE-IFAN.txt), written
to be readable next to the Rust engine and to be the thing the Rust engine is
compared against. It is deliberately NOT a copy of the upstream package: it
takes a state_dict directly, has no `config` object, no logging and no dataset
code.

THREE THINGS IN HERE LOOK LIKE BUGS AND ARE FAITHFUL:

1. `IAC` ignores `kernel2`. Upstream `models/IAC.py:37` multiplies the
   horizontal pass by `kernel1` a second time, where its own comment says
   `kernel2` was meant. Every released checkpoint was trained through that
   path, so reproducing it is not optional - a "fixed" port would not match the
   weights. `IFAN_REF_FIX_KERNEL2=1` switches to the intended form, which
   exists only to MEASURE how much the bug costs.
2. `refine_image` CROPS rather than pads, so an image whose sides are not
   multiples of 8 is evaluated on a crop. That is what `predict.py` and
   `eval.py` both do, so it is the deployment contract, not just an eval quirk.
3. The last IAC iteration IS activated. The paper's IAC figure shows an LReLU
   after every block; upstream's `is_act_last` defaults to True, so N
   activations, and the weights were trained that way.

The convolution padding in SAC is `mode="replicate"` in BOTH passes, and the
two passes DO NOT collapse into a single gather. The sample indices do compose
into a clamped map - (clamp(y+k-pad), clamp(x+k2-pad)) - but the TAP does not:
the horizontal pass reads the vertical pass's output at column
`clamp(x+k2-pad)`, and the tap the vertical sum applied to that sample is
`kv[y][clamp(x+k2-pad)][k]`, not `kv[y][x][k]`. Measured on a 5x7x2 case in
float64, the exact composed form matches this two-pass code to 3.6e-15 while a
form with every tap at (y, x) is off by 34.5 absolute. SAC is therefore two
sequential passes with an intermediate plane, in the reference and in both Rust
backends alike - see the long comment above `if_sac_vertical` in cuda/ifan.cu.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# config, from configs/config_IFAN.py
# ---------------------------------------------------------------------------
CH = 32
KS = 3
FS = 3          # IAC filter (kernel) size
N_ITER = 17
RES_NUM = 2
REFINE_VAL = 8
NORM_VAL = 255


# ---------------------------------------------------------------------------
# nn_common.py
# ---------------------------------------------------------------------------
def conv(in_ch, out_ch, kernel_size=KS, stride=1, dilation=1, bias=True, act=True):
    """Upstream `conv`. `act=False` returns a BARE Conv2d, not a 1-element
    Sequential - the distinction is load-bearing: a bare module names its
    weights `DME.3.weight`, a Sequential names them `DME.3.0.weight`, and the
    released checkpoint uses the bare form for the two act=None convs (the DME
    head and F's 1x1 head)."""
    m = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                  dilation=dilation, padding=((kernel_size - 1) // 2) * dilation,
                  bias=bias)
    if not act:
        return m
    return nn.Sequential(m, nn.LeakyReLU(0.1, inplace=True))


def upconv(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ResnetBlock(nn.Module):
    """`resnet_block`, including the `res_num > 1` skip that is easy to miss.

    forward: for each of res_num stem[i]: x = x + stem[i](x); x = lrelu(x)
             and if res_num > 1: x = x + temp (the block's INPUT, not the
             running value).
    """

    def __init__(self, ch, kernel_size=KS, dilation=(1, 1), bias=True, res_num=1):
        super().__init__()
        self.res_num = res_num
        self.stem = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=kernel_size, stride=1, dilation=dilation[0],
                          padding=((kernel_size - 1) // 2) * dilation[0], bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(ch, ch, kernel_size=kernel_size, stride=1, dilation=dilation[1],
                          padding=((kernel_size - 1) // 2) * dilation[1], bias=bias),
            ) for _ in range(res_num)
        ])

    def forward(self, x):
        temp = x if self.res_num > 1 else None
        for i in range(self.res_num):
            x = x + self.stem[i](x)
            x = F.leaky_relu(x, 0.1)
        if temp is not None:
            x = x + temp
        return x


# ---------------------------------------------------------------------------
# IAC.py
# ---------------------------------------------------------------------------
def sac(feat_in, k1, k2, ksize, fix_kernel2=False):
    """Separable adaptive convolution, 1-D vertical then 1-D horizontal.

    `k1`, `k2` are [N, c*ksize, H, W]: one ksize-vector per (position, channel).
    Returns [N, c, H, W].
    """
    n, ck, h, w = k1.shape
    c = feat_in.size(1)
    pad = (ksize - 1) // 2

    x = F.pad(feat_in, (0, 0, pad, pad), mode="replicate")
    x = x.unfold(2, ksize, 1).permute(0, 2, 3, 1, 4).reshape(n, h, w, c, ksize)
    kk = k1.permute(0, 2, 3, 1).reshape(n, h, w, c, ksize)
    x = torch.sum(torch.mul(x, kk), -1).permute(0, 3, 1, 2)

    x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
    x = x.unfold(3, ksize, 1).permute(0, 2, 3, 1, 4).reshape(n, h, w, c, ksize)
    kk2 = (k2 if fix_kernel2 else k1).permute(0, 2, 3, 1).reshape(n, h, w, c, ksize)
    x = torch.sum(torch.mul(x, kk2), -1).permute(0, 3, 1, 2)
    return x


def iac(feat_in, filt, n_iter, c, ksize, is_act_last=True, fix_kernel2=False):
    fs = torch.split(filt[:, :n_iter * (c * ksize * 2), :, :], c * ksize * 2, dim=1)
    f_bs = torch.split(filt[:, n_iter * (c * ksize * 2):, :, :], c, dim=1)
    f = feat_in
    for i in range(n_iter):
        f1, f2 = torch.split(fs[i], c * ksize, dim=1)
        f = sac(f, f1, f2, ksize, fix_kernel2=fix_kernel2)
        f = f + f_bs[i]
        if i < (n_iter - 1) or is_act_last:
            f = F.leaky_relu(f, 0.1)
    return f


# ---------------------------------------------------------------------------
# archs/IFAN.py
# ---------------------------------------------------------------------------
class Network(nn.Module):
    """The single-image (C) path of the network, as `run.py`/`predict.py` use it."""

    def __init__(self, ch=CH, n_iter=N_ITER, fs=FS, res_num=RES_NUM):
        super().__init__()
        self.ch4 = ch * 4
        self.N = n_iter
        self.Fs = fs
        ch1, ch2, ch3, ch4 = ch, ch * 2, ch * 4, ch * 4

        self.conv1_1 = conv(3, ch1)
        self.conv1_2 = conv(ch1, ch1)
        self.conv1_3 = conv(ch1, ch1)
        self.conv2_1 = conv(ch1, ch2, stride=2)
        self.conv2_2 = conv(ch2, ch2)
        self.conv2_3 = conv(ch2, ch2)
        self.conv3_1 = conv(ch2, ch3, stride=2)
        self.conv3_2 = conv(ch3, ch3)
        self.conv3_3 = conv(ch3, ch3)
        self.conv4_1 = conv(ch3, ch4, stride=2)
        self.conv4_2 = conv(ch4, ch4)
        self.conv4_3 = conv(ch4, ch4)
        self.conv4_4 = nn.Sequential(
            conv(2 * ch4, ch4),
            ResnetBlock(ch4, res_num=res_num),
            ResnetBlock(ch4, res_num=res_num),
            conv(ch4, ch4),
        )
        self.conv_res = nn.Sequential(
            conv(ch4, ch4),
            ResnetBlock(ch4, res_num=3),
            conv(ch4, ch4),
        )
        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = ResnetBlock(ch3, res_num=1)
        self.upconv3_2 = ResnetBlock(ch3, res_num=1)
        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = ResnetBlock(ch2, res_num=1)
        self.upconv2_2 = ResnetBlock(ch2, res_num=1)
        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = ResnetBlock(ch1, res_num=1)
        self.upconv1_2 = ResnetBlock(ch1, res_num=1)
        self.out_res = conv(ch1, 3)

        self.kconv1_1 = conv(3, ch1)
        self.kconv1_2 = conv(ch1, ch1)
        self.kconv1_3 = conv(ch1, ch1)
        self.kconv2_1 = conv(ch1, ch2, stride=2)
        self.kconv2_2 = conv(ch2, ch2)
        self.kconv2_3 = conv(ch2, ch2)
        self.kconv3_1 = conv(ch2, ch3, stride=2)
        self.kconv3_2 = conv(ch3, ch3)
        self.kconv3_3 = conv(ch3, ch3)
        self.kconv4_1 = conv(ch3, ch4, stride=2)
        self.kconv4_2 = conv(ch4, ch4)
        self.kconv4_3 = conv(ch4, ch4)

        self.DME = nn.Sequential(
            conv(ch4, ch4),
            ResnetBlock(ch4, res_num=res_num),
            ResnetBlock(ch4, res_num=res_num),
            conv(ch4, 1, kernel_size=3, act=False),
        )
        self.conv_DME = conv(1, ch4)
        self.kernel_dim = n_iter * (ch4 * fs * 2) + n_iter * ch4
        self.F = nn.Sequential(
            conv(ch4, ch4),
            ResnetBlock(ch4, res_num=res_num),
            ResnetBlock(ch4, res_num=res_num),
            conv(ch4, self.kernel_dim, kernel_size=1, act=False),
        )

    def forward(self, c_in, is_train=False):
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(c_in)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f_c = self.conv4_3(self.conv4_2(self.conv4_1(f3)))

        f = self.kconv1_3(self.kconv1_2(self.kconv1_1(c_in)))
        f = self.kconv2_3(self.kconv2_2(self.kconv2_1(f)))
        f = self.kconv3_3(self.kconv3_2(self.kconv3_1(f)))
        f = self.kconv4_3(self.kconv4_2(self.kconv4_1(f)))

        dm = self.DME(f)
        f_dm = self.conv_DME(dm)
        f = self.conv4_4(torch.cat([f, f_dm], 1))
        filt = self.F(f)

        x = iac(f_c, filt, self.N, self.ch4, self.Fs,
                fix_kernel2=bool(os.environ.get("IFAN_REF_FIX_KERNEL2")))
        x = self.conv_res(x)

        x = self.upconv3_u(x) + f3
        x = self.upconv3_2(self.upconv3_1(x))
        x = self.upconv2_u(x) + f2
        x = self.upconv2_2(self.upconv2_1(x))
        x = self.upconv1_u(x) + f1
        x = self.upconv1_2(self.upconv1_1(x))
        out = self.out_res(x) + c_in

        outs = {"raw": out}
        outs["result"] = torch.clip(out, 0, 1.0) if not is_train else out
        if is_train:
            outs["Filter"] = filt
        return outs


# ---------------------------------------------------------------------------
# data_loader/utils.py + predict.py's input path
# ---------------------------------------------------------------------------
def read_frame(path, norm_val=NORM_VAL):
    """RGB in [0,1], float32, HWC."""
    import cv2
    if norm_val == (2 ** 16 - 1):
        frame = cv2.imread(path, -1)
        frame = np.flip(frame, -1).astype(np.float32)
        frame = frame / norm_val
    else:
        frame = cv2.imread(path)
        frame = np.flip(frame, -1).astype(np.float32)
        frame = frame / norm_val
    return np.ascontiguousarray(frame[None, ...])


def load_frame_rgb(path, norm_val=NORM_VAL):
    """The engine's input contract: RGB HWC float32 in [0,1], no crop.

    Equivalent to read_frame, but written without cv2 so the golden generator
    does not depend on an OpenCV build (the engine reads PNGs with `png`).
    """
    from PIL import Image
    with Image.open(path) as im:
        if norm_val == (2 ** 16 - 1):
            a = np.asarray(im.convert("I;16")) if im.mode.startswith("I") else np.asarray(im)
            scale = float(norm_val)
        else:
            a = np.asarray(im.convert("RGB"))
            scale = float(norm_val)
    return np.ascontiguousarray(a.astype(np.float32)[None, ...] / scale)


def refine_image(image, val):
    """CROP (h, w) down to multiples of `val`. Upstream's refine_image."""
    h, w = image.shape[1], image.shape[2]
    image = image[:, 0:h - (h % val), 0:w - (w % val)]
    return np.ascontiguousarray(image)


def preprocess(path, val=REFINE_VAL, norm_val=NORM_VAL, max_side=None):
    """The full input path of predict.py: read -> (downscale) -> crop. CHW NCHW."""
    x = load_frame_rgb(path, norm_val)
    _, h, w, _ = x.shape
    if max_side is not None and max(h, w) > max_side:
        import cv2
        r = max_side / max(h, w)
        x = np.expand_dims(cv2.resize(x[0], dsize=(int(w * r), int(h * r)),
                                      interpolation=cv2.INTER_AREA), 0)
    x = refine_image(x, val)
    return x.transpose(0, 3, 1, 2).copy()


def load_state_dict(path, key="module.Network."):
    """Load IFAN.pytorch and keep only the Network half, unprefixed.

    The checkpoint also carries `module.reblurNet.*` (a training-only re-blur
    branch) which is dropped here and never converted.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in sd.items():
        if not k.startswith(key):
            continue
        if torch.is_tensor(v):
            out[k[len(key):]] = v
    return out


def build(ckpt, ch=CH, n_iter=N_ITER, fs=FS, res_num=RES_NUM):
    net = Network(ch=ch, n_iter=n_iter, fs=fs, res_num=res_num).eval()
    missing, unexpected = net.load_state_dict(load_state_dict(ckpt), strict=True), None
    return net


def main():
    ap = argparse.ArgumentParser(description="IFAN reference forward pass (torch, CPU).")
    ap.add_argument("-m", "--model", default="/home/jacob/lightgpu-family/models/IFAN.pytorch")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True, help=".npy of the clipped output, NCHW f32")
    ap.add_argument("--raw", action="store_true",
                    help="write the PRE-clip output (out['raw']) instead, which is what "
                         "a divergence should be measured on - clipping hides them")
    ap.add_argument("--refine-val", type=int, default=REFINE_VAL)
    ap.add_argument("--norm-val", type=float, default=NORM_VAL)
    a = ap.parse_args()

    net = build(a.model)
    x = torch.from_numpy(preprocess(a.input, a.refine_val, a.norm_val))
    with torch.no_grad():
        o = net(x, is_train=False)
    arr = (o["raw"] if a.raw else o["result"]).numpy()
    np.save(a.output, arr)
    print(f"ifan-ref: {a.input} -> {a.output} {arr.shape}")


if __name__ == "__main__":
    main()
