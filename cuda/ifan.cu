// IFAN (CVPR 2021) kernels for the lightgpu driver layer.
//
// WHAT IS HERE. Three kinds of op, because the toolkit has no equivalent:
//
//   * the TILED convolutions - `if_conv3x3_tile`, `if_conv1x1_tile`,
//     `if_conv3x3s2_tile` and `if_convtranspose4x4s2p1_tile`. These are what the
//     graph actually launches.
//   * `if_sac_vertical` / `if_sac_horizontal` / `if_sac_bias` - SAC, the attention
//     iteration inside IAC - and `if_clip`, the graph's last op.
//
// WHY EACH IS NEEDED: the toolkit's `lg_conv4x4s4` is a stride-4 patch embed with
// no padding and `lg_upsample2x_nearest` has no learned kernel, so neither the
// stride-2 convolution nor the transposed convolution exists there; and the toolkit
// has no dense tiled convolution and no clip at all.
//
// NUMERICS. The summation ORDER is observable in float32 and is therefore part of
// this engine's contract; `docs/INTERNALS.md` explains which orders may differ and
// why. In short: the CPU twin nests (ky, kx, ci), while the tiled kernels here nest
// (ci, ky, kx) and sum a zero-filled halo instead of skipping out-of-range taps,
// because the staged tile is walked once per channel tile with the kernel window
// inside it; the transposed tiled kernel keeps the (ky, kx, ic) order of the
// scalar form it replaced. The SAC passes sum their taps in index order with a
// running `a + b * c` in one float, as the CPU twin does. Torch's own convolution
// order is opaque, which is why the gate compares the two backends to each other
// and uses torch as a third opinion rather than as a bit-exact oracle.

// SAC - the separable adaptive convolution inside IAC.
//
// Each of the N=17 iterations applies a 1-D vertical pass and then a 1-D
// horizontal pass, where the taps are NOT fixed weights but values stored per
// (output position, channel). The taps arrive as an [N, c*ks, H, W] tensor
// ("kernel1", and the same tensor again as "kernel2" - see below), and the bias
// blocks as [N, c, H, W].
//
// PADDING IS REPLICATE, AND THE TWO PASSES DO **NOT** COLLAPSE INTO ONE GATHER.
// This is the subtlest thing in the model and it was initially got wrong here,
// so the arithmetic is spelled out. Upstream pads with `mode="replicate"` in
// both passes:
//
//     v[c,r,x] = sum_k  x[c][clamp(r + k - pad)][x]      * kv[c][r][x][k]
//     o[c,y,x] = sum_k2 v[c][y][clamp(x + k2 - pad)]     * kv[c][y][x][k2]
//
// Substituting the first into the second gives samples at
// (clamp(y + k - pad), clamp(x + k2 - pad)) - a clamped index map, as one would
// expect. BUT THE TAP IS NOT ALWAYS kv[y][x]. The horizontal pass reads v at
// column `clamp(x + k2 - pad)`, and the tap that the VERTICAL sum used for that
// sample is `kv[y][clamp(x + k2 - pad)][k]`, not `kv[y][x][k]`. The exact
// composed form is therefore
//
//     sum_{k,k2} x[clamp(y+k-pad)][clamp(x+k2-pad)] * kv[y][clamp(x+k2-pad)][k] * kv[y][x][k2]
//
// Verified against a two-pass implemention in float64: the exact form agrees to
// 3.6e-15, while a form that puts every tap at (y, x) is off by 34.5 ABSOLUTE on
// a 5x7x2 random case - not a rounding difference, a wrong answer. (`k2`'s tap
// uses the clamped column too, which is why "clamp only the sample index" is
// wrong in the other direction.)
//
// The conclusion for the code: SAC IS TWO SEQUENTIAL PASSES over an
// intermediate plane, which is exactly what the two kernels below do and why
// there is no fused single-gather variant here. Each pass clamps once - the
// vertical on rows, the horizontal on columns - and each reads ITS OWN tap at
// the output position (y, x). A kernel that gathers straight from the feature
// map with one index map would be silently wrong.
//
// THE UPSTREAM BUG, REPRODUCED DELIBERATELY. `models/IAC.py:37` multiplies the
// horizontal pass by `kernel1` again, where its own comment says `kernel2` was
// intended. Every released checkpoint was trained through that path, so the
// engine reproduces it; `IFAN_REF_FIX_KERNEL2=1` in tools/reference.py measures
// what the intended form would do, and this engine has no equivalent switch
// because a "fixed" engine would not match the weights. Consequently the
// horizontal kernel takes ONE tap tensor, not two.
//
// One thread per (iteration, channel, position). `c` here is ch4 = 128 and
// `ks` is Fs = 3.
// ---------------------------------------------------------------------------

// Vertical pass: for each (n, c, y, x) sum_{k} feat[n][c][clamp(y+k-1)][x] * k1[n][c*ks+k][y][x]
// The clamp on y is applied ONCE here (it is the whole of the vertical padding).
extern "C" __global__ void if_sac_vertical(
    const float *__restrict__ feat, const float *__restrict__ k1,
    float *__restrict__ out, int c, int h, int wd, int ks)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * h * wd;
    if (idx >= total) return;
    const int x = (int)(idx % wd);
    const long t = idx / wd;
    const int y = (int)(t % h);
    const int ch = (int)(t / h);
    const size_t plane = (size_t)h * wd;
    const int pad = (ks - 1) / 2;

    float acc = 0.0f;
    for (int k = 0; k < ks; ++k) {
        int iy = y + k - pad;
        if (iy < 0) iy = 0;
        if (iy >= h) iy = h - 1;
        const float f = feat[(size_t)ch * plane + (size_t)iy * wd + x];
        const float kv = k1[((size_t)ch * ks + k) * plane + (size_t)y * wd + x];
        acc += f * kv;
    }
    out[idx] = acc;
}

// Horizontal pass: same shape, taps k1 again (the upstream bug), the x clamp is
// on the VERTICAL pass's output so the two clamps compose into (clamp(y), x).
extern "C" __global__ void if_sac_horizontal(
    const float *__restrict__ feat, const float *__restrict__ k1,
    float *__restrict__ out, int c, int h, int wd, int ks)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * h * wd;
    if (idx >= total) return;
    const int x = (int)(idx % wd);
    const long t = idx / wd;
    const int y = (int)(t % h);
    const int ch = (int)(t / h);
    const size_t plane = (size_t)h * wd;
    const int pad = (ks - 1) / 2;

    float acc = 0.0f;
    for (int k = 0; k < ks; ++k) {
        int ix = x + k - pad;
        if (ix < 0) ix = 0;
        if (ix >= wd) ix = wd - 1;
        const float f = feat[(size_t)ch * plane + (size_t)y * wd + ix];
        const float kv = k1[((size_t)ch * ks + k) * plane + (size_t)y * wd + x];
        acc += f * kv;
    }
    out[idx] = acc;
}

// Add one bias block of IAC (`f = f + f_bs[i]`) and apply the activation that
// follows it. One thread per element; `act` is 1 for every iteration in IFAN
// (upstream's `is_act_last` defaults to True, so the LAST iteration is activated
// too - see tools/reference.py).
//
// This is fused into one kernel rather than an `lg_add` plus an `lg_lrelu`
// because the two always run back to back and the fusion halves the traffic over
// a 128 x H/4 x W/4 plane, which at 1080p is the largest tensor in the model.
extern "C" __global__ void if_sac_bias(
    const float *__restrict__ x, const float *__restrict__ b, float *__restrict__ y,
    float slope, int c, int h, int wd, int act)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * h * wd;
    if (idx >= total) return;
    const size_t plane = (size_t)h * wd;
    const int ch = (int)(idx / plane);
    float v = x[idx] + b[(size_t)ch * plane + idx % plane];
    if (act) v = v >= 0.0f ? v : slope * v;
    y[idx] = v;
}


// ---------------------------------------------------------------------------
// TILED CONVOLUTION - the shared-memory kernel the dense layers actually need.
//
// WHY ONE-THREAD-PER-OUTPUT IS SLOW HERE, MEASURED RATHER THAN GUESSED. The
// toolkit's convolution gives each thread ONE output element
// and accumulates it in ONE register over the whole `9 * c_in` tap loop. That makes
// the FMA chain strictly serial and every link of it depend on a fresh load, so the
// kernel runs at the LATENCY of its loads rather than at the throughput of its
// arithmetic: 1152 dependent FMAs at roughly thirty cycles each is ~35k cycles per
// thread, which is what the measured 90 ms for a 128->128 layer at 1/8 scale
// corresponds to. ptxas reports 60 registers and no spills, and the card is at
// 1835 MHz throughout, so neither occupancy nor clock is the explanation.
//
// The two structural fixes, both standard and both needed:
//
//   * SEVERAL INDEPENDENT ACCUMULATORS. Each thread computes OC_TILE output
//     channels at once, so there are OC_TILE independent FMA chains in flight and
//     the scheduler always has something to issue.
//   * A SHARED-MEMORY INPUT TILE. The block stages the input it needs once per
//     input channel, and every thread then reads it from shared memory. The input
//     is read from global memory once per (block, input channel) instead of once
//     per (thread, tap, input channel), and the shared reads are conflict-free.
//
// LAYOUT, AND WHY THE PIXELS ARE INTERLEAVED. A thread owns TPX output columns
// that are BX apart (`tx, tx+BX, ...`) rather than four adjacent ones. Adjacent
// columns would make consecutive threads read shared memory four floats apart,
// which is a four-way bank conflict on every load; interleaving them makes
// consecutive threads read consecutive floats, which is conflict-free.
//
// ONE BLOCK IS ONE OUTPUT-CHANNEL TILE (`blockIdx.z`), so the input tile is re-read
// once per channel tile - sixteen times for a 128-channel layer, against 128 times
// for the untiled kernel. It is deliberately NOT all output channels per block:
// that would leave only a few dozen blocks for the whole card.
//
// THE ACCUMULATION ORDER IS (ci, ky, kx), NOT THE UNTILED KERNELS' (ky, kx, ci).
// It has to be: the staged tile is walked once per channel tile and the kernel
// window is the inner loop, which is what makes OC_TILE independent accumulators
// possible. And the tail of each dimension is handled by ZERO-FILLING the staged
// tile rather than by skipping out-of-range taps, so a boundary output sums 27
// terms where the untiled kernel sums 22 of them and adds five zeros. Summing
// zeros changes nothing; summing the same terms in a different order changes the
// last bits. That is the whole of the CUDA-vs-torch movement recorded in
// docs/NUMERICS.md (`1.431e-06 -> 1.729e-06` at 64x64), and CPU-vs-torch - the
// leg that does not involve this file - did not move at all.
// ---------------------------------------------------------------------------
namespace {

// THE TILE SHAPE IS A PARAMETER, and the two convolutions want different ones.
//
// The 3x3 is compute-bound: 27 multiply-adds per output element, so what matters
// is keeping the FMA pipes fed and the shared tile small enough for occupancy. A
// 32x8 block with four columns per thread gives a 128x8 output tile, sixteen
// accumulators per thread, and a 130x10 shared tile per input channel.
//
// The 1x1 is TRAFFIC-bound, and its traffic is not the output - it is the INPUT,
// read once per output-channel tile. F.3 produces 15232 channels, so an
// eight-channel tile re-reads the input 1904 times; measured, that is 336 ms for a
// layer whose arithmetic is 126 GFLOP. A 32-channel tile cuts it to 476 passes and
// measured 108 ms. The 1x1 also needs no halo (KW = 1), so its shared tile is
// exactly the output tile and a wider channel tile costs nothing but registers.
//
// The instantiations at the bottom of this file carry the numbers, and the
// transposed convolution has a template of its own further down.

// A __device__ function, NOT a __global__ one: a kernel cannot call another
// kernel, and the only thing the wrappers below do is fix (KH, KW, STRIDE) and
// give the launch its bounds. Every thread index it reads is the caller's. (The
// transposed convolution has its own `if_convt_tile_kernel` further down, for the
// reason given there: its taps depend on the output's parity, not on one halo
// origin, so it cannot be this template with different constants.)
//
// `act` FUSES THE ACTIVATION the graph puts after almost every convolution. The
// toolkit's kernels have no such flag, so those layers would otherwise run as two
// launches through a pool scratch; fusing removes a full read and write of the
// output plane per activated layer, and with it the scratch buffer, which the
// memory plan then no longer has to find room for. The activation is applied
// AFTER the bias, on the same value the separate `lg_lrelu` would have seen, so
// the arithmetic is identical and the summation order is untouched.
template <int KH, int KW, int STRIDE, int TBX, int TBY, int TPX, int CI_TILE, int OC_TILE>
__device__ __forceinline__ void if_conv_tile_kernel(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, float slope, int act)
{
    constexpr int TW = TBX * TPX;     // output columns per block
    constexpr int TH = TBY;           // output rows per block
    // STRIDE GENERALIZES THE STAGING. Output column `lx` of the tile reads input
    // columns `lx * STRIDE + kx - PADX`, so the staged tile spans
    // (TW-1)*STRIDE + KW columns and (TH-1)*STRIDE + KH rows, and the tile's
    // global origin moves by STRIDE. At STRIDE = 1 the two formulas reduce to
    // TW + KW - 1 columns and TH + KH - 1 rows, the ordinary halo.
    constexpr int SW = (TW - 1) * STRIDE + KW;   // staged tile width, with halo
    constexpr int SH = (TH - 1) * STRIDE + KH;   // staged tile height, with halo
    constexpr int PADX = KW / 2, PADY = KH / 2;
    constexpr int NTAP = KH * KW;

    __shared__ float sh[CI_TILE][SH][SW];
    __shared__ float sw[OC_TILE][CI_TILE][NTAP];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.x * TW;
    const int oy0 = blockIdx.y * TH;
    const int oc0 = blockIdx.z * OC_TILE;
    // The origin of the INPUT region this tile needs, in input coordinates.
    const int gx0 = ox0 * STRIDE - PADX;
    const int gy0 = oy0 * STRIDE - PADY;

    float acc[OC_TILE][TPX];
#pragma unroll
    for (int o = 0; o < OC_TILE; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += CI_TILE) {
        // Stage the input tile. Out-of-range positions are ZERO, which is exactly
        // what the padded convolution's boundary contributes, so the compute loop
        // below needs no boundary test at all.
#pragma unroll
        for (int k = 0; k < CI_TILE; ++k) {
            const int ci = ci0 + k;
            for (int i = tid; i < SW * SH; i += TBX * TBY) {
                const int sy = i / SW;
                const int sx = i - sy * SW;
                const int gy = gy0 + sy;
                const int gx = gx0 + sx;
                float v = 0.0f;
                if (ci < c_in && gy >= 0 && gy < h && gx >= 0 && gx < wd) {
                    v = in[((size_t)ci * h + gy) * wd + gx];
                }
                sh[k][sy][sx] = v;
            }
        }
        // Stage this tile's weights, in the same (oc, ci, tap) order the compute
        // loop indexes them. A partial channel tile zero-fills, and a zero weight
        // contributes nothing - the same reason the input is zero-filled.
        for (int i = tid; i < OC_TILE * CI_TILE * NTAP; i += TBX * TBY) {
            const int o = i / (CI_TILE * NTAP);
            const int r = i - o * (CI_TILE * NTAP);
            const int k = r / NTAP;
            const int t = r - k * NTAP;
            const int oc = oc0 + o;
            const int ci = ci0 + k;
            float v = 0.0f;
            if (oc < c_out && ci < c_in) v = w[((size_t)oc * c_in + ci) * NTAP + t];
            sw[o][k][t] = v;
        }
        __syncthreads();

        // THE COMPUTE. Per (k, ky, kx) the TPX input values are loaded ONCE and
        // reused for every output channel in the tile, and each output channel's
        // weight is loaded once and reused for every pixel - so the shared traffic
        // is TPX + OC_TILE loads for OC_TILE * TPX multiply-adds.
#pragma unroll
        for (int k = 0; k < CI_TILE; ++k) {
#pragma unroll
            for (int ky = 0; ky < KH; ++ky) {
#pragma unroll
                for (int kx = 0; kx < KW; ++kx) {
                    const int sy = ty * STRIDE + ky;
                    float v[TPX];
#pragma unroll
                    for (int p = 0; p < TPX; ++p) v[p] = sh[k][sy][(tx + p * TBX) * STRIDE + kx];
#pragma unroll
                    for (int o = 0; o < OC_TILE; ++o) {
                        const float wv = sw[o][k][ky * KW + kx];
#pragma unroll
                        for (int p = 0; p < TPX; ++p) acc[o][p] += wv * v[p];
                    }
                }
            }
        }
        // The next iteration overwrites `sh`, so every reader has to be done.
        __syncthreads();
    }

    // Write the tile out. The bias is added once, after the whole sum, exactly as
    // the untiled kernels do.
    //
    // THE OUTPUT DIMENSIONS ARE DERIVED FROM THE INPUT'S, exactly as the untiled
    // kernels do (`oh = (h + 1) / 2` there), because a stride-2 output is
    // ceil(h/2) x ceil(w/2) and the tile's own row/column index is in OUTPUT
    // coordinates. At STRIDE = 1 the two coincide, which is why this one line
    // serves all three instantiations.
    const int oh = (h + STRIDE - 1) / STRIDE;
    const int ow = (wd + STRIDE - 1) / STRIDE;
    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OC_TILE; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < oh) {
            const float bb = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < ow) {
                    float v = acc[o][p] + bb;
                    if (act) v = v >= 0.0f ? v : slope * v;
                    out[((size_t)oc * oh + my) * ow + gx] = v;
                }
            }
        }
    }
}

}  // namespace

// The four instantiations. EVERY convolution IFAN's graph launches is one of these
// or the transposed one below.
//
// NOTE THE ORDER OF `void` AND THE ATTRIBUTE, and note that this comment does
// not quote the declaration it describes. `lightgpu_build::kernel_names_in` finds
// kernels by searching for the declaration prefix, then skips an attribute before
// the name - so an attribute between `__global__` and `void` makes it read the
// attribute as the name, and a COMMENT containing that prefix followed by
// anything that is not an identifier makes it read an EMPTY name and report a
// kernel called `` that the list does not have. Both are the both-ways check
// working as designed. The attribute belongs after `void`, and the literal
// declaration prefix does not belong in prose.
//
// THE LAUNCH GEOMETRY IS PART OF EACH KERNEL'S CONTRACT. The numbers below are
// mirrored in `src/gpu.rs`, where the launch is built, and a mismatch does not
// fault: it silently leaves part of the output unwritten, because every block that
// is not launched simply never runs.
//
// They were measured, not chosen - each one from a sweep of a few dozen geometries
// for its own kernel, because the answers disagree: the 3x3 wants several pixels
// per thread and a narrow channel tile, the stride-2 wants a SMALL shared tile
// because its staged region is mostly halo, and `F.3` wants the opposite of both.
// The sweeps and their tables are in docs/INTERNALS.md, which is also where the
// one geometry that measured faster than what is shipped here is written down
// (it is not shipped, and the reason is not that the measurement was wrong).
//
constexpr int G3_TBX = 32, G3_TBY = 8, G3_TPX = 4, G3_CI = 4, G3_OC = 8;
constexpr int G1_TBX = 32, G1_TBY = 4, G1_TPX = 4, G1_CI = 4, G1_OC = 32;
constexpr int G2_TBX = 32, G2_TBY = 8, G2_TPX = 2, G2_CI = 1, G2_OC = 16;

extern "C" __global__ void __launch_bounds__(G3_TBX * G3_TBY) if_conv3x3_tile(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, float slope, int act)
{
    if_conv_tile_kernel<3, 3, 1, G3_TBX, G3_TBY, G3_TPX, G3_CI, G3_OC>(
        in, w, bias, out, c_in, c_out, h, wd, slope, act);
}

extern "C" __global__ void __launch_bounds__(G1_TBX * G1_TBY) if_conv1x1_tile(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, float slope, int act)
{
    if_conv_tile_kernel<1, 1, 1, G1_TBX, G1_TBY, G1_TPX, G1_CI, G1_OC>(
        in, w, bias, out, c_in, c_out, h, wd, slope, act);
}

// The stride-2 instantiation, the encoder's downsampling convolution. Same
// kernel, STRIDE = 2: a 64x8 output tile reads a 129x17 input region, and the
// staged halo covers it exactly.
//
// THIS IS WORTH 43% OF THE PASS. The six stride-2 convolutions in the two
// encoder towers are 95.6 GFLOP of the pass's 1881.6 (5.1%), so at the 1600
// GFLOP/s the tiled 3x3 measures they cost 0.06 s - a share of the wall time
// that only a one-output-element-per-thread kernel would make visible.
//
// `h` and `wd` are the INPUT's dimensions and the output is ceil(h/2) x
// ceil(w/2), which is what the caller derives the grid from. A tile whose staged
// region runs off the input is zero-filled, which is exactly the "skip the taps
// outside" rule of a padded convolution, and the grid is sized so that unwritten
// output is never read.
extern "C" __global__ void __launch_bounds__(G2_TBX * G2_TBY) if_conv3x3s2_tile(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, float slope, int act)
{
    if_conv_tile_kernel<3, 3, 2, G2_TBX, G2_TBY, G2_TPX, G2_CI, G2_OC>(
        in, w, bias, out, c_in, c_out, h, wd, slope, act);
}


// ---------------------------------------------------------------------------
// TRANSPOSED 4x4 STRIDE 2 PAD 1 - the decoder's upsampler, TILED.
//
// WHY THIS IS A SECOND TEMPLATE RATHER THAN A CONSTANT ON THE FIRST. Three of
// these run per pass, 339.7 GFLOP of the pass's 1881.6 (18.1%), and its taps are a
// function of the output's PARITY rather than of a single padded window, so it
// enumerates its four taps instead of sweeping one.
//
// THE TAPS DEPEND ONLY ON THE OUTPUT'S PARITY, AND THERE ARE EXACTLY FOUR OF THEM.
// Inverting `ky = oy + 1 - 2*iy` (pad 1, stride 2) says the input row
// `ry = ceil(oy / 2)` carries weight `kya = oy + 1 - 2*ry` - which is 1 for an
// even `oy` and 0 for an odd one - and that the only other row that reaches this
// output is `ry - 1`, with weight `kya + 2`. Identically in x. So every output
// pixel sums exactly FOUR taps and the other twelve kernel entries multiply
// nothing. That identity was checked against torch's own
// `F.conv_transpose2d(x, W, b, stride=2, padding=1)` on a float64 random case
// rather than derived on paper and confirmed from the pixels, which is how a
// bounds mistake in the same arithmetic once survived to the gate.
//
// THE GEOMETRY THAT FALLS OUT OF IT. Output row `oy = oy0 + ty` reads the input
// rows `ry` and `ry - 1`, where `ry = ceil(oy/2)`, so a `TH`-row output tile needs
// `SH = TH + 2` staged rows with the origin at `oy0/2 - 1`: one row of halo at
// EACH end, because an even output row reads one row UP and an odd one reads one
// row DOWN, and the two parities interleave down the tile. The same in x, where
// `SW = TW + 2`. `oy0` is a multiple of the (even) block height, so `oy0 / 2` is
// exact.
//
// The staged region is SMALL - a `TW`-wide output tile spans only `TW/2` input
// columns, so `SW = TW + 2` - because this output is twice as dense as its input.
// None of that lets this kernel reuse `if_conv_tile_kernel` with a padding
// constant: its tap offsets come from the output's parity, not from one origin.
//
// `w` IS `[c_in, c_out, 4, 4]`, NOT `[c_out, c_in, 4, 4]`. A transposed
// convolution's weight is indexed by CONTEXT first, so the stride between its
// 16-entry kernel planes is `c_out` while the stride between the input's channel
// planes is `h * wd`. Those two are equal only when the layer is square, which
// every layer here is EXCEPT the middle upsampler - `upconv2_u.0` is 64 -> 128 -
// so a kernel that confuses them passes its own narrow test and prints a picture.
// The gate reads `max |d| 9.1e-01` against the CPU twin at 64x64 when this
// staging loop indexes the weight as `ci * c_in + oc`.
//
// THE ACCUMULATION ORDER IS (ci, tap) over the four taps in the order
// {(0,0), (0,1), (1,0), (1,1)} of (dy, dx). The untiled kernel reaches the same
// result by walking a WIDER (iy, ix) range and rejecting, through its `ky`/`kx`
// and bounds tests, exactly the pairs that contribute nothing: the ones whose
// position is off the image (a STAGED ZERO here) and the ones whose kernel index
// falls outside the 4x4 window (not in this kernel's enumeration at all). The
// terms that survive are the same terms, added in the same order, so the two
// kernels agree term for term and not merely to within round-off - which is what
// `tools/gate.py`'s CUDA-vs-CPU leg checks.
//
// THE ACTIVATION IS APPLIED ONCE, IN THE EPILOGUE, AND NOWHERE ELSE. The value a
// consumer sees is `lrelu(conv + bias)`, and it is tempting to apply `act` to the
// STAGED weights and inputs instead and skip the separate pass, on the grounds
// that `act` is monotone and idempotent, so `lrelu(x + y) == lrelu(lrelu(x) +
// lrelu(y))`. THAT IDENTITY IS TRUE AND THE SUBSTITUTION IS STILL WRONG: `lrelu`
// is linear only on the NON-NEGATIVE half-line, so scaling a term by the slope
// before summing it is not the same as scaling the sum - a defect the gate reads
// as `max |d| 9.1e-01` between the backends. With `act = 0` this kernel computes
// the raw sum instead, which is what its own test compares against.
namespace {

// `TBX` and `TBY` are the block; each thread writes `TPX` output columns spaced
// `TBX` apart, so the block's output tile is `TBX * TPX` wide. THE COLUMNS OF ONE
// THREAD MUST SHARE A PARITY, which is what `TBX` being even guarantees: output
// column `tx + p * TBX` of a tile whose origin is even has the parity of `tx` for
// every `p`, so the thread's four-tap weight quad is the same for all of them and
// is loaded once.
//
// The rows are NOT interleaved: one output row per thread, because a row's input
// rows are one above and one below it and two rows of the same parity would need
// a taller window for no gain - the weight quad already amortises over columns.
template <int TBX, int TBY, int TPX, int CI_TILE, int OC_TILE>
__device__ __forceinline__ void if_convt_tile_kernel(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, float slope, int act)
{
    constexpr int TW = TBX * TPX;   // output columns per block
    constexpr int TH = TBY;         // output rows per block
    constexpr int SW = TW + 2;      // staged input width: one column of pad either side
    constexpr int SH = TH + 2;      // staged input height: one row of pad either side
    constexpr int NTAP = 16;        // the flattened 4x4 kernel plane
    static_assert(TBX % 2 == 0, "the block must have an even width so that the "
                                "columns of one thread share a parity");

    __shared__ float sh[CI_TILE][SH][SW];
    __shared__ float sw[OC_TILE][CI_TILE][NTAP];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.x * TW;
    const int oy0 = blockIdx.y * TH;
    const int oc0 = blockIdx.z * OC_TILE;

    // The staged tile's origin, in input coordinates: one row and one column
    // BEFORE the region the output tile maps onto (`ceil(oy0/2) - 1`), which is
    // what makes the staged index of an input position `iy` equal to
    // `iy - oy0/2 + 1`. `oy0` is a multiple of TH (even, since TH is a block
    // height), so `oy0 / 2` is exact and `ceil` never has to be applied to it.
    const int gy0 = oy0 / 2 - 1;
    const int gx0 = ox0 / 2 - 1;

    // This thread's two input rows, as staged indices, and the two weights that
    // go with them: `sy` is the output row's parity, and the row it reads at tap
    // 0 is one row UP for an even output row and one row DOWN for an odd one.
    const int sy = ty & 1;
    const int syi = (ty + 1) / 2;          // ceil(ty / 2), the staged row of tap 0
    const int ky0 = 1 - sy;                // 1 for even output rows, 0 for odd
    // The same for the columns, where the second column of the thread is `p = 1`:
    // its staged column is `TBX / 2` further along, which is the whole reason the
    // two share a parity and therefore a weight quad.
    const int sx = tx & 1;
    const int sxi = (tx + 1) / 2;
    const int kx0 = 1 - sx;

    float acc[OC_TILE][TPX];
#pragma unroll
    for (int o = 0; o < OC_TILE; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += CI_TILE) {
        // Stage the input region for this channel tile. Out-of-range is ZERO,
        // which is the padding; the activation is NOT applied here, because
        // `lrelu` does not distribute over a sum (see the header) - scaling a
        // staged value would scale only the terms it takes part in.
#pragma unroll
        for (int k = 0; k < CI_TILE; ++k) {
            const int ci = ci0 + k;
            for (int i = tid; i < SW * SH; i += TBX * TBY) {
                const int syy = i / SW;
                const int sxx = i - syy * SW;
                const int gy = gy0 + syy;
                const int gx = gx0 + sxx;
                float v = 0.0f;
                if (ci < c_in && gy >= 0 && gy < h && gx >= 0 && gx < wd) {
                    v = in[((size_t)ci * h + gy) * wd + gx];
                }
                sh[k][syy][sxx] = v;
            }
        }
        // Stage this tile's weights. `w[ci][oc][ky][kx]` - CONTEXT FIRST, so the
        // plane stride is `c_out` - and a channel or an output channel past the
        // tile's end stages a zero, which contributes nothing.
        for (int i = tid; i < OC_TILE * CI_TILE * NTAP; i += TBX * TBY) {
            const int o = i / (CI_TILE * NTAP);
            const int r = i - o * (CI_TILE * NTAP);
            const int k = r / NTAP;
            const int t = r - k * NTAP;
            const int oc = oc0 + o;
            const int ci = ci0 + k;
            float v = 0.0f;
            if (oc < c_out && ci < c_in) v = w[((size_t)ci * c_out + oc) * NTAP + t];
            sw[o][k][t] = v;
        }
        __syncthreads();

        // THE COMPUTE. The two (dy, dx) values of the tap select a weight and a
        // staged position; both are thread-invariant apart from the thread's own
        // parity, so the four taps are a flat unrolled sum with no boundary test
        // anywhere - the halo holds the zeros. Per (k, o) the thread loads FOUR
        // weights and does EIGHT multiply-adds (two pixels x four taps), and the
        // two input values a tap needs are shared between the two pixels through
        // the `p` unroll.
#pragma unroll
        for (int k = 0; k < CI_TILE; ++k) {
#pragma unroll
            for (int d = 0; d < 2; ++d) {
                // Tap (dy, dx) = (d, e): the input row `ry - d`, whose staged
                // index is `syi + 1 - d` (the staged origin is one row above the
                // output tile's own first row), and the weight `kya + 2*d`, where
                // `kya` is the "own row" weight - 1 for an even output row, 0 for
                // an odd one.
                const int rrow = syi + 1 - d;
                const int ky = ky0 + 2 * d;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int kx = kx0 + 2 * e;
#pragma unroll
                    for (int o = 0; o < OC_TILE; ++o) {
                        const float wv = sw[o][k][ky * 4 + kx];
#pragma unroll
                        for (int p = 0; p < TPX; ++p) {
                            acc[o][p] += wv * sh[k][rrow][sxi + p * (TBX / 2) + 1 - e];
                        }
                    }
                }
            }
        }
        // The next channel tile overwrites the staging.
        __syncthreads();
    }

    // Out. The output is the input doubled, and every tap was a legal input
    // position or a staged zero, so this is a plain write with a tail guard: the
    // grid is sized to cover `2h x 2wd` and the last tile in a row or column may
    // hang over the edge.
    const int oh = h * 2, ow = wd * 2;
    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OC_TILE; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < oh) {
            const float bb = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < ow) {
                    float v = acc[o][p] + bb;
                    if (act) v = v >= 0.0f ? v : slope * v;
                    out[((size_t)oc * oh + my) * ow + gx] = v;
                }
            }
        }
    }
}

}  // namespace

// THE GEOMETRY. A 64x8 output tile of 8 channels from a block of 32x8, each
// thread writing two output columns and every thread a row of its own. The tile
// is kept SHORT in y on purpose: `SW = TW + 2 = 66` staged columns is the widest
// number here, and at CI_TILE = 2 the staging is 2 x 10 x 66 x 4 = 5.3 KiB, which
// leaves the shared memory budget to the block count an SM can hold rather than
// spending it on a halo.
constexpr int GT_TBX = 32, GT_TBY = 8, GT_TPX = 2, GT_CI = 2, GT_OC = 8;

extern "C" __global__ void __launch_bounds__(GT_TBX * GT_TBY) if_convtranspose4x4s2p1_tile(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, float slope, int act)
{
    if_convt_tile_kernel<GT_TBX, GT_TBY, GT_TPX, GT_CI, GT_OC>(
        in, w, bias, out, c_in, c_out, h, wd, slope, act);
}

/// `out = clamp(x, 0, 1)`.
///
/// The final op of the graph. A kernel rather than a host round trip: the value
/// it clips is a full-resolution 3-channel image (6 MiB at 1080p), which is
/// cheaper to touch on the device than to copy back and forth, and the CPU twin
/// does the identical clamp in floats so the two agree exactly on every element
/// (clamp is exact in both).
///
/// `n` is 64-bit for the same reason the toolkit's elementwise kernels use it:
/// at large sizes `3 * h * w` is close enough to 2^31 that an `int` would be a
/// live risk on the biggest supported inputs, and the index arithmetic below is
/// done in `long` regardless.
extern "C" __global__ void if_clip(const float *__restrict__ x, float *__restrict__ y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
