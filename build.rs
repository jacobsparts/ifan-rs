//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/ifan.cu` (PROJECT_KERNELS). Each compiles to its own fatbin with its
//! own `--entries` list and `src/cuda.rs` loads them as separate modules, so
//! neither can shadow a name in the other.
//!
//! A kernel missing from its list is PRUNED from the fatbin and fails at launch
//! rather than at build time, so both lists are checked against the source they
//! are compiled from before nvcc runs: a typo, or a kernel moved between files,
//! fails the build.

/// Generic ops from the shared toolkit. Exactly the ones `exec_gpu` launches:
/// an entry here is a kernel embedded in the binary, so a name that no op calls
/// is wasted bytes.
const TOOLKIT_KERNELS: &[&str] = &[
    // NO CONVOLUTIONS. Every convolution in IFAN now goes through this project's
    // own `if_conv3x3_tile` / `if_conv1x1_tile` (or its stride-2 and transposed
    // kernels), because the toolkit's versions give each thread ONE output element
    // accumulated in ONE register - a strictly serial FMA chain behind a fresh load
    // every time - and measured 88 to 226 GFLOP/s against the tiled kernels' 800 to
    // 2000. `lg_conv3x3s1p1` and `lg_conv1x1` therefore stay in the toolkit for
    // other engines but are not embedded here: an entry in this list is a kernel in
    // the binary, so keeping a name no op launches is wasted bytes, and this list is
    // checked against what `exec_gpu` actually calls by construction.
    //
    // LeakyReLU(0.1), the only activation in the model that is NOT fused into a
    // convolution (the convolutions carry their own `act` flag now): inside a
    // ResnetBlock where the input is added first, and the standalone op.
    "lg_lrelu",
    // The residual adds (`x + stem[i](x)`, `x + temp`, `x + f3/f2/f1`) and the
    // final `out_res(x) + c_in`.
    "lg_add",
    // Plain device-to-device copy, used to seed the IAC iteration input and to
    // move a band edge into place.
    "lg_copy",
];

/// This project's own kernels, in `cuda/ifan.cu`. Three of these are ops the
/// toolkit does not have at all: a STRIDED convolution and a TRANSPOSED
/// convolution (the toolkit has neither - `lg_conv4x4s4` is a stride-4 patch
/// embed and `lg_upsample2x_nearest` has no learned kernel), and the SEPARABLE
/// ADAPTIVE CONVOLUTION tap passes (a 1-D vertical and a 1-D horizontal pass
/// whose weights vary per position, which no fixed-kernel op can express).
const PROJECT_KERNELS: &[&str] = &[
    "if_conv1x1_tile",
    "if_conv3x3s2_tile",
    "if_conv3x3_tile",
    "if_convtranspose4x4s2p1_tile",
    "if_sac_vertical",
    "if_sac_horizontal",
    "if_sac_bias",
    // The final clamp. The toolkit has no clip and the alternative was a host
    // round trip of a full-resolution image.
    "if_clip",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/ifan.cu");

    // `cargo build --no-default-features` is the pure-Rust CPU build: it must
    // not need nvcc, and `src/cuda.rs` (which includes the fatbins) is not
    // compiled at all, so the env vars it would embed are not needed.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/ifan.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/ifan.cu").expect("read cuda/ifan.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/ifan.cu (it has {})",
            defined.join(", ")
        );
    }
    // The other direction matters just as much: a kernel defined but NOT listed
    // is pruned from the fatbin by `--entries`, and then it fails at LAUNCH
    // rather than at build time. Checking both directions makes the list and the
    // source agree rather than merely overlap.
    for d in &defined {
        assert!(
            PROJECT_KERNELS.contains(&d.as_str()),
            "cuda/ifan.cu defines `{d}`, which PROJECT_KERNELS does not list - \
             it would be pruned from the fatbin and fail at launch"
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "ifan_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/ifan.cu",
            out_name: "ifan_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
