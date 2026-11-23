# foundationstereo-rknn

[![CI](https://github.com/jlu-lujing/fast-foundationstereo-rknn-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/jlu-lujing/fast-foundationstereo-rknn-rust/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rust inference library for [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
on Rockchip RK3588.

The library wraps Rockchip's RKNN C API directly and exposes a safe,
tensor-oriented Rust API. It runs the feature encoder and segmented cost
aggregation on the NPU, computes the group-wise correlation volume in Rust, and
runs the remaining Resize/refinement ONNX graphs through ONNX Runtime.

## Status

- RKNN runtime loaded dynamically; no Rockchip SDK required at compile time.
- Multiple float32 inputs and outputs, including rank-5 tensors.
- Per-model NPU core selection for `feature`, `pre-resize`, and `post-resize`.
- Group-wise correlation volume through a vendored AArch64 SIMD/NEON kernel,
  with a pure Rust scalar reference used by tests.
- ONNX Runtime dynamic loading through the `ort` crate.
- Synchronous pipeline API and a runnable pair-inspection example.
- Models and weights are not redistributed.

## Tested platform

- NanoPC-T6-LTS / RK3588
- `librknnrt` 2.3.2
- ONNX Runtime 1.19.2
- `192x384_it2_d96` and `192x384_it4_d96` segmented artifacts

The Python reference pipeline reaches 695 ms/frame with the 2-iteration profile
under the board's shipped CPU limit through cross-frame overlap. The current Rust
API is synchronous and measured **1423.7 ms/frame**; a threaded streaming
scheduler is the next milestone.

Validation against the Python segmented pipeline on an original exported stereo
pair produced difference MAE `8.34e-6 px`, P99 `3.05e-5 px`, max
`2.40e-4 px`, and correlation `0.9999999999992`.

## Result comparison

![Original Fast-FoundationStereo input: original FP32 pipeline vs Rust RK3588](assets/readme_comparison.png)

This uses a real stereo pair exported by the original Fast-FoundationStereo
pipeline, not a Blender-generated scene. The difference view is restricted to
pixels visible in both cameras. Against the original FP32 result, Rust has
overlap MAE `0.00772 px` and P99 `0.0451 px` at `192x384`.

## Build

```bash
cargo build --release
cargo test
```

A C compiler is required because the correlation kernel is built and linked by
`build.rs`.

The RKNN runtime is resolved in this order:

1. `RknnOptions::library_path`
2. `RKNN_LIBRARY`
3. `/usr/lib/librknnrt.so`
4. system loader lookup of `librknnrt.so`

ONNX Runtime is supplied by the caller:

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

## Artifacts

The pipeline expects:

```text
feature_single.rknn
segment_pre_resize.rknn
segment_post_resize_full.rknn
stage_resize_only.onnx
segment_refine.onnx
```

If `segment_refine_native_ffs.onnx` or `segment_refine_ffs.onnx` is present, the
library selects it automatically. Register the corresponding FFS operator library
with `--ffs-lib`.

Build them with the Python toolkit repository:

<https://github.com/jlu-lujing/fast-foundationstereo-rknn>

## Example

```bash
cargo run --release --example run_pair -- \
  --artifacts /path/to/artifacts \
  --ort /path/to/libonnxruntime.so \
  --ffs-lib /path/to/libffsops.so \
  --left left.png \
  --right right.png \
  --output disparity.png
```

The two images must be rectified and have the model's exported spatial size.

## API sketch

```rust,no_run
use foundationstereo_rknn::{FoundationStereo, PipelineOptions};

let options = PipelineOptions::new("/path/to/artifacts");
let mut stereo = FoundationStereo::load(options)?;
let disparity = stereo.infer(&left_nchw, &right_nchw)?;
assert_eq!(disparity.shape, vec![1, 1, 192, 384]);
# Ok::<(), foundationstereo_rknn::Error>(())
```

## Current limits

- Synchronous invocation only; callers must schedule adjacent frames themselves.
- `segment_refine_ffs.onnx` requires the compatible FFS custom-op library and
  ONNX Runtime 1.19.x.
- The C correlation kernel is optimized for AArch64; other hosts use the Rust
  scalar reference.
- No pretrained weights or `.rknn` files are distributed.

## License

This library is MIT licensed. NVIDIA's upstream model weights and derivatives
remain subject to their own research/non-commercial terms. No model weights are
included in this repository.

