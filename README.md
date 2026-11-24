# foundationstereo-rknn

[![CI](https://github.com/jlu-lujing/fast-foundationstereo-rknn-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/jlu-lujing/fast-foundationstereo-rknn-rust/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rust inference library for [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
on Rockchip RK3588.

The library wraps Rockchip's RKNN C API directly and exposes a safe,
tensor-oriented Rust API. It supports both the segmented graph path and the
faster single-RKNN path:

- `FoundationStereo`: feature encoder and segmented cost aggregation on the NPU,
  group-wise correlation volume in Rust, and Resize/refinement in ONNX Runtime.
- `SingleGraphStereo`: the whole network in one `192x384` RKNN graph, with two
  FP16 NHWC inputs passed directly to all three NPU cores and one FP32
  disparity output.

## Status

- RKNN runtime loaded dynamically; no Rockchip SDK required at compile time.
- Multiple float32 inputs and outputs, including rank-5 tensors.
- Direct FP16 NHWC pass-through for low-overhead single-graph inference.
- Per-model NPU core selection for `feature`, `pre-resize`, and `post-resize`.
- Group-wise correlation volume through a vendored AArch64 SIMD/NEON kernel,
  with a pure Rust scalar reference used by tests.
- ONNX Runtime dynamic loading through the `ort` crate.
- Synchronous pipeline API and a runnable pair-inspection example.
- Single-graph stereo backend and benchmark example.
- Models and weights are not redistributed.

## Tested platform

- NanoPC-T6-LTS / RK3588
- `librknnrt` 2.3.2 for segmented graphs; 2.5.1 for the custom-op single graph
- ONNX Runtime 1.19.2
- `192x384_it2_d96` single graph and `192x384_it2_d96` / `it4` segmented artifacts

The Python reference pipeline reaches 695 ms/frame with the 2-iteration profile
under the board's shipped CPU limit through cross-frame overlap. The current Rust
API is synchronous and measured **1423.7 ms/frame**; a threaded streaming
scheduler is the next milestone.

That number applies to the segmented Rust path. A compatible whole-network RKNN
graph removes the repeated RKNN launches and the CPU/ONNX Runtime stage. On two
real rectified pairs, the `192x384`, 2-iteration, `max_disp=96` single graph
measures **227.5-234.7 ms/frame (4.26-4.40 FPS)** with prepared FP16 inputs and
**235.3-239.9 ms/frame** including RGB-to-FP16 packing. The C++ demo using the
same graph and runtime measures 236.0-236.4 ms, so the Rust wrapper has no
measurable inference-path overhead.

The public K3588 exporter reports 197.0 ms / 5.08 FPS for this configuration.
That figure was not reproduced here: the exporter does not distribute its exact
model artifact, and its own C++ demo against a locally exported official
checkpoint measures 236 ms. Treat 227-236 ms (4.2-4.4 FPS) as the current
independently verified result.

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

For a deployment that uses only `SingleGraphStereo`, disable the optional ONNX
Runtime backend and its dependency tree:

```bash
cargo build --release --no-default-features --example bench_single_graph
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

For a compatible whole-network model, prepare a rectified `384x192` RGB pair and
run:

```bash
cargo run --release --example bench_single_graph -- \
  --model /path/to/stereo.rknn \
  --rknn-lib /path/to/librknnrt.so \
  --left left.png \
  --right right.png \
  --warmup 2 \
  --frames 30
```

The benchmark reports both `model` and `rgb8` timings. `model` reuses prepared
`F16Image` inputs; `rgb8` includes the per-frame RGB-to-FP16 packing that a live
camera pipeline needs.

For the segmented path:

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

Single-graph inference does not require ONNX Runtime:

```rust,no_run
use foundationstereo_rknn::{NpuCore, RknnOptions, SingleGraphStereo};

let options = RknnOptions {
    library_path: Some("/usr/lib/librknnrt.so".into()),
    core: NpuCore::All,
};
let mut stereo = SingleGraphStereo::load("/path/to/stereo.rknn", options)?;
let disparity = stereo.infer_rgb8(&left_rgb_384x192, &right_rgb_384x192)?;
assert_eq!(disparity.shape, vec![1, 1, 192, 384]);
# Ok::<(), foundationstereo_rknn::Error>(())
```

## Current limits

- Synchronous invocation only; callers must schedule adjacent frames themselves.
- `SingleGraphStereo` expects the fixed single-RKNN layout and two FP16 NHWC RGB
  inputs. Its model is not redistributed.
- `segment_refine_ffs.onnx` requires the compatible FFS custom-op library and
  ONNX Runtime 1.19.x.
- The C correlation kernel is optimized for AArch64; other hosts use the Rust
  scalar reference.
- No pretrained weights or `.rknn` files are distributed.

## License

This library is MIT licensed. NVIDIA's upstream model weights and derivatives
remain subject to their own research/non-commercial terms. No model weights are
included in this repository.

