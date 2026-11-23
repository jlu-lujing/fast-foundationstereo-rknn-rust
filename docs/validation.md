# RK3588 validation

Date: 2026-11-24

## Platform

- NanoPC-T6-LTS, RK3588
- `librknnrt` 2.3.2
- ONNX Runtime 1.19.2
- Rust 1.100.0, release build
- `192x384_it2_d96` model preset
- Original Fast-FoundationStereo `192x384_it2_d96` exported stereo pair

The shipped CPU frequency limits were left enabled.

## ABI

The Rockchip 2.3.2 header defines `rknn_context` as `uint64_t` on 64-bit
platforms. `rknn_input.type` and `rknn_input.fmt` are 32-bit enums. The crate uses
that native ABI directly.

## Stage agreement with Python/RKNNLite

All left and right feature outputs were bit-identical to RKNNLite. The SIMD
correlation volume differed only by floating-point accumulation order.

| Stage | MAE vs Python | Max abs diff | Cosine |
|---|---:|---:|---:|
| feature outputs, left/right | 0 | 0 | 1.0 |
| GWC volume | 1.34e-8 | 1.19e-7 | 1.0 |
| final disparity, Rust vs Python FFS | **8.34e-6 px** | **2.40e-4 px** | **0.9999999999992** |

On pixels visible in both cameras, Rust versus the original FP32 export has MAE
`0.00772 px` and P99 `0.0451 px`.

## Synchronous throughput

Ten measured frames after two warmups:

```text
median = 1423.7 ms
p95    = 1433.9 ms
fps    = 0.702
```

This is the sequential Rust path. The Python reference reaches about 695 ms
through a separate four-worker cross-frame scheduler. The Rust streaming
scheduler is intentionally left as the next milestone rather than hiding the
difference in the benchmark.

## Build result

The crate builds and its unit tests pass natively on the RK3588:

```text
5 passed; 0 failed
```

The tests cover NCHW-to-NHWC packing, rank-5 pass-through, Rust scalar GWC, Rust
NEON GWC, and the vendored SIMD GWC against the scalar reference.

