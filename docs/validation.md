# RK3588 validation

Date: 2026-11-25

## Platform

- NanoPC-T6-LTS, RK3588
- CPU affinity `4-7`, A76 OPP 2.256 GHz
- All CPU policies at their hardware maximum OPP
- NPU 1.0 GHz, all three NPU cores enabled
- RKNN Runtime and Toolkit 2.5.1
- Rust release build
- One `192x384` RKNN graph, `valid_iters=2`, `max_disp=96`, FP16

The board's optional 80% CPU power cap was disabled for this validation. Results
from the capped setting are not used as published performance.

## Accuracy protocol

For comparison with the NVIDIA project, the RKNN model consumes the exact
`feature_inputs.npz` used to produce the project's FP32
`disp_fp32_ref.npy`. Metrics are calculated only in the binocular effective
computation region, where:

- the projected point from the left camera remains inside the right camera;
- a forward z-buffer visibility test accepts the point;
- both outputs are finite.

Flat borders and pixels visible in only one camera are excluded.

| Comparison | Scored pixels | MAE | RMSE | P95 | P99 | Correlation |
|---|---:|---:|---:|---:|---:|---:|
| RKNN vs NVIDIA FP32 | 64,774 | **0.142 px** | 0.358 px | **0.414 px** | 0.895 px | **0.99918** |

Reference SHA256: `0c9d14c773f39ebcaac850c922d15a12dfc5b2109795e879a5203fd13c480479`.
RKNN output SHA256: `311e5dc70ede0e9f50c53de56922de77f1a4b2494d640a6e4f38124b5beac454`.

This verifies consistency with the original NVIDIA project output.

### Blender rendered ground truth

The current Rust API was rerun on 32 Blender stereo pairs through
`FoundationStereo::infer()` with pre-packed FP16 NHWC inputs. Metrics use exact
rendered right-disparity consistency masks, so only mutually visible and
non-occluded pixels are counted.

| Suite | Scenes | Scored pixels | MAE | RMSE | P95 | P99 | Error > 2 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| Near-object | 16 | 747,333 | **0.179 px** | 0.755 px | **0.489 px** | 1.706 px | 0.860% |
| Far textured background | 16 | 1,061,605 | **0.160 px** | 0.637 px | **0.501 px** | 1.460 px | 0.633% |

Flat-region MAE is `0.056 px` and `0.066 px`; edge-region MAE is `0.815 px` and
`0.601 px`. The remaining error is concentrated at depth discontinuities.

## Full-frequency performance

The published result uses full hardware OPPs: A55 `0-3` at 1.8 GHz, A76 `4-7` at
2.256 GHz, and NPU at 1.0 GHz. Each scene was measured in three independent runs
with 10 warmups and 50 frames per run. The values below are median-of-run-medians.

| Scene | Prepared FP16 | Run range | FPS | RGB8 | Run range |
|---|---:|---:|---:|---:|---:|
| Real pair 1 | **206.489 ms** | 200.288-211.345 ms | **4.843** | **208.950 ms** | 202.323-211.212 ms |
| Real pair 2 | **209.160 ms** | 206.955-212.328 ms | **4.781** | **212.607 ms** | 202.159-213.379 ms |

The original 80% CPU limit was restored after testing. It is not used for any
published throughput number.

## Test suite

The repository tests cover:

- NCHW RGB to contiguous NHWC FP16 packing;
- RGB8 length validation;
- tensor shape and element-count validation.

The RKNN calls require hardware and are exercised by `examples/bench_stereo` and
`examples/run_prepared` on the board.
