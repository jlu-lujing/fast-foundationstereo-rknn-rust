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

## Output agreement

On both real rectified pairs and two Blender scenes, the Rust library and the
matching C++ RKNN demo produced byte-identical FP32 outputs. Each output is
exactly 192x384 values.

```text
pair 1: Rust == C++, SHA256 match
pair 2: Rust == C++, SHA256 match
Blender near scene 0000: Rust == C++, SHA256 match
Blender far scene 0000: Rust == C++, SHA256 match
```

The Rust API tested both prepared FP16 inputs and RGB-to-FP16 packing. Prepared
inputs isolate the model path; RGB8 timings represent a live camera handoff.

## Accuracy protocol

The original image is first rotated clockwise and stereo-rectified with
`EV202503002`. The crop contains only the intended measurement region. Metrics
are calculated only where:

- the projected point from the left camera remains inside the right camera;
- a forward z-buffer visibility test accepts the point;
- the local patch has nonzero texture;
- SGBM/depth checks additionally require the device SGBM valid mask.

Flat borders and pixels visible in only one camera are excluded.

| Pair | Binocular overlap | Textured overlap | SGBM-comparable | Disparity MAE | P95 | Depth MAE |
|---|---:|---:|---:|---:|---:|---:|
| Pair 1 | 66,836 px | 61,108 px | 28,080 px | 0.582 px | 1.430 px | 0.313 mm |
| Pair 2 | 65,905 px | 64,040 px | 51,466 px | 1.637 px | 2.477 px | 0.694 mm |

The device SGBM result is an independent comparison, not ground truth. Pair 2
has a persistent approximately `-1.605 px` disparity bias against SGBM.

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

Against the previously recorded it2 results, the single graph improves the
near-object suite from `0.198 px` to `0.179 px` MAE. On the far-background suite
it measures `0.160 px` versus `0.154 px` for the former segmented pipeline, while
removing every intermediate transfer and the CPU refinement stage.

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

## Comparison with the previous export

On the same textured binocular overlap, the current official-checkpoint graph
matches the previous self-built export with MAE `0.071 px` / `0.104 px`, P95
`0.195 px` / `0.300 px`, and correlation `0.99975` / `0.99940`. This validates
the conversion path; it is not an absolute accuracy claim because the previous
export is not ground truth.

## Test suite

The repository tests cover:

- NCHW RGB to contiguous NHWC FP16 packing;
- RGB8 length validation;
- tensor shape and element-count validation.

The RKNN calls require hardware and are exercised by `examples/bench_stereo` and
`examples/run_prepared` on the board.
