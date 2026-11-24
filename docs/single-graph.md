# Single-graph RK3588 backend

The segmented path is useful for reproducing individual FFS stages, but it is a
poor throughput architecture. On this board it launches RKNN for the encoder,
pre-resize post stage and post-resize stage, leaves the NPU idle during a CPU
correlation/Resize/refinement interval, and transfers intermediate tensors back
to the CPU.

The single-graph backend keeps the complete two-iteration network resident in
one RK3588 context. The measured path in `wyd-3gi/K3588_Fast-FoundationStereo`
uses this interface:

```text
left  FP16 NHWC [1,192,384,3], RGB 0..255, pass_through=1
right FP16 NHWC [1,192,384,3], RGB 0..255, pass_through=1
disparity FP32 [192,384]
```

The Rust adapter validates those tensor shapes and types, packs NCHW RGB tensors
to NHWC FP16, and normalizes the returned disparity to `[1,1,H,W]`.

## Measured performance

| architecture | model time | effective rate | remaining bottleneck |
| --- | ---: | ---: | --- |
| segmented Rust path, 2 iterations | 1423.7 ms | 0.70 FPS | serial RKNN stages plus CPU/ORT post |
| single RKNN, 2 iterations, d96, scene 1 | 227.5 ms | 4.40 FPS | whole-network NPU execution |
| single RKNN, 2 iterations, d96, scene 2 | 234.7 ms | 4.26 FPS | whole-network NPU execution |
| matching C++ demo | 236.0-236.4 ms | 4.23-4.24 FPS | whole-network NPU execution |

These runs used five warmups, 50 measured frames, CPU affinity `4-7`, all three
NPU cores, and the matching RKNN Runtime 2.5.1. The Rust and C++ outputs were
byte-for-byte identical on both scenes. RGB-to-FP16 packing adds roughly 1-12 ms,
depending on the scene, bringing the end-to-end model path to 235-240 ms.

The public exporter reports 197.0 ms / 5.08 FPS, but that result was not
reproduced. Its exact model artifact is not published, so the difference cannot
currently be attributed to a specific graph, toolchain or measurement choice.

## Next useful configurations

The existing `192x384_it2_d96` graph is already a low-resolution output. For the
real `2026-05-18` pairs, the corresponding `192x384` disparity maxima are 39.21
and 43.78 px. A `max_disp=48` graph therefore covers those scenes without the
d96 search range. For a safety margin on nearer objects, `max_disp=64` is the
more conservative choice.

Recommended measurements, in order:

1. `192x384`, 2 iterations, d96: establish a repeatable 227-236 ms baseline.
2. `192x384`, 2 iterations, d48/d64: quantify cost and disparity drift.
3. `192x384`, 1 iteration, d48/d64: fastest practical profile if the error is
   acceptable.
4. Selective INT8 with FP16 on sensitive layers: only after step 3, because
   single-scale activation quantization can materially change stereo accuracy.

Do not describe `128x256` as free. It is faster, but it loses native detail and
requires upsampling for the final point cloud.
