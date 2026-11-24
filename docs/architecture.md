# Architecture

`FoundationStereo` owns one RKNN context containing the complete two-iteration
network. Each frame follows one path:

```text
left RGB  ---> NHWC FP16 buffer --\
                                   rknn_inputs_set -> rknn_run -> rknn_outputs_get
right RGB ---> NHWC FP16 buffer --/                                      |
                                                                         v
                                                              [1,1,H,W] FP32
```

There are no intermediate CPU tensors, ONNX Runtime sessions, correlation-volume
kernels, or per-stage RKNN launches.

## Runtime configuration

- `RknnOptions::default()` selects `NpuCore::All` and resolves the runtime from
  `RKNN_LIBRARY`, `/usr/lib/librknnrt.so`, then the system loader.
- Inputs use RKNN pass-through mode, so RKNN does not convert FP32 to FP16 again.
- `infer` accepts prepared `F16Image` values and avoids per-frame allocation.
- `infer_rgb8` and `infer_nchw` reuse internal buffers and repack only the two
  camera planes.
- The RKNN model bytes and runtime library remain alive for the context lifetime,
  as required by the RKNN C API.

## Reference measurement

Full-hardware configuration on NanoPC-T6-LTS:

| Setting | Value |
|---|---|
| CPU affinity | `4-7` |
| A76 frequency | 2.256 GHz |
| A55 frequency | 1.8 GHz |
| NPU frequency | 1.0 GHz |
| NPU core mask | `7` (all three cores) |
| RKNN Runtime | 2.5.1 |
| Model | `192x384`, `valid_iters=2`, `max_disp=96`, FP16 |

| Scene | Prepared FP16 median | Run range | P95 | FPS | RGB8 median |
|---|---:|---:|---:|---:|---:|
| Pair 1 | 206.489 ms | 200.288-211.345 ms | 212.323 ms | 4.843 | 208.950 ms |
| Pair 2 | 209.160 ms | 206.955-212.328 ms | 212.737 ms | 4.781 | 212.607 ms |

The table reports the median of three independent run medians, each using 10
warmups and 50 measured frames.

## Further speed work

The graph is already single-pass and uses all three NPU cores. Meaningful next
options require changing the model rather than the wrapper:

1. Reduce `max_disp` from 96 to 48 or 64 after validating the expected depth
   range. The two real validation pairs need only 39.21 and 43.78 model pixels.
2. Evaluate one refinement iteration with task-specific error limits.
3. Use selective INT8 only after collecting representative calibration data and
   validating depth discontinuities; uniform INT8 is risky for this network.
