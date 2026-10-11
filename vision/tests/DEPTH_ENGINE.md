# Depth TensorRT engines (Depth-Anything-V2-Metric-Indoor-Small)

Two static engines, picked per frame by aspect in `vision/orin/depth_trt.py`:

| file (in `vision/orin/`) | input (H x W) | used for |
|---|---|---|
| `depth_252x336.engine` | 1x3x252x336 | landscape frames (640x480) |
| `depth_336x252.engine` | 1x3x336x252 | portrait frames (480x640) |

Both must exist or `make_depth()` falls back to the HF model (fp16 on CUDA, same 336-wide input).
Engines are tied to the exact TensorRT version + GPU they were built on: build them **on the Orin**, rebuild after any JetPack/TensorRT upgrade. Do not commit `.engine` or `.onnx` files.

## 1. Export ONNX (PC or Orin, needs torch + transformers)

```bash
pip install torch transformers onnx onnxruntime      # onnxruntime optional: enables the torch-vs-ONNX check
python vision/tests/depth_export.py                  # writes vision/tests/onnx/depth_252x336.onnx and depth_336x252.onnx
```

Check the output: each shape prints `onnxruntime vs torch: max abs ...` and should be well under 0.01 m. Opset 17, legacy (TorchScript) exporter, static shapes, input `pixel_values`, output `predicted_depth` (1xHxW metres). The engine loader reads tensor names from the engine, so renaming is harmless.

## 2. Copy to the Orin (skip if you exported on the Orin)

```bash
scp vision/tests/onnx/depth_*.onnx <user>@orin.local:~/Knighthacks9/vision/orin/
```

## 3. Build engines on the Orin (repo root, ~3-8 min each)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks          # max clocks (MAXN SUPER) for build + run
/usr/src/tensorrt/bin/trtexec --onnx=vision/orin/depth_252x336.onnx --saveEngine=vision/orin/depth_252x336.engine --fp16 --memPoolSize=workspace:2048MiB
/usr/src/tensorrt/bin/trtexec --onnx=vision/orin/depth_336x252.onnx --saveEngine=vision/orin/depth_336x252.engine --fp16 --memPoolSize=workspace:2048MiB
```

trtexec prints a latency summary at the end (`GPU Compute Time: mean = ...`). Stop the navbelt server while building (memory).

## 4. Verify (fp16 correctness + latency)

```bash
cd vision/orin
python3 depth_trt.py . ../tests/room.jpg
```

Prints backend (`TensorRT 252x336/336x252`), ms per frame and the depth range for the landscape image and its portrait rotation. For `room.jpg` expect plausible indoor metres (roughly 0.5..8 m, median a few metres).

Check fp16 against fp32: temporarily move the engines away, run the same command (HF fallback path; fp16 too) — or better, build an fp32 engine (same trtexec command without `--fp16`, e.g. `--saveEngine=/tmp/depth_fp32.engine`) and compare. Signs of fp16 trouble in the DPT head: NaN/inf, all-constant depth, values pinned at 0 or at the max (20 m), heavy blockiness. If that happens:

1. Rebuild without `--fp16` (fp32, roughly 2x slower but still far faster than the HF pipeline), or
2. Keep fp16 for the ViT backbone and force the head to fp32:
   `--fp16 --precisionConstraints=obey --layerPrecisions=<head layer names>:fp32` (get names from `--dumpLayerInfo --profilingVerbosity=detailed`; the layers under `/m/head/...` and the final sigmoid/mul by max depth).

## Expected latency (estimate, not yet measured on this board)

- HF pipeline today (518 internal resize, fp32, every other frame): ~80-150 ms per call.
- HF fallback in this module (fp16, 336 wide): ~40-70 ms.
- TensorRT fp16 at 252x336: ~12-25 ms; fp32: ~25-45 ms.

Inference runs on `DepthWorker`'s thread, so the perception loop (YOLO) no longer waits on it; depth arrives at its own rate and carries its `frame_id`/`meta` so stale depth can be detected.

## Fallback

- Engines missing or fail to load -> HF fp16 model (logged: `Depth: ...`).
- HF also fails -> `make_depth` raises; the server should run without estimated depth (LiDAR depth from the phone and YOLO-box distances still work), as `USE_DEPTH=0` does today.
- Inference errors inside the worker are logged at most once per 10 s and never kill the thread; `latest()` keeps returning the last good result (check its timestamp).
