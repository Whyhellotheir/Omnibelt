"""Fast monocular depth for the Orin: Depth-Anything-V2-Metric-Indoor-Small as a TensorRT engine
(fallback: the HF model in fp16), run on a background thread so perception never waits on it.

  model = make_depth("vision/orin")         # TRT if depth_252x336.engine + depth_336x252.engine exist, else HF
  worker = DepthWorker(model)
  worker.submit(frame, meta, frame_id)      # never blocks; only the newest pending frame is kept
  r = worker.latest()                       # None or (depth HxW float32 metres, meta, frame_id, t, infer_ms)

Every model object is callable: model(bgr) -> float32 (h, w) metres at model resolution.
tensorrt / torch / transformers are imported lazily so this module imports anywhere (numpy + cv2 only).
"""
import os
import threading
import time

import cv2
import numpy as np

HF_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
LAND_HW, PORT_HW = (252, 336), (336, 252)          # multiples of 14 (ViT patch), ~4:3
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, same as the HF DPT image processor
STD = np.array([0.229, 0.224, 0.225], np.float32)


def engine_path(engine_dir, hw):
    return os.path.join(engine_dir, "depth_%dx%d.engine" % hw)


def pick_hw(shape):
    """model input (h, w) for an image of this shape: landscape unless taller than wide."""
    return PORT_HW if shape[0] > shape[1] else LAND_HW


def preprocess(bgr, h, w):
    """BGR uint8 (H, W, 3) -> float32 (1, 3, h, w), RGB, /255, ImageNet-normalised. Pure numpy/cv2."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) * (1.0 / 255.0) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class TRTDepth:
    """One static-shape TensorRT 10 engine (input 1x3xhxw -> depth 1xhxw or hxw). No pycuda: torch owns the buffers."""

    def __init__(self, path, h, w):
        import tensorrt as trt
        import torch
        self.torch, self.h, self.w = torch, h, w
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("could not deserialize " + path)
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        tdt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}
        self.inp = self.out = None
        self.bufs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(s < 0 for s in shape):
                raise RuntimeError("engine %s has dynamic shape %s for %s; export it static" % (path, shape, name))
            buf = torch.empty(shape, dtype=tdt[self.engine.get_tensor_dtype(name)], device="cuda")
            self.bufs[name] = buf
            self.ctx.set_tensor_address(name, buf.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inp = name
            else:
                self.out = name
        ishape = tuple(self.bufs[self.inp].shape)
        if ishape[-2:] != (h, w):
            raise RuntimeError("engine %s input is %s, expected (..., %d, %d)" % (path, ishape, h, w))

    def __call__(self, bgr):
        torch = self.torch
        x = torch.from_numpy(preprocess(bgr, self.h, self.w))
        with torch.cuda.stream(self.stream):
            self.bufs[self.inp].copy_(x.to("cuda", non_blocking=True))
            if not self.ctx.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            d = self.bufs[self.out].float().reshape(self.h, self.w).cpu()
        self.stream.synchronize()
        return d.numpy().astype(np.float32, copy=False)


class DualTRT:
    """landscape + portrait engines; picks one by the frame's aspect."""

    def __init__(self, engine_dir):
        self.land = TRTDepth(engine_path(engine_dir, LAND_HW), *LAND_HW)
        self.port = TRTDepth(engine_path(engine_dir, PORT_HW), *PORT_HW)
        self.name = "TensorRT %dx%d/%dx%d" % (LAND_HW + PORT_HW)

    def __call__(self, bgr):
        return (self.port if bgr.shape[0] > bgr.shape[1] else self.land)(bgr)


class HFDepth:
    """Fallback: the HF model (loaded via the pipeline) fed the same 336-wide preprocessed tensor as the engines,
    fp16 on CUDA. Skips the pipeline's own resize-to-518, which is most of the old cost."""

    def __init__(self, model_id=HF_MODEL):
        import torch
        from transformers import pipeline
        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.dtype = torch.float16 if self.cuda else torch.float32
        pipe = pipeline("depth-estimation", model=model_id, device=0 if self.cuda else -1, torch_dtype=self.dtype)
        self.model = pipe.model.eval()
        self.dev = "cuda" if self.cuda else "cpu"
        self.name = "HF %s %s" % ("fp16 cuda" if self.cuda else "fp32 cpu", model_id.split("/")[-1])

    def __call__(self, bgr):
        torch = self.torch
        h, w = pick_hw(bgr.shape)
        x = torch.from_numpy(preprocess(bgr, h, w)).to(self.dev, self.dtype)
        with torch.inference_mode():
            d = self.model(pixel_values=x).predicted_depth
        d = d.float().reshape(d.shape[-2], d.shape[-1]).cpu().numpy()
        return d if d.shape == (h, w) else cv2.resize(d, (w, h))


def make_depth(engine_dir):
    """TRT if both engines exist and load, else HF. Raises only if HF also fails (caller may run without depth)."""
    if all(os.path.exists(engine_path(engine_dir, hw)) for hw in (LAND_HW, PORT_HW)):
        try:
            m = DualTRT(engine_dir)
            print("Depth:", m.name)
            return m
        except Exception as e:
            print("Depth: TensorRT engines failed to load, falling back to HF:", e)
    else:
        print("Depth: no TensorRT engines in %s, using HF" % engine_dir)
    m = HFDepth()
    print("Depth:", m.name)
    return m


class DepthWorker:
    """Runs model(frame) on a daemon thread. submit() never blocks and replaces any frame not yet started."""

    def __init__(self, model, err_every=10.0):
        self.model, self.err_every = model, err_every
        self.cv = threading.Condition()
        self.pending, self.result = None, None
        self.dropped, self.done, self.errors = 0, 0, 0
        self._last_err_log, self._stop = 0.0, False
        self.thread = threading.Thread(target=self._run, name="depth", daemon=True)
        self.thread.start()

    def submit(self, frame, meta, frame_id):
        f = frame.copy()        # caller may draw on / reuse its frame
        with self.cv:
            if self.pending is not None:
                self.dropped += 1
            self.pending = (f, meta, frame_id)
            self.cv.notify()

    def latest(self):
        with self.cv:
            return self.result

    def stop(self):
        with self.cv:
            self._stop = True
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None and not self._stop:
                    self.cv.wait(0.5)
                if self._stop:
                    return
                frame, meta, fid = self.pending
                self.pending = None
            try:
                t0 = time.time()
                d = np.asarray(self.model(frame), np.float32)
                d = d.reshape(d.shape[-2], d.shape[-1])
                H, W = frame.shape[:2]
                if d.shape != (H, W):
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                res = (d, meta, fid, time.time(), (time.time() - t0) * 1000.0)
                with self.cv:
                    self.result = res
                    self.done += 1
            except Exception as e:
                self.errors += 1
                now = time.time()
                if now - self._last_err_log >= self.err_every:
                    self._last_err_log = now
                    print("Depth worker error (%d so far): %r" % (self.errors, e))


if __name__ == "__main__":
    # on the Orin:  python3 depth_trt.py [engine_dir] [image]  -> which backend, latency, depth range
    import sys
    edir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(sys.argv[2]) if len(sys.argv) > 2 else None
    if img is None:
        img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    m = make_depth(edir)
    for frame in (img, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)):
        m(frame)
        t0 = time.time()
        for _ in range(20):
            d = m(frame)
        print("%s in %s -> out %s  %.1f ms  depth %.2f..%.2f m (median %.2f)" % (
            m.name, frame.shape[:2], d.shape, (time.time() - t0) / 20 * 1000, d.min(), d.max(), np.median(d)))
