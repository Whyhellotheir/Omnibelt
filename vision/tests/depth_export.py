"""Export Depth-Anything-V2-Metric-Indoor-Small to static ONNX for TensorRT (landscape 252x336 + portrait 336x252).

Run on any machine with torch + transformers (PC or the Orin):
    pip install torch transformers onnx onnxruntime     # onnxruntime optional (sanity check)
    python vision/tests/depth_export.py [out_dir]       # default out_dir: vision/tests/onnx
Then build engines on the Orin with the printed trtexec commands (see vision/tests/DEPTH_ENGINE.md).
Preprocessing must match vision/orin/depth_trt.preprocess (RGB, /255, ImageNet mean/std, INTER_AREA resize).
"""
import inspect
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SHAPES = [(252, 336), (336, 252)]        # (H, W): landscape, portrait


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth      # (1, H, W) metres


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "onnx")
    os.makedirs(out_dir, exist_ok=True)
    net = Wrap(AutoModelForDepthEstimation.from_pretrained(MODEL).eval().float())
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    paths = []
    for h, w in SHAPES:
        path = os.path.join(out_dir, "depth_%dx%d.onnx" % (h, w))
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            ref = net(x).numpy()
            torch.onnx.export(net, (x,), path, input_names=["pixel_values"], output_names=["predicted_depth"],
                              opset_version=17, do_constant_folding=True, **kw)
        print("wrote %s  (torch out %s, %.1f MB)" % (path, ref.shape, os.path.getsize(path) / 1e6))
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
            err = np.abs(got - ref)
            print("  onnxruntime vs torch: max abs %.4f m, mean abs %.5f m (torch range %.2f..%.2f)"
                  % (err.max(), err.mean(), ref.min(), ref.max()))
            assert err.max() < 1e-2 * max(1.0, float(np.abs(ref).max())), "ONNX output diverges from torch"
        except ImportError:
            print("  (onnxruntime not installed: skipped ONNX sanity check)")
        paths.append(path)
    print("\nOn the Orin (from the repo root, ONNX files copied to vision/orin/):")
    for h, w in SHAPES:
        print("/usr/src/tensorrt/bin/trtexec --onnx=vision/orin/depth_%dx%d.onnx --saveEngine=vision/orin/depth_%dx%d.engine"
              " --fp16 --memPoolSize=workspace:2048MiB" % (h, w, h, w))


if __name__ == "__main__":
    main()
