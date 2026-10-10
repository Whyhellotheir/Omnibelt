"""Orin track. Runs Depth Anything V2 (metric indoor, meters) on one image or the webcam.
    pip install transformers pillow
    python3 depth_test.py room.jpg
Prints the distance at the image center and how long each frame takes; saves depth.png.
If the model name has changed, search Hugging Face for 'Depth-Anything-V2 Metric Indoor Small'."""
import sys, time
import numpy as np, cv2, torch
from PIL import Image
from transformers import pipeline

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
pipe = pipeline("depth-estimation", model=MODEL, device=0 if torch.cuda.is_available() else -1)
if len(sys.argv) > 1:
    frames = [cv2.imread(sys.argv[1])]
else:
    cap = cv2.VideoCapture(0)
    frames = [cap.read()[1] for _ in range(20)]
for f in frames:
    small = cv2.resize(f, (320, 240))
    t0 = time.time()
    out = pipe(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    d = out["predicted_depth"].squeeze().cpu().numpy()
    print(f"{(time.time() - t0) * 1000:5.0f} ms   center = {d[d.shape[0] // 2, d.shape[1] // 2]:.2f} m")
vis = cv2.applyColorMap(cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imwrite("depth.png", vis)
print("saved depth.png")
