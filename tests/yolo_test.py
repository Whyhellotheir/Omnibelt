"""Orin track. Exports YOLO to TensorRT once, then runs it on the webcam or a video.
    python3 yolo_test.py            # webcam
    python3 yolo_test.py walk.mp4   # video file
Prints frames/sec. Expect well above 15 on the Orin Nano Super with the .engine file."""
import os, sys, time
import cv2
from ultralytics import YOLO

if not os.path.exists("yolo11n.engine"):
    YOLO("yolo11n.pt").export(format="engine", half=True)   # a few minutes, once
model = YOLO("yolo11n.engine", task="detect")
src = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(src)
n, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model.track(frame, persist=True, verbose=False)[0]
    n += 1
    if n % 30 == 0:
        names = [model.names[int(c)] for c in res.boxes.cls.tolist()]
        print(f"{n / (time.time() - t0):5.1f} fps  {names}")
    cv2.imshow("yolo", res.plot())
    if cv2.waitKey(1) == 27:
        break
