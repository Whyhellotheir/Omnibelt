"""Orin track. Checks the find-anything detector (YOLO-World) the server uses for 'find my ...'.
    python3 find_test.py desk.jpg keys wallet bottle      # one photo, any words
    python3 find_test.py 0 keys                           # webcam, live
    python3 find_test.py walk.mp4 keys cup                # a recorded video
Prints what it found, how sure it is (0-1) and how long it took; saves found_local.jpg.
First run downloads the model and its text encoder (needs internet). If that step fails:
    pip install git+https://github.com/ultralytics/CLIP.git"""
import sys, time
import cv2
from ultralytics import YOLOWorld

src, words = sys.argv[1], sys.argv[2:] or ["keys"]
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(words)
conf = 0.15                                   # same threshold as the server (OPEN_VOCAB_CONF)

def run(frame):
    t0 = time.time()
    r = model.predict(frame, conf=conf, verbose=False)[0]
    ms = (time.time() - t0) * 1000
    found = [(words[int(c)], round(float(p), 2)) for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    return r, ms, found

if src.lower().endswith((".jpg", ".jpeg", ".png")):
    img = cv2.imread(src)
    run(img)                                  # warm-up
    r, ms, found = run(img)
    print(f"{ms:.0f} ms  found: {found or 'nothing'}")
    cv2.imwrite("found_local.jpg", r.plot())
    print("saved found_local.jpg")
else:
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r, ms, found = run(cv2.resize(frame, (640, 480)))
        print(f"{ms:5.0f} ms  {found}")
        cv2.imshow("find-anything", r.plot())
        if cv2.waitKey(1) == 27:
            break
