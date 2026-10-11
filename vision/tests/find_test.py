"""Orin track. Checks the finder the server uses for 'find my ...' (vision/orin/finder.py: YOLO-World with one
fixed vocabulary, distractor rule, optional TensorRT engine and taught items).
    python3 find_test.py desk.jpg keys wallet bottle      # one photo, any words
    python3 find_test.py 0 keys                           # webcam, live (Confirmer must agree 3 of 5 frames)
    python3 find_test.py walk.mp4 keys cup                # a recorded video
    TEACH_DIR=../orin/taught python3 find_test.py desk.jpg wallet   # rank by the taught wallet too
Prints every detection, the pick for each word, how sure it is (0-1) and how long it took; saves found_local.jpg.
Words outside the vocabulary use the separate open-vocabulary model. Engine: cd ../orin && python3 finder.py export
First run downloads the model and its text encoder (needs internet). If that step fails:
    pip install git+https://github.com/ultralytics/CLIP.git"""
import os, sys, time
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orin"))
import finder as F

src, words = sys.argv[1], sys.argv[2:] or ["keys"]
det = F.WorldDetector()
teach = F.TeachStore(os.environ["TEACH_DIR"]) if os.environ.get("TEACH_DIR") else None
conf_ = F.Confirmer()
print("vocabulary:", F.VOCAB, "| engine" if det.engine else "| pytorch")
print("words ->", {w: F.canonical(w) for w in words})

def run(frame):
    t0 = time.time()
    dets = det.detect(frame)
    picks = {}
    for w in words:
        if F.canonical(w) or (teach and teach.has(w)):
            picks[w] = F.pick_target(dets, w, teach, frame)
        else:
            o = det.detect_open(frame, w)
            picks[w] = o[0] if o else None
    return dets, picks, (time.time() - t0) * 1000

def draw(frame, dets, picks):
    out = frame.copy()
    for d in dets:
        x0, y0, x1, y1 = map(int, d["box"])
        cv2.rectangle(out, (x0, y0), (x1, y1), (128, 128, 128), 1)
        cv2.putText(out, f"{d['label']} {d['conf']:.2f}", (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for w, p in picks.items():
        if p:
            x0, y0, x1, y1 = map(int, p["box"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(out, f"{w} {p.get('score', p['conf']):.2f}", (x0, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out

def summary(picks, now=None, confirm=False):
    s = {}
    for w, p in picks.items():
        if p is None:
            s[w] = None
            continue
        s[w] = round(p.get("score", p["conf"]), 2)
        if confirm and not conf_.confirmed(w, p["box"], now):
            s[w] = f"{s[w]} (unconfirmed)"
    if confirm:
        for w, p in picks.items():
            if p is None:
                conf_.confirmed(w, None, now)
    return s

if src.lower().endswith((".jpg", ".jpeg", ".png")):
    img = cv2.imread(src)
    run(img)                                  # warm-up
    dets, picks, ms = run(img)
    print(f"{ms:.0f} ms  all: {[(d['label'], round(d['conf'], 2)) for d in dets] or 'nothing'}")
    print("picks:", summary(picks))
    cv2.imwrite("found_local.jpg", draw(img, dets, picks))
    print("saved found_local.jpg")
else:
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480))
        dets, picks, ms = run(frame)
        print(f"{ms:5.0f} ms  {summary(picks, time.time(), confirm=True)}")
        cv2.imshow("find-anything", draw(frame, dets, picks))
        if cv2.waitKey(1) == 27:
            break
