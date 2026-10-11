"""Mock tests for vision/orin/depth_trt.py (no GPU / torch needed). Run: python -I vision/tests/test_depth_trt.py"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orin"))
import numpy as np
import cv2
import depth_trt as D


class FakeModel:
    """returns a constant array at model resolution (by aspect) = 1 + frame[0,0,0]; optional sleep / failures."""

    def __init__(self, sleep=0.1, fail_on=()):
        self.sleep, self.fail_on, self.seen = sleep, set(fail_on), []

    def __call__(self, frame):
        tag = int(frame[0, 0, 0])
        self.seen.append(tag)
        time.sleep(self.sleep)
        if tag in self.fail_on:
            raise ValueError("boom %d" % tag)
        h, w = D.pick_hw(frame.shape)
        return np.full((1, h, w), 1.0 + tag, np.float32)


def frame(tag, h=480, w=640):
    f = np.zeros((h, w, 3), np.uint8)
    f[0, 0, 0] = tag
    return f


def wait_for(worker, fid, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = worker.latest()
        if r is not None and r[2] == fid:
            return r
        time.sleep(0.01)
    raise AssertionError("no result for frame %s (latest %s)" % (fid, worker.latest() and worker.latest()[2]))


# --- preprocessing math
img = np.zeros((480, 640, 3), np.uint8)
img[..., 0], img[..., 1], img[..., 2] = 10, 128, 250          # B, G, R
x = D.preprocess(img, 252, 336)
assert x.shape == (1, 3, 252, 336) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
exp = (np.array([250, 128, 10], np.float32) / 255 - D.MEAN) / D.STD          # RGB order
assert np.allclose(x[0, :, 100, 100], exp, atol=1e-5), (x[0, :, 100, 100], exp)
xp = D.preprocess(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), *D.PORT_HW)
assert xp.shape == (1, 3, 336, 252)
assert D.pick_hw((480, 640)) == (252, 336) and D.pick_hw((640, 480)) == (336, 252)
assert D.engine_path("e", D.LAND_HW).endswith("depth_252x336.engine")
print("OK preprocess: RGB order, /255, ImageNet mean/std, NCHW shape, orientation pick")

# --- newest frame wins, stale dropped, resize to 640x480
m = FakeModel(sleep=0.15)
w = D.DepthWorker(m)
t0 = time.time()
for i in range(1, 6):
    w.submit(frame(i), {"i": i}, i)
dt = time.time() - t0
assert dt < 0.05, dt
r = wait_for(w, 5)
d, meta, fid, t, ms = r
assert d.shape == (480, 640) and d.dtype == np.float32 and np.allclose(d, 6.0) and meta == {"i": 5}
assert 5 not in m.seen[:-1] and len(m.seen) <= 2 and 2 not in m.seen and 3 not in m.seen and 4 not in m.seen, m.seen
assert ms >= 140, ms
print("OK newest wins: model saw %s of 1..5, dropped=%d, submit x5 took %.4f s, infer %.0f ms" % (m.seen, w.dropped, dt, ms))

# --- portrait 480x640 (h=640, w=480)
w.submit(frame(7, 640, 480), {}, 7)
d = wait_for(w, 7)[0]
assert d.shape == (640, 480) and np.allclose(d, 8.0)
print("OK portrait frame -> depth", d.shape)

# --- submit never blocks while inference is busy
m.sleep = 0.5
w.submit(frame(8), {}, 8)
time.sleep(0.05)
worst = 0.0
for i in range(9, 30):
    t0 = time.time()
    w.submit(frame(i), {}, i)
    worst = max(worst, time.time() - t0)
assert worst < 0.05, worst
assert w.latest()[2] in (7, 8)          # latest() is non-blocking too and returns the previous result
m.sleep = 0.05
wait_for(w, 29)
print("OK submit never blocks during a 0.5 s inference (worst %.4f s)" % worst)

# --- model exception: thread survives, error counted, next frame works
m.fail_on = {40}
w.submit(frame(40), {}, 40)
time.sleep(0.3)
assert w.thread.is_alive() and w.errors == 1 and w.latest()[2] == 29
w.submit(frame(41), {}, 41)
assert np.allclose(wait_for(w, 41)[0], 42.0)
for _ in range(3):                      # repeated failures are logged at most once per 10 s
    w.submit(frame(40), {}, 40)
    time.sleep(0.12)
assert w.thread.is_alive() and w.errors == 4
print("OK model exception caught, worker alive, %d errors" % w.errors)

w.stop()
w.thread.join(2)
assert not w.thread.is_alive()

# --- make_depth with no engines tries HF; on this box (no torch) that raises ImportError, which is the caller's cue
try:
    D.make_depth(os.path.dirname(os.path.abspath(__file__)))
    print("OK make_depth (HF available here)")
except ImportError as e:
    print("OK make_depth falls back to HF, which needs torch (absent here):", e)
print("ALL OK")
