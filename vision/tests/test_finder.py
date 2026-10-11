"""finder.py logic without a GPU: vocabulary, distractor rule, confirmation, teach store, detector thread.
    python -I vision/tests/test_finder.py"""
import os, sys, tempfile, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orin"))
import numpy as np
import finder as F


def det(label, box, conf):
    return {"label": label, "box": list(box), "conf": conf}


# ---- vocabulary + synonyms
assert len(set(F.VOCAB)) == len(F.VOCAB) and F.VOCAB[:3] == ["keys", "wallet", "water bottle"]
for said, want in [("keys", "keys"), ("key", "keys"), ("my car keys", "keys"), ("keychain", "keys"),
                   ("Wallet", "wallet"), ("my purse", "wallet"), ("bottle", "water bottle"), ("water", "water bottle"),
                   ("my water bottle", "water bottle"), ("the black wallet", "wallet"), ("glasses", "glasses"),
                   ("glasses case", "glasses case"), ("cell phone", "phone"), ("mugs", "cup"), ("remote", "remote control"),
                   ("unicorn", None), ("", None), (None, None)]:
    assert F.canonical(said) == want, (said, F.canonical(said), want)
assert all(v in F.VOCAB for v in F.SYNONYMS.values())
print("OK synonyms")

# ---- wallet vs phone overlap rule
wallet_under_phone = det("wallet", (100, 100, 200, 160), 0.30)
phone_on_top = det("phone", (102, 98, 198, 162), 0.45)          # same object, detector prefers 'phone'
real_wallet = det("wallet", (400, 300, 480, 350), 0.22)
dets = [phone_on_top, wallet_under_phone, real_wallet]
assert F.pick_target(dets, "wallet")["box"] == real_wallet["box"]
assert F.pick_target(dets, "my purse")["box"] == real_wallet["box"]
# wallet stronger than the overlapping phone -> it stays the wallet
dets2 = [det("wallet", (100, 100, 200, 160), 0.50), det("phone", (102, 98, 198, 162), 0.35)]
assert F.pick_target(dets2, "wallet")["conf"] == 0.50
# phone far away does not suppress
dets3 = [det("wallet", (100, 100, 200, 160), 0.20), det("phone", (400, 300, 480, 400), 0.9)]
assert F.pick_target(dets3, "wallet")["conf"] == 0.20
assert F.pick_target([phone_on_top, wallet_under_phone], "wallet") is None
assert F.pick_target(dets, "unicorn") is None and F.pick_target([], "keys") is None
assert F.pick_target([det("water bottle", (0, 0, 50, 120), 0.3)], "bottle")["label"] == "water bottle"
# memory: distractors never written, dominated boxes dropped
mem = F.memory_items(dets + [det("cup", (10, 10, 60, 60), 0.4), det("keys", (300, 10, 330, 40), 0.1)])
assert [d["label"] for d in mem] == ["cup"], mem
print("OK wallet-vs-phone rule")

# ---- Confirmer: 3 of the last 5 within 2 s
c = F.Confirmer(n=5, need=3, iou=0.3, max_age=2.0)
b = (100, 100, 200, 200)
assert not c.confirmed("keys", b, 0.0)
assert not c.confirmed("keys", None, 0.1)
assert not c.confirmed("keys", (105, 102, 205, 198), 0.2)
assert c.confirmed("keys", (98, 100, 202, 205), 0.3)               # 3rd overlapping sighting
assert not c.confirmed("wallet", b, 0.3)                           # per label
c.reset("keys")
for i, bx in enumerate([b, (400, 400, 450, 450), b, (0, 0, 20, 20)]):
    c.confirmed("keys", bx, i * 0.1)
assert not c.confirmed("keys", (300, 0, 340, 30), 0.5)            # boxes elsewhere don't help
c.reset()
assert not c.confirmed("keys", b, 0.0) and not c.confirmed("keys", b, 0.1)
assert not c.confirmed("keys", b, 3.0)                             # old sightings expired
for t in (3.1, 3.2, 3.3, 3.4):
    c.confirmed("keys", None, t)                                   # misses push sightings out of the window of 5
assert not c.confirmed("keys", b, 3.5)
print("OK confirmer needs 3 of 5")

# ---- TeachStore with a fake embedder (mean colour), persistence
calls = []


def fake_embed(crops):
    calls.append(len(crops))
    return np.array([c.reshape(-1, 3).mean(axis=0) + 1.0 for c in crops], np.float32)


frame = np.zeros((480, 640, 3), np.uint8)
frame[100:200, 100:220] = (40, 40, 220)          # red wallet (BGR)
frame[300:400, 400:520] = (220, 60, 30)          # blue wallet
frame[100:180, 400:450] = (40, 50, 210)          # reddish thing the detector calls 'phone'
red, blue, reddish = (100, 100, 220, 200), (400, 300, 520, 400), (400, 100, 450, 180)
x = F.clip_preprocess(frame[100:200, 100:220])
assert x.shape == (3, 224, 224) and x.dtype == np.float32
with tempfile.TemporaryDirectory() as d:
    ts = F.TeachStore(d, embed=fake_embed, sim_min=0.9)
    assert ts.labels() == [] and ts.similarity("wallet", []) is None
    assert ts.teach("my purse", frame, red, det_class="wallet") == 2       # crop + mirror
    assert ts.labels() == ["wallet"] and ts.has("wallet") and ts.classes("wallet") == {"wallet"}
    assert os.path.exists(os.path.join(d, "wallet.npz")) and os.path.exists(os.path.join(d, "wallet.jpg"))
    dets = [det("wallet", blue, 0.6), det("wallet", red, 0.3)]
    assert F.pick_target(dets, "wallet")["box"] == list(blue)                       # untaught: confidence wins
    p = F.pick_target(dets, "wallet", teach=ts, frame=frame)
    assert p["box"] == list(red) and p["sim"] > 0.99, p                             # taught: my wallet wins
    assert F.pick_target([det("wallet", blue, 0.9)], "wallet", teach=ts, frame=frame) is None   # not mine
    # the taught wallet was once detected as 'phone': phone boxes become candidates, checked by similarity
    ts.teach("wallet", frame, red, det_class="phone")
    assert ts.classes("wallet") == {"wallet", "phone"} and len(ts.items["wallet"]["emb"]) == 4
    p = F.pick_target([det("phone", reddish, 0.5), det("wallet", blue, 0.7)], "wallet", teach=ts, frame=frame)
    assert p["box"] == list(reddish), p
    for _ in range(3):
        ts.teach("wallet", frame, red)
    assert len(ts.items["wallet"]["emb"]) == 6                                       # capped at 6
    ts.teach("bottle", frame, blue, det_class="water bottle")
    s_red = ts.similarity("wallet", [frame[100:200, 100:220], frame[300:400, 400:520]])
    assert s_red[0] > s_red[1]
    ts2 = F.TeachStore(d, embed=fake_embed)                                          # reload from disk
    assert ts2.labels() == ["wallet", "water bottle"] and ts2.classes("wallet") == {"wallet", "phone"}
    assert np.allclose(ts2.similarity("wallet", [frame[100:200, 100:220]]), s_red[:1], atol=1e-5)
    assert ts2.forget("wallet") and not ts2.has("wallet") and not os.path.exists(os.path.join(d, "wallet.npz"))
    assert F.TeachStore(d, embed=fake_embed).labels() == ["water bottle"]
print("OK teach store ranks taught instance first, persists and reloads")

# ---- pick_teach_box
dets = [det("laptop", (0, 0, 640, 480), 0.5), det("wallet", (300, 200, 360, 250), 0.3),
        det("cup", (50, 50, 120, 140), 0.6)]
hand = [{"bbox": [290, 190, 60, 50]}]
assert F.pick_teach_box(dets, hand, (640, 480))["label"] == "wallet"
assert F.pick_teach_box(dets, [], (640, 480))["label"] == "wallet"     # nearest the centre beats the cup
assert F.pick_teach_box([det("door", (0, 0, 640, 480), 0.9)], [], (640, 480)) is None
assert F.default_teach_box(hand, (640, 480))[0] < 290
print("OK pick_teach_box")


# ---- DetectorThread: newest frame only, never blocks
class SlowDetector:
    def __init__(self):
        self.seen = []

    def detect(self, frame):
        self.seen.append(int(frame[0, 0, 0]))
        time.sleep(0.15)
        if int(frame[0, 0, 0]) == 13:
            raise RuntimeError("boom")
        return [det("keys", (0, 0, 10, 10), 0.5)]


sd = SlowDetector()
dt = F.DetectorThread(sd, hz=50)
assert dt.latest() == ([], None, None, 0.0)
t0 = time.time()
worst = 0.0
for i in range(30):
    s = time.time()
    dt.submit(np.full((4, 4, 3), i, np.uint8), {"i": i}, i)
    worst = max(worst, time.time() - s)
    dets, meta, fid, t = dt.latest()
    time.sleep(0.01)
assert worst < 0.05, worst
r = dt.wait_newer(time.time(), timeout=2.0)
if r is not None and r[2] != 29:                                       # the frame in flight finished first
    r = dt.wait_newer(r[3], timeout=2.0)
assert r is not None and r[2] == 29 and r[1] == {"i": 29}, r
assert len(sd.seen) < 10 and sd.seen[-1] == 29 and dt.dropped > 15, (sd.seen, dt.dropped)
assert dt.latest(with_frame=True)[4][0, 0, 0] == 29
dt.submit(np.full((4, 4, 3), 13, np.uint8), None, 13)                 # detector error -> empty result, thread lives
r = dt.wait_newer(r[3], timeout=2.0)
assert r[0] == [] and r[2] == 13
dt.submit(np.full((4, 4, 3), 14, np.uint8), None, 14)
assert dt.wait_newer(r[3], timeout=2.0)[2] == 14
# rate limit: hz=4 -> at most ~4 results a second
sd2 = SlowDetector()
sd2.detect = lambda f: []
dt2 = F.DetectorThread(sd2, hz=4)
n0, end = dt2.processed, time.time() + 1.0
while time.time() < end:
    dt2.submit(np.zeros((4, 4, 3), np.uint8))
    time.sleep(0.005)
assert dt2.processed - n0 <= 5, dt2.processed
dt.stop(); dt2.stop()
print("OK detector thread drops stale frames, never blocks, rate-limited")
print("ALL OK")
