import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orin"))
import numpy as np
from memory import Memory

rng = np.random.default_rng(0)
CAM = (0.0, 0.0)


def jit(p, s=0.06):
    return (p[0] + rng.normal(0, s), p[1] + rng.normal(0, s))


# --- moved keys: A -> B with noisy interleaved sightings
m = Memory()
A, B = (1.0, -2.0), (-1.2, -2.6)
t = 100.0
for _ in range(5):
    assert m.add("keys", jit(A), 0.75, 0.8, CAM, t); t += 1
e = m.recall("keys")
assert e and np.hypot(*(e["pos"] - A)) < 0.15 and not e["moved"]
m.add("keys", (3.0, 1.0), 0.1, 0.4, CAM, t); t += 1            # single false positive elsewhere
e = m.recall("my keys")
assert np.hypot(*(e["pos"] - A)) < 0.15, e
for i in range(6):
    m.add("keys", jit(B), 0.75, 0.8, CAM, t); t += 1
    if i == 1:
        m.add("keys", (-3.0, -1.0), 0.3, 0.3, CAM, t); t += 1   # more noise
e = m.recall("where are my key")
assert np.hypot(*(e["pos"] - B)) < 0.15 and e["moved"], e
assert np.hypot(*(e["moved_from"] - A)) < 0.15
print("OK moved A->B, moved=True, pos", e["pos"].round(2))

# --- false positive does not overwrite a confirmed position
m = Memory()
for i in range(4):
    m.add("wallet", jit(A), 0.7, 0.9, CAM, 10 + i)
m.add("wallet", (-2.0, -3.0), 0.7, 0.9, CAM, 20)
e = m.recall("wallet")
assert np.hypot(*(e["pos"] - A)) < 0.15 and not e["moved"] and e["n"] == 4
print("OK false positive ignored")

# --- unconfirmed only -> None; synonyms/word matching
m = Memory()
m.add("water bottle", (0.5, -1.0), 0.8, 0.9, CAM, 1)
assert m.recall("bottle") is None
m.add("water bottle", (0.52, -1.0), 0.8, 0.9, CAM, 2)
assert m.recall("bottle")["label"] == "water bottle"
assert m.recall("drink", synonyms=["water bottle"])["label"] == "water bottle"
assert m.recall("keys") is None
print("OK recall matching")

# --- out of range rejected
m = Memory(max_range=3.5)
assert not m.add("keys", (0.0, -4.0), 0.7, 1.0, CAM, 1)
assert m.add("keys", (0.0, -3.0), 0.7, 1.0, CAM, 1)
assert m.add("keys", (0.0, 3.4), 0.7, 1.0, CAM, 2)
assert not m.add("keys", (5.0, 5.0), 0.7, 1.0, (0.0, 0.0), 3)
print("OK out of range")

# --- describe phrases
NOW = 1000.0


def ent(label, pos, h, t=NOW):
    return {"label": label, "pos": np.array(pos, float), "height": h, "t": t}


cases = [
    (ent("keys", (1.0, 2.0), 0.75), 0, (0, 0), "Your keys are on the table behind you, to your right, about 2 meters away."),
    (ent("wallet", (0.0, -1.5), 0.1), 0, (0, 0), "Your wallet is on the floor straight ahead, about one and a half meters away."),
    (ent("keys", (-0.9, -0.4), 1.5), 0, (0, 0), "Your keys are on a shelf to your left, about one meter away."),
    (ent("keys", (0.0, 3.0), 0.75), 0, (0, 0), "Your keys are on the table right behind you, about 3 meters away."),
    (ent("keys", (0.3, -0.5), 0.35), 0, (0, 0), "Your keys are about waist high slightly to your right, within reach."),
    (ent("water bottle", (0.0, -2.0), None), 0, (0, 0), "Your water bottle is straight ahead, about 2 meters away."),
    (ent("keys", (0.0, -1.5), 0.75), 90, (0, 0), "Your keys are on the table to your left, about one and a half meters away."),
]
for en, hd, pos, want in cases:
    got = m.describe(en, hd, pos, NOW)
    assert got == want, f"\n got:  {got}\n want: {want}"
# turned to face judges: keys at (1,-2) world, user heading 180
got = m.describe(ent("keys", (1.0, -2.0), 0.75), 180, (0, 0), NOW)
assert got == "Your keys are on the table behind you, to your left, about 2 meters away.", got
# age phrasing
assert m.describe(ent("keys", (0, -2), 0.75, NOW - 45), 0, (0, 0), NOW).endswith("I saw them 45 seconds ago.")
assert m.describe(ent("keys", (0, -2), 0.75, NOW - 300), 0, (0, 0), NOW).endswith("I saw them 5 minutes ago.")
assert "ago" not in m.describe(ent("keys", (0, -2), 0.75, NOW - 15), 0, (0, 0), NOW)
assert "I saw it" in m.describe(ent("wallet", (0, -2), 0.75, NOW - 45), 0, (0, 0), NOW)
assert len(m.describe(ent("keys", (1, 2), 0.75, NOW - 300), 0, (0, 0), NOW).split()) <= 22
print("OK describe:", got)

# --- negative evidence
m = Memory()
for i in range(3):
    m.add("keys", (0.0, -1.5), 0.75, 0.9, CAM, i)
c0 = m.recall("keys")["conf"]
assert not m.negative("keys", CAM, 120, 35, 10)          # looking away: no miss
assert not m.negative("keys", CAM, 0, 35, 10)
assert not m.negative("keys", CAM, 0, 35, 11)
assert not m.recall("keys")["stale"]
assert m.negative("keys", CAM, 0, 35, 12)
e = m.recall("keys")
assert e["stale"] and e["conf"] < c0 and m.pins(20)[0]["stale"]
m.add("keys", (0.0, -1.5), 0.75, 0.9, CAM, 13)            # fresh sighting resets
assert not m.recall("keys")["stale"]
far = Memory()
for i in range(2):
    far.add("keys", (0.0, -3.0), 0.75, 0.9, CAM, i)
for _ in range(5):
    far.negative("keys", CAM, 0, 35, 5)                    # beyond 2 m: never counts
assert not far.recall("keys")["stale"]
print("OK negative evidence")

# --- pins / snapshot / clear / thumbs
m = Memory()
m.add("keys", (1, -2), 0.7, 0.9, CAM, 1, thumb=b"jpg1")
m.add("keys", (1, -2), 0.7, 0.9, CAM, 2, thumb=b"jpg2")
p = m.pins(32)[0]
assert p["label"] == "keys" and abs(p["age_s"] - 30) < 1e-6 and set(p) >= {"x", "z", "height", "conf", "moved", "stale"}
assert m.thumbs["keys"] == b"jpg2" and "keys" in m.snapshot()
m.clear()
assert not m.pins(0) and not m.thumbs and m.recall("keys") is None
print("OK pins/snapshot/clear")

# --- thread safety smoke
m = Memory()
errs = []


def writer(i):
    try:
        for j in range(300):
            m.add(f"obj{i % 3}", (rng.random() * 3, -rng.random() * 3), 0.7, 0.9, CAM, j)
            m.negative(f"obj{i % 3}", CAM, 0, 35, j)
    except Exception as ex:
        errs.append(ex)


def reader():
    try:
        for _ in range(300):
            m.pins(1.0); m.snapshot(); m.recall("obj1")
    except Exception as ex:
        errs.append(ex)


ths = [threading.Thread(target=writer, args=(i,)) for i in range(4)] + [threading.Thread(target=reader) for _ in range(2)]
[x.start() for x in ths]
[x.join() for x in ths]
assert not errs, errs
print("OK thread safety")
