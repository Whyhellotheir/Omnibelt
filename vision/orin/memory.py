"""Object memory for 'where did I leave my keys?'. Pure python + numpy, thread-safe.

Per label we keep tracks (clusters of sightings). A track's position is a weighted median of its last
observations, so one false positive cannot overwrite a confirmed position. A new confirmed track while the
old one has not been seen = the object was moved."""
import math
import threading
import time
from collections import deque

import numpy as np

ASSIGN_M, OUTLIER_M, NEG_RANGE_M, NEG_MISSES = 0.6, 0.8, 2.0, 3
MAX_OBS, MAX_TRACKS, UNCONFIRMED_TTL_S = 8, 6, 30.0
STOP = {"my", "the", "a", "an", "your", "some", "of"}


def _stem(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(text):
    return {_stem(w) for w in str(text).lower().replace("-", " ").split() if w not in STOP}


def _wmedian(vals, w):
    o = np.argsort(vals)
    v, w = np.asarray(vals)[o], np.asarray(w)[o]
    c = np.cumsum(w)
    return float(v[min(int(np.searchsorted(c, c[-1] / 2.0)), len(v) - 1)])


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class _Track:
    def __init__(self, tid):
        self.id, self.obs, self.n = tid, deque(maxlen=MAX_OBS), 0   # obs: (x, z, w, conf, h, t)
        self.first_t = self.last_t = 0.0
        self.misses, self.stale, self.moved, self.moved_from, self.moved_at = 0, False, False, None, None
        self.pos = np.zeros(2)

    @property
    def confirmed(self):
        return self.n >= 2

    def update(self, x, z, w, conf, h, t):
        if self.n == 0:
            self.first_t = t
        self.obs.append((x, z, w, conf, h, t))
        self.n += 1
        self.last_t = max(self.last_t, t)
        self.misses, self.stale = 0, False
        self.refit()

    def refit(self):
        a = np.array([(o[0], o[1], o[2]) for o in self.obs])
        p = np.array([_wmedian(a[:, 0], a[:, 2]), _wmedian(a[:, 1], a[:, 2])])
        if len(a) >= 3:
            keep = np.hypot(a[:, 0] - p[0], a[:, 1] - p[1]) <= OUTLIER_M
            if keep.any() and not keep.all():
                a = a[keep]
                p = np.array([_wmedian(a[:, 0], a[:, 2]), _wmedian(a[:, 1], a[:, 2])])
        self.pos = p

    @property
    def conf(self):
        c = float(np.mean([o[3] for o in self.obs])) if self.obs else 0.0
        return c * (0.4 if self.stale else 1.0)

    @property
    def height(self):
        hs = [o[4] for o in self.obs if o[4] is not None]
        return float(np.median(hs)) if hs else None


class Memory:
    def __init__(self, max_range=3.5):
        self.max_range = max_range
        self.lock = threading.RLock()
        self.tracks = {}      # label -> [_Track]
        self.thumbs = {}      # label -> latest jpeg bytes
        self._tid = 0

    # ------------------------------------------------------------------ write
    def add(self, label, pos_xz, height=None, conf=1.0, cam_pos_xz=None, t=None, thumb=None):
        """Record a sighting. Returns False if rejected (out of range / bad input)."""
        label = str(label).lower().strip()
        if not label or pos_xz is None:
            return False
        t = time.time() if t is None else float(t)
        p = np.asarray(pos_xz, dtype=float).reshape(-1)[:2]
        if not np.all(np.isfinite(p)):
            return False
        d = 1.0
        if cam_pos_xz is not None:
            d = float(np.hypot(*(p - np.asarray(cam_pos_xz, dtype=float).reshape(-1)[:2])))
            if d > self.max_range:
                return False
        w = max(float(conf), 1e-3) / max(d, 0.5) ** 2
        with self.lock:
            tr = self.tracks.setdefault(label, [])
            tr[:] = [k for k in tr if k.confirmed or t - k.last_t < UNCONFIRMED_TTL_S]
            near = [(float(np.hypot(*(k.pos - p))), k) for k in tr]
            near = [x for x in near if x[0] <= ASSIGN_M]
            if near:
                k = min(near, key=lambda x: x[0])[1]
            else:
                self._tid += 1
                k = _Track(self._tid)
                tr.append(k)
                if len(tr) > MAX_TRACKS:
                    tr.remove(min((x for x in tr if x is not k), key=lambda x: (x.confirmed, x.last_t)))
            was = k.confirmed
            k.update(float(p[0]), float(p[1]), w, float(conf), height, t)
            if k.confirmed and not was:
                olds = [o for o in tr if o is not k and o.confirmed and o.last_t <= k.first_t]
                if olds:
                    o = max(olds, key=lambda x: x.last_t)
                    k.moved, k.moved_from, k.moved_at = True, o.pos.copy(), t
            if thumb is not None:
                self.thumbs[label] = thumb
        return True

    def negative(self, label, cam_pos_xz, heading_deg, half_fov_deg, now=None):
        """A scan of the visible area did not see `label`. Returns True if the track just went stale."""
        label = str(label).lower().strip()
        with self.lock:
            k = self._current(label)
            if k is None:
                return False
            d = k.pos - np.asarray(cam_pos_xz, dtype=float).reshape(-1)[:2]
            if float(np.hypot(*d)) > NEG_RANGE_M:
                return False
            rel = _wrap(math.degrees(math.atan2(d[0], -d[1])) - heading_deg)
            if abs(rel) > half_fov_deg:
                return False
            k.misses += 1
            if k.misses >= NEG_MISSES and not k.stale:
                k.stale = True
                return True
        return False

    # ------------------------------------------------------------------ read
    def _current(self, label):
        c = [k for k in self.tracks.get(label, []) if k.confirmed]
        return max(c, key=lambda k: k.last_t) if c else None

    def _entry(self, label, k):
        return {"label": label, "pos": k.pos.copy(), "height": k.height, "t": k.last_t, "conf": k.conf,
                "moved": k.moved, "n": k.n, "stale": k.stale,
                "moved_from": None if k.moved_from is None else k.moved_from.copy(), "moved_at": k.moved_at}

    def recall(self, target, synonyms=None):
        tw = _words(target)
        want = set(tw)
        for s in synonyms or []:
            want |= _words(s)
        if not want:
            return None
        with self.lock:
            best = None
            for label in self.tracks:
                lw = _words(label)
                k = self._current(label) if lw & want else None
                if k is None:
                    continue
                score = (lw == tw, len(lw & want), k.last_t)
                if best is None or score > best[0]:
                    best = (score, label, k)
            return self._entry(best[1], best[2]) if best else None

    def pins(self, now=None):
        now = time.time() if now is None else now
        out = []
        with self.lock:
            for label in self.tracks:
                k = self._current(label)
                if k is not None:
                    out.append({"label": label, "x": float(k.pos[0]), "z": float(k.pos[1]), "height": k.height,
                                "age_s": max(0.0, now - k.last_t), "conf": k.conf, "moved": k.moved,
                                "stale": k.stale})
        return out

    def snapshot(self):
        with self.lock:
            return {label: {"tracks": [{"x": float(k.pos[0]), "z": float(k.pos[1]), "n": k.n, "t": k.last_t,
                                        "conf": k.conf, "confirmed": k.confirmed, "stale": k.stale,
                                        "moved": k.moved} for k in tr]}
                    for label, tr in self.tracks.items()}

    def clear(self):
        with self.lock:
            self.tracks.clear()
            self.thumbs.clear()

    def describe(self, entry, heading_deg, pos_xz, now=None):
        return describe(entry, heading_deg, pos_xz, now)


# ------------------------------------------------------------------ speech
def surface_words(h):
    if h is None:
        return ""
    if h < 0.25:
        return "on the floor"
    if 0.5 <= h <= 1.2:
        return "on the table"
    if h > 1.2:
        return "on a shelf"
    return "about waist high"


def dir_words(rel):
    a, side = abs(rel), ("right" if rel > 0 else "left")
    if a < 15:
        return "straight ahead"
    if a < 45:
        return f"slightly to your {side}"
    if a < 135:
        return f"to your {side}"
    if a < 165:
        return f"behind you, to your {side}"
    return "right behind you"


def dist_words(d):
    if d < 0.8:
        return "within reach"
    if d < 2.0:
        r = math.floor(d * 2 + 0.5) / 2
        return {1.0: "about one meter away", 1.5: "about one and a half meters away"}.get(r, "about 2 meters away")
    return f"about {int(math.floor(d + 0.5))} meters away"


def describe(entry, heading_deg, pos_xz, now=None):
    """Spoken phrase (no follow-up question). heading_deg: + = turned right; pos_xz: camera/user position."""
    now = time.time() if now is None else now
    label = entry["label"]
    plural = label.endswith("s") and not label.endswith("ss")
    d = np.asarray(entry["pos"], dtype=float).reshape(-1)[:2] - np.asarray(pos_xz, dtype=float).reshape(-1)[:2]
    rel = _wrap(math.degrees(math.atan2(d[0], -d[1])) - heading_deg)
    where = " ".join(p for p in (surface_words(entry.get("height")), dir_words(rel)) if p)
    s = f"Your {label} {'are' if plural else 'is'} {where}, {dist_words(float(np.hypot(*d)))}."
    age = now - float(entry["t"])
    if age > 20:
        ago = f"{int(age)} seconds" if age <= 90 else f"{int(age / 60 + 0.5)} minutes"
        s += f" I saw {'them' if plural else 'it'} {ago} ago."
    return s
