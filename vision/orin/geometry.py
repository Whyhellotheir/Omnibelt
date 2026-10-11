"""Pure geometry helpers for the Orin server: orientation, depth scale, target distance, reach gate.
Everything takes explicit arguments (no server globals). Frame size is always derived from the image/depth
shape or meta["W"]/meta["H"]; 640x480 is only the fallback default."""
import json
import math
from collections import deque

import cv2
import numpy as np

IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def frame_wh(meta, depth=None):
    if depth is not None:
        return depth.shape[1], depth.shape[0]
    return int((meta or {}).get("W", 640)), int((meta or {}).get("H", 480))


def T44(meta):
    return np.array((meta or {}).get("T") or IDENT, float).reshape(4, 4).T


def pose(meta):
    """heading (deg, + = turned right since start), pitch (deg, + = up), position (x, z) in meters."""
    T = T44(meta)
    fwd = -T[:3, 2]
    heading = math.degrees(math.atan2(fwd[0], -fwd[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
    return heading, pitch, np.array([T[0, 3], T[2, 3]])


def cam(meta, W, H, flip=False):
    fx, fy, cx, cy = (meta or {}).get("intr") or [500.0, 500.0, W / 2, H / 2]
    if flip:
        cx, cy = W - cx, H - cy
    return fx, fy, cx, cy


def px_bearing(u, meta, W, H, flip=False):
    fx, _, cx, _ = cam(meta, W, H, flip)
    return math.degrees(math.atan2(u - cx, fx))


def floor_dist(y_bottom, meta, cam_h, W, H, flip=False):
    """distance to something standing on the floor, from the bottom edge of its box (9.9 = unknown)."""
    _, fy, _, cy = cam(meta, W, H, flip)
    below = -pose(meta)[1] + math.degrees(math.atan2(y_bottom - cy, fy))
    return 9.9 if below <= 1 else min(9.9, cam_h / math.tan(math.radians(below)))


# ----------------------------------------------------------------- orientation
K_PHI = {0: 0.0, 1: -90.0, 2: 180.0, 3: 90.0}


def pick_orientation(T16, cur_k):
    """how far to rotate the sent image so it is upright: 0 none, 1 = 90 CW, 2 = 180, 3 = 90 CCW."""
    if not T16:
        return cur_k
    R = np.array(T16, float).reshape(4, 4).T
    ax, ay = R[1, 0], R[1, 1]               # world-up component of the camera x / y axes
    if math.hypot(ax, ay) < 0.5:            # looking straight up/down: roll is meaningless
        return cur_k
    phi = math.degrees(math.atan2(ax, ay))
    if abs(wrap(phi - K_PHI.get(cur_k, 0.0))) < 55:
        return cur_k
    return min(K_PHI, key=lambda k: abs(wrap(phi - K_PHI[k])))


def rot_point(x, y, w, h, k):
    if k == 1:
        return h - y, x
    if k == 3:
        return y, w - x
    if k == 2:
        return w - x, h - y
    return x, y


def rot_T(T16, k):
    c = np.array(T16, float).reshape(4, 4)      # rows = columns of the matrix = camera x, y, z axes, position
    c0, c1 = c[0].copy(), c[1].copy()
    c[0], c[1] = {1: (c1, -c0), 3: (-c1, c0), 2: (-c0, -c1)}[k]
    return c.reshape(-1).tolist()


def apply_rotation(img, meta, k):
    """rotate the image and everything in meta that is in image pixels / camera axes. No-op if the phone
    already rotated (meta has "rot") or k == 0. Works on depth maps too (any 2D/3D array)."""
    if k == 0 or meta is None or "rot" in meta:
        return img, meta
    h, w = img.shape[:2]
    img = cv2.rotate(img, {1: cv2.ROTATE_90_CLOCKWISE, 3: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180}[k])
    m = json.loads(json.dumps(meta))
    fx, fy, cx, cy = cam(m, w, h)
    m["intr"] = {1: [fy, fx, h - cy, cx], 3: [fy, fx, cy, w - cx], 2: [fx, fy, w - cx, h - cy]}[k]
    if m.get("T"):
        m["T"] = rot_T(m["T"], k)
    for hd in m.get("hands") or []:
        for key, val in list(hd.items()):
            if key == "bbox" and val and len(val) == 4:
                bx, by, bw, bh = val
                pts = [rot_point(x, y, w, h, k) for x, y in ((bx, by), (bx + bw, by + bh))]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                hd[key] = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            elif isinstance(val, list) and len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
                hd[key] = list(rot_point(val[0], val[1], w, h, k))
    if m.get("fp"):
        m["fp"] = [list(rot_point(p[0], p[1], w, h, k)) + list(p[2:]) for p in m["fp"]]
    m["W"], m["H"] = img.shape[1], img.shape[0]
    m["rot_applied"] = k
    return img, m


# ----------------------------------------------------------------- depth scale
class DepthScale:
    """corrects the monocular depth model's scale, from ARKit feature points or from the known floor height."""

    def __init__(self, alpha=0.1, lo=0.4, hi=2.5):
        self.alpha, self.lo, self.hi = alpha, lo, hi
        self.reset()

    def reset(self):
        self.log_s, self.scale, self.source, self.n = 0.0, 1.0, "none", 0

    def _feed(self, s, source):
        if not np.isfinite(s) or s <= 0:
            return None
        self.log_s += self.alpha * (math.log(s) - self.log_s)
        self.log_s = min(math.log(self.hi), max(math.log(self.lo), self.log_s))
        self.scale, self.source, self.n = math.exp(self.log_s), source, self.n + 1
        return self.scale, source

    def update(self, depth, meta, cam_h):
        return self.update_from_points(depth, meta) or self.update_from_floor(depth, meta, cam_h)

    def update_from_points(self, depth, meta):
        if depth is None or not (meta or {}).get("tracking", True) or not meta.get("fp"):
            return None
        H, W = depth.shape[:2]
        ratios = []
        for p in meta["fp"]:
            u, v, z = int(p[0]), int(p[1]), float(p[2])
            if not (0 <= u < W and 0 <= v < H and 0.1 < z < 8):
                continue
            patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
            patch = patch[patch > 0.05]
            if patch.size:
                ratios.append(z / float(np.median(patch)))
        return self._feed(float(np.median(ratios)), "points") if len(ratios) >= 12 else None

    def update_from_floor(self, depth, meta, cam_h):
        if depth is None or not (meta or {}).get("tracking", True):
            return None
        H, W = depth.shape[:2]
        _, fy, _, cy = cam(meta, W, H)
        p = math.radians(pose(meta)[1])
        rows = np.arange(2, H, 4)
        up = math.sin(p) + (cy - rows) / fy * math.cos(p)
        rows, up = rows[up < -0.12], up[up < -0.12]
        cols = np.arange(int(W * 0.2), int(W * 0.8), 4)
        if not len(rows):
            return None
        z_floor = np.repeat((-cam_h / up)[:, None], len(cols), axis=1)
        d = depth[rows][:, cols]
        ok = (z_floor < 4.5) & (d > 0.2)
        if ok.sum() < 300:
            return None
        # obstacles are nearer than the floor and inflate the ratio, so a low percentile skips them
        return self._feed(float(np.percentile(z_floor[ok] / d[ok], 30)), "floor")

    def apply(self, depth):
        return None if depth is None else depth * np.float32(self.scale)


# ----------------------------------------------------------------- target distance
def cam_ray(u, v, meta, W, H, flip=False):
    """world origin and direction of the ray through pixel (u, v), scaled so the step is 1 m of camera z-depth."""
    fx, fy, cx, cy = cam(meta, W, H, flip)
    T = T44(meta)
    return T[:3, 3], T[:3, :3] @ np.array([(u - cx) / fx, (cy - v) / fy, -1.0])


def plane_hit(u, v, meta, planes, W, H, flip=False):
    """nearest ARKit plane hit by the ray through (u, v): {"z", "point", "cls", "id"} or None."""
    o, d = cam_ray(u, v, meta, W, H, flip)
    best = None
    for pl in planes or []:
        A = np.array(pl["T"], float).reshape(4, 4).T
        n, p0 = A[:3, 1], A[:3, 3]
        den = float(n @ d)
        if abs(den) < 1e-6:
            continue
        t = float(n @ (p0 - o)) / den
        if not 0.2 < t < 8 or (best and t >= best["z"]):
            continue
        P = o + t * d
        loc = A[:3, :3].T @ (P - p0)
        poly = np.array(pl.get("poly") or [], np.float32).reshape(-1, 1, 2)
        if len(poly) >= 3 and cv2.pointPolygonTest(poly, (float(loc[0]), float(loc[2])), False) >= 0:
            best = {"z": t, "point": P, "cls": pl.get("cls", "none"), "id": pl.get("id")}
    return best


def floor_y(meta, planes, cam_h):
    ys = [np.array(p["T"], float).reshape(4, 4).T[1, 3] for p in planes or [] if p.get("cls") == "floor"]
    return min(ys) if ys else T44(meta)[1, 3] - cam_h


def from_world(P, meta, cam_h, source):
    T = T44(meta)
    dx, dz = P[0] - T[0, 3], P[2] - T[2, 3]
    return {"dist": float(math.hypot(dx, dz)), "pos_xz": np.array([P[0], P[2]], float),
            "height": float(P[1] - (T[1, 3] - cam_h)), "bearing": math.degrees(math.atan2(dx, -dz)),
            "source": source}


def target_estimate(box, meta, depth, cam_h, planes=None, flip=False):
    """level distance, world (x, z), height above floor and world bearing of the item in box (x0, y0, x1, y1).
    The depth point is projected through the full camera pose, which equals the pinhole formulas
    level = z*(cos p - t*sin p), side = z*(u-cx)/fx, range = hypot(level, side) when there is no roll."""
    W, H = frame_wh(meta, depth)
    x0, y0, x1, y1 = box
    u, v = (x0 + x1) / 2, y0 + 0.75 * (y1 - y0)          # lower middle: the item's near surface
    z = None
    if depth is not None:
        bw, bh = x1 - x0, y1 - y0
        xa, xb = int(max(0, x0 + 0.15 * bw)), int(min(W, x1 - 0.15 * bw + 1))
        ya, yb = int(max(0, y0 + 0.15 * bh)), int(min(H, y1 - 0.15 * bh + 1))
        patch = depth[ya:yb, xa:xb]
        patch = patch[patch > 0.2]
        if patch.size:
            z = float(np.percentile(patch, 25))          # near surface, not the table/wall behind it
    hit = plane_hit(u, v, meta, planes, W, H, flip) if planes else None
    if hit is not None:
        solid = hit["cls"] == "table" or hit["point"][1] - floor_y(meta, planes, cam_h) > 0.3
        if solid and (z is None or abs(hit["z"] - z) > 0.4 * hit["z"]):
            return from_world(hit["point"], meta, cam_h, "plane")
    if z is not None:
        o, d = cam_ray(u, v, meta, W, H, flip)
        return from_world(o + z * d, meta, cam_h, "depth")
    heading, _, pos = pose(meta)
    dist = floor_dist(y1, meta, cam_h, W, H, flip)
    dist = dist if dist < 9.9 else 1.5
    b = heading + px_bearing(u, meta, W, H, flip)
    tpos = pos + dist * np.array([math.sin(math.radians(b)), -math.cos(math.radians(b))])
    return {"dist": dist, "pos_xz": tpos, "height": 0.0, "bearing": b, "source": "floor-guess"}


# ----------------------------------------------------------------- reach trigger
def box_age(tgt, now):
    t = (tgt or {}).get("box_t")
    return math.inf if t is None else now - t

class ReachGate:
    """decides when to switch from walking (belt) to reaching (ring). Needs `need` ticks in a row with a fresh box
    and either a close robust distance or a big box at the bottom of the frame. The distance is a median of the
    last 5 readings moved forward along a robust (Theil-Sen) closing speed, plus the distance covered while the
    streak confirms - otherwise a walking user is ~0.4 m past the threshold before it fires."""

    def __init__(self, reach_m=0.75, need=3, fresh_s=1.0, n=5, n_speed=12):
        self.reach_m, self.need, self.fresh_s, self.n = reach_m, need, fresh_s, n
        self.hist = deque(maxlen=n_speed)       # speed uses a longer window than the median: less noise
        self.streak = 0

    def reset(self):
        self.hist.clear()
        self.streak = 0

    def robust_dist(self, now):
        if not self.hist:
            return None
        t = np.array([h[0] for h in self.hist])
        d = np.array([h[1] for h in self.hist])
        sl = [(d[j] - d[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t)) if t[j] - t[i] > 0.05]
        closing = min(1.2, max(0.0, -float(np.median(sl)))) if sl else 0.0
        dt = min(0.3, float(np.median(np.diff(t)))) if len(t) > 1 else 0.1
        t, d = t[-self.n:], d[-self.n:]
        return float(np.median(d - closing * (now - t))) - closing * dt * (self.need - 1)

    def update(self, tgt, dist, box, frame_h, now):
        if dist is not None:
            self.hist.append((now, float(dist)))
        fresh = bool(tgt) and box is not None and box_age(tgt, now) <= self.fresh_s
        near = False
        if fresh:
            rd = self.robust_dist(now)
            big = box[3] > 0.88 * frame_h and box[3] - box[1] > 0.30 * frame_h
            near = big or (rd is not None and rd < self.reach_m)
        self.streak = self.streak + 1 if near else 0
        return self.streak >= self.need

    def should_exit(self, tgt, dist, now):
        stale = not tgt or box_age(tgt, now) > 3.0
        return stale and dist is not None and dist > 1.2


# ----------------------------------------------------------------- obstacles
def obstacle_columns_any(depth, meta, cam_h, n_cols=64, obst=(0.15, 1.9), flip=False):
    """nearest obstacle level distance in each of n_cols image columns, for any frame size."""
    H, W = depth.shape[:2]
    _, fy, _, cy = cam(meta, W, H, flip)
    p = math.radians(pose(meta)[1])
    rows = np.arange(0, H, 4)
    t = (cy - rows) / fy
    xs = ((np.arange(n_cols * 8) + 0.5) * W / (n_cols * 8)).astype(int)
    d = depth[rows][:, xs].reshape(len(rows), n_cols, 8).min(axis=2)
    height = cam_h + d * (math.sin(p) + t * math.cos(p))[:, None]
    ok = (height > obst[0]) & (height < obst[1]) & (d > 0.2)
    ahead = d * (math.cos(p) - t * math.sin(p))[:, None]
    return np.percentile(np.where(ok, ahead, 9.9), 5, axis=0).astype(np.float32)
