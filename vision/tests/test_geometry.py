import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orin"))
import cv2
import numpy as np
import geometry as G

rng = np.random.default_rng(0)


def make_T(pitch_deg, heading_deg=0.0, pos=(0.0, 1.0, 0.0), roll_deg=0.0):
    """ARKit camera-to-world, column-major 16 floats."""
    p, h, r = map(math.radians, (pitch_deg, heading_deg, roll_deg))
    x, y, z = np.eye(3)
    Rr = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])   # roll about z
    Rp = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])   # pitch about x
    Rh = np.array([[math.cos(h), 0, -math.sin(h)], [0, 1, 0], [math.sin(h), 0, math.cos(h)]])   # yaw, + = right
    R = Rh @ Rp @ Rr
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, pos
    return M.T.reshape(-1).tolist()


def project(P, meta, W, H):
    fx, fy, cx, cy = G.cam(meta, W, H)
    T = G.T44(meta)
    c = T[:3, :3].T @ (np.asarray(P, float) - T[:3, 3])
    return cx + fx * c[0] / -c[2], cy - fy * c[1] / -c[2]


# ================================================================ 1. orientation
w, h = 640, 480
for k in (1, 2, 3):
    T = make_T(-10, 25, (0.3, 1.0, -0.2), roll_deg=7)
    meta = {"T": T, "intr": [510.0, 505.0, 322.0, 238.0], "tracking": True,
            "hands": [{"side": "right", "conf": 0.9, "bbox": [100, 50, 80, 120], "wrist": [140, 170],
                       "index": [150, 60], "thumb": [120, 90], "middle": [140, 55], "ring": [130, 58], "little": [110, 70]}]}
    for P in ([0.5, 0.3, -2.0], [-0.4, -0.3, -1.5], [0.1, 0.2, -3.0]):       # camera space
        Pw = G.T44(meta)[:3, :3] @ np.array(P) + G.T44(meta)[:3, 3]
        u, v = project(Pw, meta, w, h)
        assert 0 <= u < w and 0 <= v < h, (u, v)
        meta["fp"] = [[u, v, 1.5]]
        img = np.zeros((h, w), np.uint8)
        ui, vi = int(u), int(v)
        img[vi, ui] = 255
        img2, m2 = G.apply_rotation(img, meta, k)
        W2, H2 = m2["W"], m2["H"]
        u2, v2 = project(Pw, m2, W2, H2)
        ex = G.rot_point(u, v, w, h, k)
        assert abs(u2 - ex[0]) < 1e-6 and abs(v2 - ex[1]) < 1e-6, (k, u2, v2, ex)
        vv, uu = np.argwhere(img2 == 255)[0]
        assert abs(uu + 0.5 - G.rot_point(ui + 0.5, vi + 0.5, w, h, k)[0]) < 1e-6
        assert abs(vv + 0.5 - G.rot_point(ui + 0.5, vi + 0.5, w, h, k)[1]) < 1e-6
        assert abs(m2["fp"][0][0] - ex[0]) < 1e-6 and m2["fp"][0][2] == 1.5
    hp, pp, pos = G.pose(meta)
    hp2, pp2, pos2 = G.pose(m2)
    assert abs(hp - hp2) < 1e-9 and abs(pp - pp2) < 1e-9 and np.allclose(pos, pos2)
    hd, hd2 = meta["hands"][0], m2["hands"][0]
    assert list(hd2["wrist"]) == list(G.rot_point(140, 170, w, h, k))
    bx, by, bw, bh = hd2["bbox"]
    c1, c2 = G.rot_point(100, 50, w, h, k), G.rot_point(180, 170, w, h, k)
    assert (bx, by, bx + bw, by + bh) == (min(c1[0], c2[0]), min(c1[1], c2[1]), max(c1[0], c2[0]), max(c1[1], c2[1]))
    assert (W2, H2) == ((h, w) if k != 2 else (w, h)) and m2["rot_applied"] == k and img2.shape == (H2, W2)
    assert meta["intr"] == [510.0, 505.0, 322.0, 238.0]                # input meta untouched
print("OK rotation k=1,2,3: world points, image pixels, fp, hands, bbox match; heading/pitch unchanged")

img = np.zeros((480, 640, 3), np.uint8)
m = {"T": make_T(0), "rot": 1}
assert G.apply_rotation(img, m, 1)[1] is m and G.apply_rotation(img, {"T": make_T(0)}, 0)[0] is img
print("OK rotation no-op when meta has rot or k == 0")

# portrait held upright: camera x = world down, y = world right, z = backward
portrait = np.eye(4)
portrait[:3, 0], portrait[:3, 1], portrait[:3, 2], portrait[:3, 3] = [0, -1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]
Tp = portrait.T.reshape(-1).tolist()
assert G.pick_orientation(Tp, 0) == 1
assert G.pick_orientation(make_T(-10), 0) == 0 and G.pick_orientation(make_T(-10), 1) == 0
assert G.pick_orientation(make_T(-10, roll_deg=-50), 1) == 1          # hysteresis keeps portrait
assert G.pick_orientation(make_T(-10, roll_deg=-50), 0) == 0
assert G.pick_orientation(make_T(-10, roll_deg=-90), 0) == 1         # rolled -90 -> camera x points down
assert G.pick_orientation(make_T(-10, roll_deg=90), 0) == 3
assert G.pick_orientation(make_T(-10, roll_deg=180), 0) == 2
assert G.pick_orientation(make_T(-89), 1) == 1                        # looking at the floor: keep
_, m2 = G.apply_rotation(np.zeros((480, 640)), {"T": Tp, "intr": [500, 500, 320, 240]}, 1)
R2 = G.T44(m2)[:3, :3]
assert np.allclose(R2[:, 0], [1, 0, 0]) and np.allclose(R2[:, 1], [0, 1, 0])   # upright after rotation
print("OK pick_orientation: portrait -> k=1, hysteresis, rolls, flat-hold keeps current")


# ================================================================ scene renderer (z-depth)
def render(W, H, intr, pitch_deg, cam_h, boxes=(), wall=5.0):
    """z-depth of a floor at -cam_h, a wall `wall` m ahead, and axis-aligned boxes (r0,r1,y0,y1,f0,f1)
    in the camera's level frame (r = right, y = up from camera, f = forward)."""
    fx, fy, cx, cy = intr
    p = math.radians(pitch_deg)
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    a, b = (uu - cx) / fx, (cy - vv) / fy
    dr, dy, df = a, b * math.cos(p) + math.sin(p), math.cos(p) - b * math.sin(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(dy < 0, -cam_h / dy, np.inf)
        z = np.minimum(z, np.where(df > 0, wall / df, np.inf))
        for r0, r1, y0, y1, f0, f1 in boxes:
            tn, tf = np.full_like(a, -np.inf), np.full_like(a, np.inf)
            for dd, lo, hi in ((dr, r0, r1), (dy, y0, y1), (df, f0, f1)):
                t1, t2 = lo / dd, hi / dd
                tn, tf = np.maximum(tn, np.minimum(t1, t2)), np.minimum(tf, np.maximum(t1, t2))
            z = np.where((tn <= tf) & (tn > 0), np.minimum(z, tn), z)
    return np.minimum(z, 20).astype(np.float32)


def box_px(bx, intr, pitch_deg):
    fx, fy, cx, cy = intr
    p = math.radians(pitch_deg)
    us, vs = [], []
    for r in bx[0:2]:
        for y in bx[2:4]:
            for f in bx[4:6]:
                zc, up = y * math.sin(p) + f * math.cos(p), y * math.cos(p) - f * math.sin(p)
                us.append(cx + fx * r / zc)
                vs.append(cy - fy * up / zc)
    return min(us), min(vs), max(us), max(vs)


# ================================================================ 3. depth scale
INTR = [500.0, 500.0, 320.0, 240.0]
for pitch in (-8, -12):
    meta = {"T": make_T(pitch), "intr": INTR, "tracking": True}
    true = render(640, 480, INTR, pitch, 1.0)
    for k in (0.7, 1.4):
        for obstacle in (False, True):
            t2 = true.copy()
            if obstacle:     # 30% of the lower image, nearer than the floor
                t2[240:, 200:392] = np.minimum(t2[240:, 200:392], 0.9)
            ds = G.DepthScale()
            for _ in range(30):
                model = t2 * k * rng.normal(1, 0.03, t2.shape).astype(np.float32)
                r = ds.update_from_floor(model, meta, 1.0)
                assert r and r[1] == "floor"
            err = abs(ds.scale * k - 1)
            assert err < (0.08 if obstacle else 0.05), (pitch, k, obstacle, ds.scale)
            print(f"OK floor scale pitch {pitch} model x{k} obstacle={obstacle}: scale {ds.scale:.3f} "
                  f"(want {1 / k:.3f}, err {err * 100:.1f}%)")

# feature points path (with 10% outlier points and an obstacle)
true = render(640, 480, INTR, -10, 1.0, boxes=[(-0.4, 0.3, -1.0, -0.3, 1.2, 1.5)])
meta = {"T": make_T(-10), "intr": INTR, "tracking": True}
for k in (0.7, 1.4):
    ds = G.DepthScale()
    for _ in range(30):
        us, vs = rng.integers(0, 640, 120), rng.integers(0, 480, 120)
        zs = true[vs, us] * rng.normal(1, 0.02, 120)
        zs[:12] *= rng.uniform(0.3, 3.0, 12)
        meta["fp"] = [[float(u), float(v), float(z)] for u, v, z in zip(us, vs, zs)]
        model = true * k * rng.normal(1, 0.03, true.shape).astype(np.float32)
        r = ds.update(model, meta, 1.0)
        assert r and r[1] == "points"
    err = abs(ds.scale * k - 1)
    assert err < 0.05, (k, ds.scale)
    print(f"OK points scale model x{k}: scale {ds.scale:.3f} (want {1 / k:.3f}, err {err * 100:.1f}%)")
assert np.allclose(ds.apply(true), true * ds.scale)
ds.reset()
assert ds.scale == 1.0 and ds.source == "none"
assert ds.update(true, {"T": make_T(-10), "intr": INTR, "tracking": False, "fp": meta["fp"]}, 1.0) is None
meta.pop("fp")
assert ds.update(true * 0.1, meta, 1.0)[0] >= 0.4                      # clamped
print("OK depth scale reset / tracking-off / clamp")

# ================================================================ 4. target estimate (table scene)
CAM_H, TABLE, PITCH = 1.0, 0.75, -20
table = (-1.0, 1.0, -CAM_H, TABLE - CAM_H, 0.3, 2.6)
meta = {"T": make_T(PITCH), "intr": INTR, "tracking": True}
old_err, new_err = [], []
for (D, L, hgt, pad), lowres in [(c, lr) for lr in (False, True) for c in ((0.6, 0.15, 0.05, 0.1), (1.0, -0.3, 0.05, 0.1), (2.0, 0.5, 0.05, 0.1), (1.0, 0.2, 0.02, 0.3))]:
    item = (L - 0.04, L + 0.04, TABLE - CAM_H, TABLE - CAM_H + hgt, D - 0.04, D + 0.04)
    depth = render(640, 480, INTR, PITCH, CAM_H, boxes=[table, item])
    if lowres:       # phone LiDAR path: 128x96 upscaled like the server does - small items bleed into the background
        depth = cv2.resize(cv2.resize(depth, (128, 96), interpolation=cv2.INTER_AREA), (640, 480))
    bx0, by0, bx1, by1 = box_px(item, INTR, PITCH)
    px, py = pad * (bx1 - bx0), pad * (by1 - by0)             # detector boxes are loose
    box = (bx0 - px, by0 - py, bx1 + px, by1 + py)
    truth = math.hypot(D, L)
    est = G.target_estimate(box, meta, depth, CAM_H)
    x0, y0, x1, y1 = box
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    old = float(np.median(depth[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6]))   # server box_distance
    e_new, e_old = abs(est["dist"] - truth) / truth, abs(old - truth) / truth
    new_err.append(e_new)
    old_err.append(e_old)
    assert est["source"] == "depth" and e_new < 0.10, (D, est)
    assert abs(est["height"] - (TABLE + hgt / 2)) < 0.06, est
    assert np.hypot(*(est["pos_xz"] - [L, -D])) < 0.1 * truth, est
    print(f"OK {'128x96 ' if lowres else 'full-res'} {hgt * 100:.0f} cm item at {truth:.2f} m, box pad {pad:.0%}: new {est['dist']:.3f} ({e_new * 100:.1f}%), height {est['height']:.2f} | "
          f"old center patch {old:.3f} ({e_old * 100:.1f}%)")
assert np.mean(old_err) > np.mean(new_err)
print(f"OK new method mean error {np.mean(new_err) * 100:.1f}% vs old {np.mean(old_err) * 100:.1f}%")

# heading turned + off-axis: world position follows lock_target convention (x right, -z forward)
mh = {"T": make_T(PITCH, heading_deg=30, pos=(1.0, CAM_H, 2.0)), "intr": INTR, "tracking": True}
item = (0.26, 0.34, TABLE - CAM_H, TABLE - CAM_H + 0.05, 0.96, 1.04)
depth = render(640, 480, INTR, PITCH, CAM_H, boxes=[table, item])
est = G.target_estimate(box_px(item, INTR, PITCH), mh, depth, CAM_H)
h30 = math.radians(30)
want = np.array([1.0, 2.0]) + np.array([math.cos(h30) * 0.3 + math.sin(h30) * 1.0, math.sin(h30) * 0.3 - math.cos(h30) * 1.0])
assert np.hypot(*(est["pos_xz"] - want)) < 0.08, (est, want)
print("OK world position with heading 30 deg and off-axis item")

# planes: table anchor at 0.75 m, depth badly scaled -> plane wins; floor-guess without depth
D, L = 1.0, 0.0
item = (L - 0.04, L + 0.04, TABLE - CAM_H, TABLE - CAM_H + 0.05, D - 0.04, D + 0.04)
depth = render(640, 480, INTR, PITCH, CAM_H, boxes=[table, item])
box = box_px(item, INTR, PITCH)
A = np.eye(4)
A[:3, 3] = [0, TABLE, -1.4]
planes = [{"id": "t", "cls": "table", "T": A.T.reshape(-1).tolist(), "poly": [[-1, -1.1], [1, -1.1], [1, 1.1], [-1, 1.1]]},
          {"id": "f", "cls": "floor", "T": np.eye(4).reshape(-1).tolist(), "poly": [[-5, -5], [5, -5], [5, 5], [-5, 5]]}]
hit = G.plane_hit((box[0] + box[2]) / 2, box[1] + 0.75 * (box[3] - box[1]), meta, planes, 640, 480)
assert hit and hit["cls"] == "table", hit
est = G.target_estimate(box, meta, depth * 1.8, CAM_H, planes)
assert est["source"] == "plane" and abs(est["dist"] - D) < 0.1 and abs(est["height"] - TABLE) < 0.02, est
est = G.target_estimate(box, meta, depth, CAM_H, planes)
assert est["source"] == "depth", est                                    # agree -> keep depth
est = G.target_estimate(box, meta, None, CAM_H)
assert est["source"] == "floor-guess" and est["dist"] > 0, est
print(f"OK planes: table hit used when depth is off by 80% ({G.target_estimate(box, meta, depth * 1.8, CAM_H, planes)['dist']:.2f} m); "
      f"floor-guess without depth ({est['dist']:.2f} m, item is really {D} m on a table)")

# ================================================================ 5. reach gate
fires = []
for seed in range(40):
    r = np.random.default_rng(seed)
    gate, now, d, hit_at = G.ReachGate(), 0.0, 3.0, None
    tgt = {"box_t": 0.0}
    while d > 0.1:
        if now - tgt["box_t"] >= 0.4:
            tgt["box_t"] = now                                              # refresher re-finds every 0.4 s
        noisy = d + r.normal(0, 0.04) + (r.uniform(-1, 0) if r.random() < 0.05 else 0)
        if gate.update(tgt, noisy, (300, 200, 340, 240), 480, now):
            hit_at = round(d, 3)
            break
        now, d = now + 0.1, d - 0.1
    assert hit_at is not None and 0.6 <= hit_at <= 0.8 + 1e-9, (seed, hit_at)
    fires.append(hit_at)
print(f"OK reach gate approach 1 m/s: fires at {min(fires):.2f}-{max(fires):.2f} m over 40 noisy runs")

gate, tgt = G.ReachGate(), {"box_t": 0.0}
for i in range(40):
    now = i * 0.1
    tgt["box_t"] = now
    assert not gate.update(tgt, 0.2 if i == 20 else 1.5 + rng.normal(0, 0.03), (300, 200, 340, 240), 480, now)
print("OK reach gate ignores a single noisy reading")

gate, n = G.ReachGate(), 0
for i in range(10):
    n += gate.update({"box_t": i * 0.1}, 0.7, (300, 200, 340, 240), 480, i * 0.1)
assert n and not G.ReachGate().update({"box_t": 0}, 0.7, (300, 200, 340, 240), 480, 0)
gate = G.ReachGate()
assert not any(gate.update({"box_t": 0.0}, 0.5, (300, 200, 340, 240), 480, 2.0 + i * 0.1) for i in range(10))
gate = G.ReachGate()
assert [gate.update({"box_t": i}, 1.4, (200, 280, 400, 440), 480, i) for i in range(3)] == [False, False, True]
assert gate.should_exit({"box_t": 0}, 1.5, 3.5) and not gate.should_exit({"box_t": 0}, 1.0, 3.5)
assert not gate.should_exit({"box_t": 1}, 1.5, 3.5)
print("OK reach gate: standing at 0.7 fires, stale box never fires, big low box fires, should_exit")


# ================================================================ 6. obstacle columns
def server_obstacle_columns(depth, meta, cam_h=1.0):          # copy of navbelt_server.obstacle_columns (depth path)
    W, H, N = 640, 480, 64
    _, fy, _, cy = G.cam(meta, W, H)
    rows = np.arange(0, H, 4)
    p = math.radians(G.pose(meta)[1])
    up = math.sin(p) + (cy - rows) / fy * math.cos(p)
    d = depth[rows][:, :N * (W // N)].reshape(len(rows), N, W // N).min(axis=2)
    height = cam_h + d * up[:, None]
    ok = (height > 0.15) & (height < 1.9) & (d > 0.2)
    ahead = d * (math.cos(p) - (cy - rows) / fy * math.sin(p))[:, None]
    return np.percentile(np.where(ok, ahead, 9.9), 5, axis=0).astype(np.float32)


meta = {"T": make_T(-10), "intr": INTR, "tracking": True}
depth = render(640, 480, INTR, -10, 1.0, boxes=[(-0.6, -0.2, -1.0, 0.2, 1.5, 1.8), (0.4, 0.8, -1.0, -0.5, 2.5, 2.8)])
a, b = G.obstacle_columns_any(depth, meta, 1.0), server_obstacle_columns(depth, meta)
assert a.shape == (64,) and np.mean(np.abs(a - b) < 0.05) > 0.95, np.abs(a - b).max()
print(f"OK obstacle columns 640x480 match server ({np.mean(np.abs(a - b) < 0.05) * 100:.0f}% of columns within 5 cm)")
INTR_P = [500.0, 500.0, 240.0, 320.0]
mp = {"T": make_T(-10), "intr": INTR_P, "tracking": True}
depth = render(480, 640, INTR_P, -10, 1.0, boxes=[(-0.6, -0.2, -1.0, 0.2, 1.5, 1.8)])
c = G.obstacle_columns_any(depth, mp, 1.0)
lo, hi = int((240 - 500 * 0.6 / 1.5) * 64 / 480) + 1, int((240 - 500 * 0.2 / 1.5) * 64 / 480) - 1
assert c.shape == (64,) and np.all(np.abs(c[lo:hi + 1] - 1.5) < 0.05) and c[hi + 4:].min() > 4, c
print("OK obstacle columns 480x640 portrait: box at 1.5 m found in the right columns, floor not an obstacle")
print("ALL GEOMETRY TESTS PASSED")
