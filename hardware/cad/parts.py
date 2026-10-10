"""
Haptic Nav Belt — parametric 3D parts, v2 (all units mm)
=========================================================
Run `python parts.py` to regenerate every STL into ./stl

Coordinate system for every belt / chest part, AS WORN:
  X = along the belt (or along the phone's length)
  Y = up
  Z = out from the body (Z = 0 is the skin side)
Belt/chest parts are exported standing on their X = 0 end, which is how they
print: belt tunnels and walls run vertically, so they need NO supports.
Ring, wrist pod, lids and shims are exported flat, also support-free.
"""
import os, math
import numpy as np
import trimesh
from manifold3d import Manifold as M, CrossSection as CS, JoinType, set_circular_segments

set_circular_segments(64)

# ======================= PARAMETERS — edit these =======================
BELT_W   = 50.8    # belt width (2")
BELT_T   = 4.4     # tunnel height. Fits belts up to ~4.2 mm; shims fill the gap for thinner belts
TUN_Y    = 26.3    # half-width of every belt tunnel (52.6 mm wide)

MOTOR_D, MOTOR_H = 10.6, 2.7        # 10x3 mm coin motor -> sits 0.3 mm proud of the skin face

ORIN_L, ORIN_W = 79.0, 100.0        # Jetson Orin Nano Super dev kit footprint (NVIDIA: 100 x 79 x 21)
BB_L, BB_W     = 86.0, 58.0         # cavity for a 400-tie breadboard (~84 x 55 x 8.5)
BATT_SPAN      = 71.5               # clear width for the power bank (INIU slim = 69 mm wide)

PHONES = {                          # length, width, thickness (no case)
    "iphone17":    (149.6, 71.5, 7.95),
    "iphone17pro": (150.0, 71.9, 8.75),
}
PHONE_CLR = 0.6
TILT_DEG  = 12.0                    # camera points this far down to see the floor

RING_SIZES = [7, 8, 9, 10, 11, 12]  # US ring sizes to export
# =======================================================================

TX = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], float)  # 2D (Y,Z) profile, extruded along X


def box(x0, x1, y0, y1, z0, z1):
    return M.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def rrect(y0, y1, z0, z1, r):
    w, h = y1 - y0, z1 - z0
    r = max(0.01, min(r, w / 2 - 0.05, h / 2 - 0.05))
    return CS.square([w - 2 * r, h - 2 * r], True).offset(r, JoinType.Round).translate([(y0 + y1) / 2, (z0 + z1) / 2])


def ext_x(cs, x0, x1):
    return cs.extrude(x1 - x0).transform(TX).translate([x0, 0, 0])


def body_x(cs, x0, x1, ch=0.8):
    """Rounded profile extruded along X with chamfered ends (profile must be convex)."""
    inner = cs.offset(-ch, JoinType.Round)
    return M.batch_hull([ext_x(inner, x0, x0 + 0.01), ext_x(cs, x0 + ch, x1 - ch), ext_x(inner, x1 - 0.01, x1)])


def cyl_z(d, h, x, y, z0):
    return M.cylinder(h, d / 2, d / 2).translate([x, y, z0])


def tunnel(x0, x1, z0):
    return box(x0 - 1, x1 + 1, -TUN_Y, TUN_Y, z0, z0 + BELT_T)


def rounded_box_flat(x0, x1, y0, y1, z0, z1, r):
    """Box rounded on its vertical edges (for parts printed flat, Z up)."""
    w, h = x1 - x0, y1 - y0
    cs = CS.square([w - 2 * r, h - 2 * r], True).offset(r, JoinType.Round).translate([(x0 + x1) / 2, (y0 + y1) / 2])
    return cs.extrude(z1 - z0).translate([0, 0, z0])


# ---------------------------------------------------------------- 1. MOTOR POD
def motor_pod(front=False):
    L, Yh = 26.0, TUN_Y + 3.3
    skin, top = 4.0, 2.0
    T = skin + BELT_T + top
    m = body_x(rrect(-Yh, Yh, 0, T, 3.0), 0, L)
    m -= tunnel(0, L, skin)
    m -= cyl_z(MOTOR_D, MOTOR_H + 0.02, L / 2, 0, -0.01)                     # motor pocket (skin side)
    m -= box(L / 2 - 1.2, L / 2 + 1.2, 0, Yh + 1, -0.01, 1.6)           # lead groove to top edge
    m -= box(L / 2 - 1.2, L / 2 + 1.2, Yh - 1.5, Yh + 1, -0.01, T + 0.01)  # ...and over the top edge
    if front:                                                           # tactile ridge = "front"
        m += box(0.8, L - 0.8, -1.2, 1.2, T - 0.5, T + 1.2)
    return m


# ---------------------------------------------------------------- 2. BELT SHIMS
def belt_shims():
    """Slide one in beside the belt to take up slack. 1 notch = 1.2 mm, 2 notches = 2.4 mm."""
    parts = []
    for i, t in enumerate((1.2, 2.4)):
        s = rounded_box_flat(0, 24, 0, 50, 0, t, 2.0).translate([i * 30, 0, 0])
        for k in range(i + 1):
            s -= box(i * 30 + 4 + k * 5, i * 30 + 6.5 + k * 5, 47, 51, -1, t + 1)
        parts.append(s)
    return parts[0] + parts[1]


# ---------------------------------------------------------------- 3. ORIN TRAY
def orin_tray():
    L = ORIN_L + 1.6
    Yb = (ORIN_W + 1.6) / 2
    Yo = Yb + 2.6
    zf = 2.0 + BELT_T + 2.4
    H = zf + 6.0
    m = body_x(rrect(-Yo, Yo, 0, H, 3.0), 0, L)
    m -= box(-1, L + 1, -Yb, Yb, zf, H + 1)        # board pocket, open both ends for ports + air
    m -= tunnel(0, L, 2.0)
    for xa, xb in ((-1.0, 27.5), (L - 27.5, L + 1.0)):   # two 1" velcro straps, run off the ends
        m -= box(xa, xb, Yo - 1.0, Yo + 1, -1, H + 1)
        m -= box(xa, xb, -Yo - 1, -Yo + 1.0, -1, H + 1)
        m -= box(xa, xb, -Yo - 1, Yo + 1, -0.01, 0.8)
    return m


# ---------------------------------------------------------------- 4. BATTERY CRADLE
def battery_cradle():
    rail, ramp = 2.5, 5.5
    L = rail + BATT_SPAN + rail + ramp
    y0, y1 = -46.0, 32.0
    zf = 2.0 + BELT_T + 2.4
    rh = 5.0
    m = body_x(rrect(y0, y1, 0, zf, 2.5), 0, L)
    m -= tunnel(0, L, 2.0)
    m += box(0.0, rail, y0 + 0.6, y1 - 0.6, zf - 0.5, zf + rh)                       # bed-end rail
    m += M.batch_hull([box(L - rail, L, y0 + 0.6, y1 - 0.6, zf - 0.5, zf + rh),   # top-end rail,
                       box(L - rail - ramp, L, y0 + 0.6, y1 - 0.6, zf - 0.5, zf - 0.49)])  # 45° underside
    m += box(0.0, L, y0 + 0.6, y0 + 3.6, zf - 0.5, zf + 6.0)                    # bottom lip
    for yc in (-28.0, 16.0):                                                         # strap notches
        m -= box(-1, L + 1, yc - 13.5, yc + 13.5, zf + rh - 1.0, zf + 7)
        m -= box(L - 1.2, L + 1, yc - 13.5, yc + 13.5, -1, zf + 7)
        m -= box(-1, L + 1, yc - 13.5, yc + 13.5, -0.01, 0.8)
    return m


# ---------------------------------------------------------------- 5. BELT HUB + LID
HUB_WALL = 2.4
def belt_hub():
    L = HUB_WALL + BB_W
    Yc = BB_L / 2
    Yo = Yc + HUB_WALL
    zf = 2.0 + BELT_T + 2.4
    H = zf + 22.0
    m = body_x(rrect(-Yo, Yo, 0, H, 3.0), 0, L)
    m -= box(HUB_WALL, L + 1, -Yc, Yc, zf, H + 1)                  # cavity, open at the +X end
    m -= tunnel(0, L, 2.0)
    m -= box(-1, HUB_WALL + 1, -10, 10, zf + 8.5, H + 1)           # wire slot, closed end
    xm = HUB_WALL + BB_W / 2
    m -= box(xm - 9, xm + 9, -Yo - 1, Yo + 1, zf + 8.0, H + 1)     # ESP32 USB slots, both walls
    return m


def hub_lid():
    L = HUB_WALL + BB_W
    Yc, Yo = BB_L / 2, BB_L / 2 + HUB_WALL
    t = 2.0
    lid = rounded_box_flat(0, L, -Yo, Yo, 0, t, 3.0)
    lid = M.batch_hull([rounded_box_flat(0.6, L - 0.6, -Yo + 0.6, Yo - 0.6, 0, 0.6, 2.4),
                        rounded_box_flat(0, L, -Yo, Yo, 0.6, t, 3.0)])            # chamfered top edge
    for s in (-1, 1):                                                              # press-fit rails
        ya, yb = sorted([s * (Yc - 0.25), s * (Yc - 0.25 - 1.6)])
        lid += box(HUB_WALL + 1.0, L - 1.0, ya, yb, t, t + 3.0)
    for i in range(5):                                                             # vent slots
        x = 10 + i * 9.5
        lid -= rounded_box_flat(x, x + 3.5, -30, 30, -1, t + 1, 1.7)
    return lid


# ---------------------------------------------------------------- 6. PHONE CHEST CRADLE
def phone_cradle(phone, tilt=None, back=2.0, motor_x=None):
    """Chest version: defaults. Belt version: tilt=8, back=4.0, motor_x=37 (M1 motor pocket on the back)."""
    tilt = TILT_DEG if tilt is None else tilt
    PL, PW, PT = PHONES[phone]
    wall, base, lip, lip_t = 2.4, 3.0, 2.5, 1.8
    claw, cam_wall, cam_wall_h = 18.0, 18.0, 2.5
    Lc, Wc, Tc = PL + PHONE_CLR, PW + 2 * PHONE_CLR, PT + PHONE_CLR
    X1 = wall + Lc
    Yc = Wc / 2 + wall
    Zt = base + Tc + lip_t
    xm = wall + Lc / 2

    c = body_x(rrect(-Yc, Yc, 0, Zt, 2.0), 0, X1, ch=0.6)
    c -= box(wall, X1 + 1, -Wc / 2, Wc / 2, base, base + Tc)                       # phone pocket
    c -= box(wall, X1 + 1, -Wc / 2 + lip, Wc / 2 - lip, base + Tc - 0.1, Zt + 1)    # open back window
    for s in (-1, 1):
        ya, yb = sorted([s * (Wc / 2 - lip - 0.01), s * (Yc + 1)])
        c -= box(wall + claw, X1 + 1, ya, yb, base, Zt + 1)                        # sides free of buttons
        # low (2.5 mm) side guides at the camera end, with a 45° ramp so they print
        ga, gb = sorted([s * Wc / 2, s * Yc])
        c += M.batch_hull([box(X1 - cam_wall, X1 - 0.6, ga, gb, base - 0.01, base + cam_wall_h),
                           box(X1 - cam_wall - cam_wall_h, X1 - 0.6, ga, gb, base - 0.01, base)])
        # strap posts in the middle, 3 mm clear of the phone's side buttons
        pa, pb = sorted([s * (Wc / 2 + 3.0), s * (Wc / 2 + 5.4)])
        ea, eb = sorted([s * (Yc - 0.5), s * (Wc / 2 + 5.4)])
        post = box(xm - 16, xm + 16, pa, pb, 0, base + Tc + 0.5) + box(xm - 16, xm + 16, ea, eb, 0, base)
        sa, sb = sorted([s * (Yc - 0.5), s * Yc])
        c += M.batch_hull([post, box(xm - 22, xm + 16, sa, sb, 0, base + Tc + 0.5)])
    c -= box(-1, wall + 1, -23, 23, base + 1.0, Zt + 1)                            # USB-C + speaker opening

    # chest-strap mount (sits against the chest)
    mY, mT = TUN_Y + 3.0, back + BELT_T + 2.5
    mount = body_x(rrect(-mY, mY, 0, mT, 2.5), 0, X1, ch=0.6)
    tun = tunnel(0, X1, back)
    if motor_x is not None:                                   # M1 motor on the skin side (belt version)
        tun = tun + cyl_z(MOTOR_D, MOTOR_H + 0.02, motor_x, 0, -0.01)
        tun = tun + box(motor_x - 1.2, motor_x + 1.2, 0, mY + 1, -0.01, 1.6)
        tun = tun + box(motor_x - 1.2, motor_x + 1.2, mY - 1.5, mY + 1, -0.01, mT)

    def place(m):
        m = m.rotate([tilt, 0, 0])
        lo = box(0, X1, -Yc, Yc, 0, base).rotate([tilt, 0, 0]).bounding_box()[2]
        return m.translate([0, 0, mT + 1.0 - lo])
    cr = place(c)
    plate = place(box(0.0, X1, -Yc + 1.5, Yc - 1.5, 0, base))
    wedge = M.batch_hull([plate, box(0.0, X1, -mY + 1, mY - 1, mT - 0.5, mT)])
    return (mount + wedge + cr) - tun, place


# ---------------------------------------------------------------- 7. RING + CAP (optional)
def ring_inner_d(us_size):
    return 11.63 + 0.8128 * us_size


def ring(us_size):
    """Printed flat. The 'bezel' holds a coin motor; it extends toward the wrist."""
    ri = ring_inner_d(us_size) / 2
    band_t, band_h = 1.8, 6.5
    bez_x, bez_z = 14.0, 13.0
    floor_y = ri + 1.0
    top_y = floor_y + 3.4
    band = M.cylinder(band_h, ri + band_t, ri + band_t) - M.cylinder(band_h + 2, ri, ri).translate([0, 0, -1])
    bez = (CS.square([bez_x - 4, top_y - (ri - 2.0) - 4], True).offset(2.0, JoinType.Round)
           .translate([0, (top_y + ri - 2.0) / 2]).extrude(bez_z))
    r = band + bez
    r -= M.cylinder(40, ri, ri).translate([0, 0, -10])                          # finger
    r -= box(-5.4, 5.4, floor_y, top_y + 1, 1.0, 11.8)                          # motor pocket
    r -= box(-1.5, 1.5, floor_y + 0.5, top_y + 1, 11.0, bez_z + 1)              # lead exit (wrist side)
    return r


def ring_cap():
    """Print in translucent filament; an LED under it glows with the motor."""
    ix, iz, skirt, wall, top, space = 14.5, 13.5, 2.5, 1.2, 1.2, 3.5
    ox, oz = ix + 2 * wall, iz + 2 * wall
    H = top + space + skirt
    cap = M.batch_hull([rounded_box_flat(-ox / 2 + 0.8, ox / 2 - 0.8, -oz / 2 + 0.8, oz / 2 - 0.8, 0, 0.8, 2.2),
                        rounded_box_flat(-ox / 2, ox / 2, -oz / 2, oz / 2, 0.8, H, 3.0)])
    cap -= rounded_box_flat(-ix / 2, ix / 2, -iz / 2, iz / 2, top, H + 1, 2.0)
    cap -= box(-1.6, 1.6, iz / 2 - 1, oz / 2 + 1, H - skirt - 1.5, H + 1)       # lead exit
    return cap


# ---------------------------------------------------------------- 8. WRIST POD + LID (optional)
POD_IN = (46.0, 28.0, 12.0)   # fits a Seeed XIAO ESP32C3 + 402030 LiPo + driver transistor
def wrist_pod():
    ix, iy, ih = POD_IN
    w, skin, tun, floor = 1.8, 1.2, 2.6, 1.6
    ox, oy = ix + 2 * w, iy + 2 * w
    z_floor = skin + tun
    H = z_floor + floor + ih
    p = rounded_box_flat(-ox / 2, ox / 2, -oy / 2, oy / 2, 0, H, 4.0)
    p -= rounded_box_flat(-ix / 2, ix / 2, -iy / 2, iy / 2, z_floor + floor, H + 1, 2.4)
    p -= box(-13.5, 13.5, -oy / 2 - 1, oy / 2 + 1, skin, skin + tun)            # 1" wrist strap tunnel
    p -= box(-ox / 2 - 1, -ix / 2 + 0.5, -6, 6, z_floor + floor + 3.0, z_floor + floor + 9.5)  # USB-C
    p -= cyl_z(4.0, 1, 0, 0, 0).rotate([0, 90, 0]).translate([ix / 2 - 0.5, 0, z_floor + floor + 6]) \
        if False else box(ix / 2 - 0.5, ox / 2 + 1, -2.2, 2.2, z_floor + floor + 4.0, z_floor + floor + 8.4)  # ring leads
    return p


def wrist_pod_lid():
    ix, iy, _ = POD_IN
    w = 1.8
    ox, oy = ix + 2 * w, iy + 2 * w
    lid = M.batch_hull([rounded_box_flat(-ox / 2 + 0.6, ox / 2 - 0.6, -oy / 2 + 0.6, oy / 2 - 0.6, 0, 0.6, 3.4),
                        rounded_box_flat(-ox / 2, ox / 2, -oy / 2, oy / 2, 0.6, 1.8, 4.0)])
    rim = rounded_box_flat(-ix / 2 + 0.2, ix / 2 - 0.2, -iy / 2 + 0.2, iy / 2 - 0.2, 1.8, 4.3, 2.2)
    rim -= rounded_box_flat(-ix / 2 + 1.6, ix / 2 - 1.6, -iy / 2 + 1.6, iy / 2 - 1.6, 1.0, 5.0, 1.0)
    return lid + rim


# ---------------------------------------------------------------- export helpers
def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


def stand_on_x0(m):
    m = m.rotate([0, -90, 0])
    b = m.bounding_box()
    return m.translate([-b[0], -b[1], -b[2]])


def on_bed(m):
    b = m.bounding_box()
    return m.translate([-b[0], -b[1], -b[2]])


def all_parts():
    """name -> (manifold in print orientation, qty, folder)"""
    P = {
        "motor_pod":            (stand_on_x0(motor_pod()), 6, "belt"),
        "motor_pod_FRONT":      (stand_on_x0(motor_pod(front=True)), 1, "belt"),
        "orin_tray":            (stand_on_x0(orin_tray()), 1, "belt"),
        "battery_cradle":       (stand_on_x0(battery_cradle()), 1, "belt"),
        "belt_hub":             (stand_on_x0(belt_hub()), 1, "belt"),
        "belt_hub_lid":         (on_bed(hub_lid()), 1, "belt"),
        "belt_shims":           (on_bed(belt_shims()), 1, "belt"),
        "phone_cradle_iphone17":    (stand_on_x0(phone_cradle("iphone17")[0]), 1, "chest"),
        "phone_cradle_iphone17pro": (stand_on_x0(phone_cradle("iphone17pro")[0]), 1, "chest"),
        "phone_BELT_cradle_iphone17": (stand_on_x0(phone_cradle("iphone17", tilt=8.0, back=4.0, motor_x=37.0)[0]), 1, "belt"),
        "phone_holder_UNIVERSAL_belt": (holder_print_orientation(universal_phone_holder()[0]), 1, "belt"),
        "ring_cap_TRANSLUCENT": (on_bed(ring_cap()), 1, "ring_optional"),
        "wrist_pod":            (on_bed(wrist_pod()), 1, "ring_optional"),
        "wrist_pod_lid":        (on_bed(wrist_pod_lid()), 1, "ring_optional"),
    }
    for s in RING_SIZES:
        P[f"ring_US{s}"] = (on_bed(ring(s)), 1, "ring_optional/ring_sizes")
    return P


if __name__ == "__main__":
    for name, (m, qty, folder) in all_parts().items():
        os.makedirs(f"stl/{folder}", exist_ok=True)
        t = to_trimesh(m)
        t.export(f"stl/{folder}/{name}.stl")
        ext = t.bounds[1] - t.bounds[0]
        print(f"{name:28s} x{qty}  watertight={t.is_watertight}  size={np.round(ext,1)}")


# ---------------------------------------------------------------- FIT COUPONS
def coupons():
    """Short slices of the big prints (bed end only) to check fit in minutes before committing hours."""
    def slice_(m, h):
        b = m.bounding_box()
        return m ^ box(b[0] - 1, b[3] + 1, b[1] - 1, b[4] + 1, -1, h)
    return {
        "COUPON_phone_cradle_iphone17_port_end": slice_(stand_on_x0(phone_cradle("iphone17")[0]), 30.0),
        "COUPON_orin_tray_slice":                slice_(stand_on_x0(orin_tray()), 10.0),
        "COUPON_belt_hub_slice":                 slice_(stand_on_x0(belt_hub()), 14.0),
    }


if __name__ == "__main__":
    os.makedirs("stl/fit_coupons", exist_ok=True)
    for name, m in coupons().items():
        t = to_trimesh(m)
        t.export(f"stl/fit_coupons/{name}.stl")
        print(f"{name:40s} watertight={t.is_watertight}  size={np.round(t.bounds[1]-t.bounds[0],1)}  ~{t.volume/1000*1.24*0.6:.0f} g")


# ---------------------------------------------------------------- UNIVERSAL BELT PHONE HOLDER
# Fits phones about 124-175 mm long, up to ~84 mm wide and ~15 mm thick (case included), landscape.
# The phone rests on two end pads (near its corners, clear of side buttons), screen against the plate,
# and one 1" velcro strap across the MIDDLE of its back holds it (cameras sit at an end, so they're clear).
# Threads onto the 2" belt; M1 motor pocket on the back; tilted down so the camera sees the floor.
UNI = dict(tilt=8.0, plate_x=48.0, y0=-56.0, y1=61.0, base=3.0, pad_x0=58.0, pad_x1=78.0,
           pad_top=-34.0, pad_depth=16.0, slot_w=27.0, slot_h=4.0, slot_lo=-50.0, slot_hi=51.0)

def universal_phone_holder():
    u = UNI
    px, y0, y1, base = u["plate_x"], u["y0"], u["y1"], u["base"]
    # --- front part, built flat (backplate in the XY plane, phone side = +Z) ---
    plate = rounded_box_flat(-px, px, y0, y1, 0, base, 6.0)
    arms = M()
    for s in (-1, 1):
        xa, xb = sorted([s * (px - 4), s * u["pad_x1"]])
        arms += rounded_box_flat(xa, xb, y0, u["pad_top"] - 2.0, 0, base, 3.0)          # arm in the plate plane
        pa, pb = sorted([s * u["pad_x0"], s * u["pad_x1"]])
        pad = M.batch_hull([box(pa, pb, y0, u["pad_top"] - 1.0, 0, base + u["pad_depth"]),
                            box(pa, pb, y0, u["pad_top"], 0, base + u["pad_depth"] - 1.0)])   # rounded-ish top edge
        arms += pad
    front = plate + arms
    for ylo in (u["slot_lo"], u["slot_hi"]):                                             # strap slots
        front -= rounded_box_flat(-u["slot_w"] / 2, u["slot_w"] / 2, ylo, ylo + u["slot_h"], -1, base + 1, 1.5)

    # --- mount (sits against the body, axis-aligned when worn) ---
    back = 4.0
    mY, mT = TUN_Y + 3.0, back + BELT_T + 2.5
    mount = body_x(rrect(-mY, mY, 0, mT, 2.5), -px, px, ch=0.6)

    def place(m):                                                        # tilt so the camera looks down
        m = m.rotate([u["tilt"], 0, 0])
        lo = box(-px, px, -30, 30, 0, base).rotate([u["tilt"], 0, 0]).bounding_box()[2]
        return m.translate([0, 0, mT + 1.0 - lo])
    fr = place(front)
    strip = place(box(-px + 1, px - 1, -30, 30, 0, 0.5))                 # back face of the plate, middle band
    sliver = place(box(-px + 1, px - 1, -44.5, -44.0, 0, 0.5))             # 45 deg gusset so it prints upright
    wedge = M.batch_hull([strip, sliver,
                          box(-px + 1, px - 1, -mY + 1, mY - 1, mT - 0.5, mT),
                          box(-px + 1, px - 1, -mY, -mY + 0.5, 0, mT)])
    part = mount + wedge + fr
    part -= tunnel(-px, px, back)
    part -= cyl_z(MOTOR_D, MOTOR_H + 0.02, 0, 0, -0.01)                  # M1 motor pocket (skin side)
    part -= box(-1.2, 1.2, 0, mY + 1, -0.01, 1.6)
    part -= box(-1.2, 1.2, mY - 1.5, mY + 1, -0.01, mT)
    return part, place


def holder_print_orientation(m):
    """Stand it upright on the bottom edge of the plate and pads (backplate vertical): no supports."""
    m = m.rotate([90.0 - UNI["tilt"], 0, 0])
    return on_bed(m)


# ---------------------------------------------------------------- SWAP RING SYSTEM (all sizes, one motor capsule)
# Each ring has a dovetail socket on top. The motor lives in a small capsule that slides into ANY ring
# from the fingertip side and stops against the wrist-side end, so the wire's pull holds it in place.
# The motor face sits flush with the capsule's underside and touches the finger through the window.
SW = dict(band_t=2.0, band_h=6.0, H=14.0, a0=6.8, a1=5.8, dt_h=5.0, neck=0.3, plate_t=2.0, plate_w=7.6,
          clr=0.2, floor=1.2, frame_w=9.6)

def swap_capsule():
    """capsule frame: x tangential, y = 0 at the face that touches the finger, z = along the finger (0 = wrist end)."""
    s = SW
    L = s["H"] - s["floor"] - 0.3
    dt = CS([[[-s["a0"], 0], [s["a0"], 0], [s["a1"], s["dt_h"]], [-s["a1"], s["dt_h"]]]]).extrude(L)
    neck = box(-s["a1"], s["a1"], s["dt_h"] - 0.01, s["dt_h"] + s["neck"], 0, L)
    yp = s["dt_h"] + s["neck"]
    plate = rounded_box_flat(-s["plate_w"], s["plate_w"], yp, yp + s["plate_t"], 0, L, 1.5) if False else \
        CS.square([2 * s["plate_w"] - 2, s["plate_t"] - 2 + 0.01], True).offset(1.0, JoinType.Round).translate([0, yp + s["plate_t"] / 2]).extrude(L)
    cap = dt + neck + plate
    cap -= M.cylinder(2.6, MOTOR_D / 2, MOTOR_D / 2).rotate([-90, 0, 0]).translate([0, -0.01, L / 2])   # motor pocket, open to the finger: 3 mm motor sticks out 0.4 mm
    cap -= box(-1.3, 1.3, -0.01, 1.8, -1, L / 2)                                  # lead groove to the wrist end
    return cap, L


def swap_ring(us_size):
    """ring frame: finger axis = z, socket on top (+y). z = 0 is the wrist end (the stop)."""
    s = SW
    ri = ring_inner_d(us_size) / 2
    yb = ri + 0.3                                       # capsule underside (touches the finger)
    band = M.cylinder(s["band_h"], ri + s["band_t"], ri + s["band_t"]) - M.cylinder(s["band_h"] + 2, ri, ri).translate([0, 0, -1])
    top = yb + s["dt_h"] + 0.15
    ftop = top + 0.1
    frame = (CS.square([2 * s["frame_w"] - 4, ftop - (ri - 1.5) - 4], True).offset(2.0, JoinType.Round)
             .translate([0, (ftop + ri - 1.5) / 2]).extrude(s["H"]))
    r = band + frame
    r -= M.cylinder(60, ri, ri).translate([0, 0, -20])                              # the finger
    c0, c1 = s["a0"] + s["clr"], s["a1"] + s["clr"]
    chan = CS([[[-c0, yb - 6], [c0, yb - 6], [c0, yb], [c1, top], [-c1, top], [-c0, yb]]]).extrude(s["H"] + 2).translate([0, 0, s["floor"]])
    chan += box(-c1, c1, top - 0.5, top + 5, s["floor"], s["H"] + 2)               # open top for the capsule neck
    r -= chan
    r -= box(-1.6, 1.6, yb - 3, yb + 2.0, -1, s["floor"] + 0.5)                     # lead notch through the stop
    n = int(us_size) - 6                                                            # size marker: dots = size - 6
    for k in range(n):
        a = math.radians(-90 + (k - (n - 1) / 2) * 12)
        r -= M.cylinder(4, 0.7, 0.7).rotate([0, 90, 0]).rotate([0, 0, math.degrees(a)]).translate(
            [(ri + s["band_t"] - 0.6) * math.cos(a), (ri + s["band_t"] - 0.6) * math.sin(a), s["band_h"] / 2])
    return r, yb


def capsule_print():
    cap, _ = swap_capsule()
    return on_bed(cap.rotate([-90, 0, 0]))               # top plate on the bed, motor pocket facing up


def swap_ring_plate(n_capsules=3):
    """All six ring sizes + spare motor capsules on one print bed. Every part prints flat, no supports."""
    out = None
    for i, size in enumerate(RING_SIZES):
        r = swap_ring(size)[0].translate([(i % 3) * 36.0, (i // 3) * 40.0, 0])
        out = r if out is None else out + r
    for k in range(n_capsules):
        c = capsule_print()
        out += c.translate([-8.0 + k * 24.0, 70.0, 0])
    return on_bed(out)
