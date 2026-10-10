"""
NavBelt Orin server  —  one program, one port (8765)
=====================================================
  ws://orin.local:8765/stream   phone sends frames + commands, gets back {"type":"say"}
  http://orin.local:8765/       dashboard for the judges (camera / depth / belt / status)
  /video/camera  /video/depth   MJPEG streams      /state  JSON      POST /cmd  typed command

Run:
  export GEMINI_API_KEY=...            # optional: without it, simple keyword commands still work
  python3 navbelt_server.py
Settings (environment variables): FLIP_IMAGE=1 if the dashboard image is upside down,
CAM_HEIGHT_M=1.0 for the belt cradle (1.3 chest), USE_DEPTH=0 to skip the depth model,
RING_HAND=left if the ring is on the left hand, YOLO_ENGINE=path, GEMINI_MODEL=name,
USE_OPEN_VOCAB=0 to turn off the find-anything detector, OPEN_VOCAB_CONF=0.25 if it finds wrong things.

Finding objects: the local find-anything detector (YOLO-World) answers first, in a fraction of a
second; Gemini is only asked when the local detector finds nothing (and for describe + memory).
Threads: perception (YOLO + depth every frame) -> control (10 Hz motor commands) ;
         refresher (re-finds the target) ; memory (logs where objects were seen).
"""
import asyncio, glob, json, math, os, re, threading, time
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ----------------------------------------------------------------- settings
YOLO_ENGINE = os.environ.get("YOLO_ENGINE", "yolo11n.engine")
USE_DEPTH = os.environ.get("USE_DEPTH", "1") == "1"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
FLIP_IMAGE = os.environ.get("FLIP_IMAGE", "0") == "1"
CAM_HEIGHT_M = float(os.environ.get("CAM_HEIGHT_M", "1.3"))
RING_HAND = os.environ.get("RING_HAND", "right")
OBSTACLE_M = 1.0          # closer than this = blocked
REACH_M = 0.9             # closer than this = switch from belt to ring
MEMORY_EVERY_S = 8          # Gemini memory scan (slow, smart)
LOCAL_MEMORY_EVERY_S = 3    # find-anything memory scan (fast, local)
REFRESH_LOCAL_S = 0.4       # how often the target is re-found with the local detector
REFRESH_GEMINI_S = 1.5      # ...and with Gemini when there is no local detector
# find-anything detector (YOLO-World): give it words, it finds them in every frame, on the Orin
USE_OPEN_VOCAB = os.environ.get("USE_OPEN_VOCAB", "1") == "1"
OPEN_VOCAB_MODEL = os.environ.get("OPEN_VOCAB_MODEL", "yolov8s-worldv2.pt")
OPEN_VOCAB_CONF = float(os.environ.get("OPEN_VOCAB_CONF", "0.15"))
MEMORY_CLASSES = ["keys", "wallet", "phone", "glasses", "cup", "bottle", "backpack", "bag",
                  "remote", "laptop", "door", "chair"]
# body angle of each belt motor in degrees, + = to the wearer's right. Measure on the wearer!
MOTOR_ANGLES = {1: 0, 2: 45, 3: 90, 4: 120, 5: -120, 6: -90, 7: -45}
RING = 8
W, H = 640, 480

INTENT_PROMPT = ("You control a navigation belt for a blind user. Convert the spoken request into JSON "
                 '{"intent": "find|where|follow|goto|describe|stop|walk", "target": "short noun phrase or empty"}. '
                 "find/where = locate an object; goto = walk to a place or thing (door, exit, chair); "
                 "follow = follow a person (target = their description or empty). Request: ")
FIND_PROMPT = ('Find "{target}" in this image. Return a JSON list of {{"label": string, '
               '"box_2d": [ymin, xmin, ymax, xmax]}} with coordinates normalized to 0-1000, best match first. '
               "Return [] if it is not visible.")
SCAN_PROMPT = ("List up to 8 objects in this image that a blind person might want to find later "
               "(keys, wallet, phone, glasses, cup, bottle, bag, backpack, remote, laptop, door, chair). "
               'Return a JSON list of {"label": short lowercase name, "box_2d": [ymin, xmin, ymax, xmax]} normalized to 0-1000.')
DESCRIBE_PROMPT = ("You are the eyes of a blind person wearing this camera on their chest. In two short sentences, "
                   "say what is in front of them and where (left, right, ahead), mentioning obstacles first.")


# ----------------------------------------------------------------- shared state
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame, self.meta, self.frame_id = None, None, 0
        self.phone_depth, self.phone_depth_t = None, 0.0
        self.depth = None
        self.annotated, self.depth_vis = None, None
        self.dets = []
        self.blocked, self.dist = [False] * 16, [9.9] * 16
        self.mode = "IDLE"            # IDLE WALK GOTO SEARCH FOLLOW REACH
        self.target = None
        self.memory = {}
        self.motors = {}
        self.last_heard, self.last_said = "", ""
        self.fps = 0.0
        self.depth_source = "none"
        self.last_finder = ""


S = State()
PHONE = {"ws": None, "loop": None}
LAST = {"sweep": 0.0, "hint": 0.0, "inside_since": 0.0, "gemini": 0.0}
GEMINI_LOCK = threading.Lock()
BELT = None
BRAIN = None
LOCAL = None


# ----------------------------------------------------------------- geometry
def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def pose(meta):
    """heading (deg, + = turned right since start), pitch (deg, + = up), position (x, z) in meters."""
    T = np.array(meta.get("T") or [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], float).reshape(4, 4).T
    fwd = -T[:3, 2]
    heading = math.degrees(math.atan2(fwd[0], -fwd[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
    return heading, pitch, np.array([T[0, 3], T[2, 3]])


def cam(meta):
    fx, fy, cx, cy = meta.get("intr") or [500.0, 500.0, W / 2, H / 2]
    if FLIP_IMAGE:
        cx, cy = W - cx, H - cy
    return fx, fy, cx, cy


def px_bearing(x, meta):
    fx, _, cx, _ = cam(meta)
    return math.degrees(math.atan2(x - cx, fx))


def floor_dist(y_bottom, meta):
    """distance to something standing on the floor, from the bottom edge of its box."""
    _, fy, _, cy = cam(meta)
    below = -pose(meta)[1] + math.degrees(math.atan2(y_bottom - cy, fy))
    return 9.9 if below <= 1 else min(9.9, CAM_HEIGHT_M / math.tan(math.radians(below)))


def heading_to(pos_from, pos_to):
    d = pos_to - pos_from
    return math.degrees(math.atan2(d[0], -d[1])), float(np.hypot(d[0], d[1]))


def direction_words(rel):
    a = abs(rel)
    side = "right" if rel > 0 else "left"
    if a < 15:
        return "straight ahead"
    if a < 45:
        return f"slightly to your {side}"
    if a < 135:
        return f"to your {side}"
    return "behind you"


def box_distance(box, meta, depth):
    x0, y0, x1, y1 = box
    if depth is not None:
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        patch = depth[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6]
        if patch.size:
            return float(np.median(patch))
    d = floor_dist(y1, meta)
    return d if d < 9.9 else 1.5


# ----------------------------------------------------------------- belt (ESP32 over USB serial)
class Belt:
    def __init__(self):
        self.ser, self.on, self.seq_until = None, {}, 0.0
        ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if ports:
            import serial
            self.ser = serial.Serial(ports[0], 115200, timeout=0.05)
            time.sleep(2.0)                      # opening the port resets the ESP32
            self.ser.reset_input_buffer()
            print("Belt connected on", ports[0])
        else:
            print("No belt found - running without motors")

    def send(self, line):
        if self.ser:
            try:
                self.ser.write((line + "\n").encode())
            except Exception as e:
                print("serial error:", e)

    def pattern(self, name):                     # SWEEP or ARRIVE
        self.send(name)
        self.on, self.seq_until = {}, time.time() + 1.1
        with S.lock:
            S.motors = {}

    def set(self, wanted):                       # {motor: (duty, hz)}, called ~10x/s
        if time.time() < self.seq_until:
            return
        wanted = dict(list(wanted.items())[:3])
        for m in list(self.on):
            if m not in wanted:
                self.send(f"M {m} 0 S")
        for m, (duty, hz) in wanted.items():
            self.send(f"M {m} {int(duty)} " + (f"P{hz:.1f}" if hz > 0 else "S"))
        self.on = wanted
        with S.lock:
            S.motors = dict(wanted)


def motors_for(angle, duty, hz):
    a = wrap(angle)
    if abs(a) > 150:
        return {4: (duty, hz), 5: (duty, hz)}
    m = min(MOTOR_ANGLES, key=lambda k: abs(wrap(MOTOR_ANGLES[k] - a)))
    return {m: (duty, hz)}


# ----------------------------------------------------------------- talking back
def say(text):
    print("SAY:", text)
    with S.lock:
        S.last_said = text
    ws, loop = PHONE["ws"], PHONE["loop"]
    if ws is not None and loop is not None:
        asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps({"type": "say", "text": text})), loop)


def say_hint(text, every):
    if time.time() - LAST["hint"] > every:
        LAST["hint"] = time.time()
        say(text)


# ----------------------------------------------------------------- Gemini brain
def simple_intent(text):
    t = text.lower()
    m = re.search(r"(?:my|the|a)\s+([a-z ]+)$", t)
    target = m.group(1).strip() if m else ""
    if "stop" in t:
        return {"intent": "stop", "target": ""}
    if "follow" in t:
        return {"intent": "follow", "target": ""}
    if "front of me" in t or "describe" in t or "what do you see" in t:
        return {"intent": "describe", "target": ""}
    if "take me" in t or "go to" in t:
        return {"intent": "goto", "target": target}
    if "walk" in t:
        return {"intent": "walk", "target": ""}
    if "where" in t or "find" in t:
        return {"intent": "find", "target": target}
    return {"intent": "", "target": ""}


def fast_intent(text):
    """Instant keyword parse for the common commands; None means 'ask Gemini'."""
    t = text.lower().strip()
    it = simple_intent(t)
    if it["intent"] in ("stop", "walk", "describe"):
        return it
    if it["intent"] in ("find", "goto") and it["target"]:
        return it
    if it["intent"] == "follow" and re.search(r"follow (me|them|him|her|us)\b", t):
        return it
    return None


class LocalFinder:
    """YOLO-World: open-vocabulary detector. set the words, it finds them; runs on the Orin's GPU."""
    def __init__(self):
        self.model, self.classes, self.lock = None, None, threading.Lock()
        if not USE_OPEN_VOCAB:
            print("Find-anything detector off (USE_OPEN_VOCAB=0)")
            return
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(OPEN_VOCAB_MODEL)
            self._set(["keys"])                        # first run downloads the text encoder (needs internet)
            self.model.predict(np.zeros((H, W, 3), np.uint8), verbose=False)
            print("Find-anything detector ready:", OPEN_VOCAB_MODEL)
        except Exception as e:
            print("Find-anything detector off:", e)
            self.model = None

    def _set(self, names):
        if names != self.classes:
            self.model.set_classes(list(names))
            self.classes = list(names)

    def detect(self, frame, names, conf=None):
        if self.model is None or frame is None or not names:
            return []
        with self.lock:
            self._set(names)
            r = self.model.predict(frame, conf=OPEN_VOCAB_CONF if conf is None else conf, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                out.append({"label": names[c] if c < len(names) else names[0], "box": b, "conf": cf})
        return sorted(out, key=lambda d: -d["conf"])


class Brain:
    def __init__(self):
        self.client, self.types = None, None
        self.no_think = True
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            from google import genai
            from google.genai import types
            self.client, self.types = genai.Client(api_key=key), types
            print("Gemini ready:", GEMINI_MODEL)
        else:
            print("GEMINI_API_KEY not set - finding uses the local detector only; no describe")

    def _ask(self, parts, as_json=True):
        kw = dict(response_mime_type="application/json" if as_json else "text/plain", temperature=0.2)
        if self.no_think:                                # skip Gemini's 'thinking' step: much faster
            try:
                kw["thinking_config"] = self.types.ThinkingConfig(thinking_budget=0)
            except Exception:
                self.no_think = False
        t0 = time.time()
        try:
            with GEMINI_LOCK:
                r = self.client.models.generate_content(model=GEMINI_MODEL, contents=parts,
                                                        config=self.types.GenerateContentConfig(**kw))
        except Exception as e:
            if self.no_think and "think" in str(e).lower():
                self.no_think = False                    # this model can't turn thinking off; retry normally
                return self._ask(parts, as_json)
            raise
        LAST["gemini"] = time.time()
        print(f"Gemini answered in {time.time() - t0:.1f} s")
        return json.loads(r.text) if as_json else r.text.strip()

    def _img(self, frame):
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        return self.types.Part.from_bytes(data=jpg, mime_type="image/jpeg")

    def intent(self, text):
        fast = fast_intent(text)
        if fast is not None:
            return fast
        if not self.client:
            return simple_intent(text)
        try:
            out = self._ask([INTENT_PROMPT + text])
            return out if isinstance(out, dict) else simple_intent(text)
        except Exception as e:
            print("intent error:", e)
            return simple_intent(text)

    def _boxes(self, frame, prompt):
        if not self.client or frame is None:
            return []
        try:
            items = self._ask([self._img(frame), prompt])
        except Exception as e:
            print("gemini error:", e)
            return []
        out = []
        for it in items if isinstance(items, list) else []:
            try:
                y0, x0, y1, x1 = it["box_2d"]
                out.append({"label": str(it.get("label", "")).lower(),
                            "box": [x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000]})
            except Exception:
                continue
        return out

    def find(self, frame, target, gemini_ok=True):
        """local find-anything detector first; Gemini only if it finds nothing."""
        items = LOCAL.detect(frame, [target]) if LOCAL is not None else []
        if items:
            with S.lock:
                S.last_finder = f"YOLO-World ({items[0]['conf']:.2f})"
            return items
        if not gemini_ok:
            return []
        items = self._boxes(frame, FIND_PROMPT.format(target=target))
        if items:
            with S.lock:
                S.last_finder = "Gemini"
        return items

    def scan(self, frame):
        return self._boxes(frame, SCAN_PROMPT)

    def describe(self, frame):
        if not self.client or frame is None:
            return "I can't describe the scene right now."
        try:
            return self._ask([self._img(frame), DESCRIBE_PROMPT], as_json=False)
        except Exception as e:
            print("describe error:", e)
            return "I can't describe the scene right now."


# ----------------------------------------------------------------- perception thread
def compute_sectors(depth, dets, meta):
    dist = [9.9] * 16
    if depth is not None:
        band = depth[int(H * 0.25):int(H * 0.80)]
        cols = np.percentile(band, 10, axis=0)
        for s in range(16):
            dist[s] = float(cols[s * W // 16:(s + 1) * W // 16].min())
    else:
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            dd = floor_dist(y1, meta) if y1 < H - 3 else 0.8
            for s in range(max(0, int(x0 * 16 / W)), min(15, int(x1 * 16 / W)) + 1):
                dist[s] = min(dist[s], dd)
    return [x < OBSTACLE_M for x in dist], dist


def draw(frame, meta, dets, blocked, depth):
    img = frame.copy()
    with S.lock:
        tgt, mode = S.target, S.mode
    for d in dets:
        x0, y0, x1, y1 = map(int, d["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
        cv2.putText(img, f"{d['name']}{'' if d['id'] is None else ' #' + str(d['id'])}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    if tgt and tgt.get("box") is not None:
        x0, y0, x1, y1 = map(int, tgt["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 3)
        cv2.putText(img, tgt.get("label", ""), (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    for h in (meta or {}).get("hands", []):
        for k, col in (("wrist", (255, 120, 0)), ("index", (255, 0, 200))):
            if h.get(k):
                cv2.circle(img, (int(h[k][0]), int(h[k][1])), 7, col, -1)
    for s in range(16):
        x = s * W // 16
        cv2.rectangle(img, (x + 1, H - 14), (x + W // 16 - 1, H - 2), (0, 0, 220) if blocked[s] else (0, 180, 0), -1)
    cv2.putText(img, mode, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    ann = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    if depth is not None:
        v = cv2.applyColorMap((255 - np.clip(depth / 5.0 * 255, 0, 255)).astype(np.uint8), cv2.COLORMAP_TURBO)
    else:
        v = np.zeros((H, W, 3), np.uint8)
        cv2.putText(v, "No depth: obstacles from YOLO boxes", (40, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    for s in range(16):
        x = s * W // 16
        cv2.rectangle(v, (x + 1, H - 14), (x + W // 16 - 1, H - 2), (0, 0, 220) if blocked[s] else (0, 180, 0), -1)
    dv = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    with S.lock:
        S.annotated, S.depth_vis = ann, dv


def perception_loop():
    from ultralytics import YOLO
    model = YOLO(YOLO_ENGINE if os.path.exists(YOLO_ENGINE) else "yolo11n.pt", task="detect")
    depth_pipe = None
    if USE_DEPTH:
        try:
            import torch
            from transformers import pipeline
            depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=0 if torch.cuda.is_available() else -1)
            print("Depth model ready")
        except Exception as e:
            print("Depth model off:", e)
    from PIL import Image
    last_id, n, t0, depth = -1, 0, time.time(), None
    while True:
        with S.lock:
            fid, frame, meta = S.frame_id, S.frame, S.meta
            pd, pdt = S.phone_depth, S.phone_depth_t
        if frame is None or fid == last_id:
            time.sleep(0.005)
            continue
        last_id = fid
        res = model.track(frame, persist=True, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
            for b, c, cf, i in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), ids):
                dets.append({"box": b, "name": model.names[c], "conf": cf, "id": i})
        source = "none"
        if pd is not None and time.time() - pdt < 0.5:
            depth, source = pd, "LiDAR"
        elif depth_pipe is not None:
            if n % 2 == 0:
                small = cv2.resize(frame, (320, 240))
                d = depth_pipe(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))["predicted_depth"]
                depth = cv2.resize(d.squeeze().cpu().numpy().astype(np.float32), (W, H))
            source = "estimated"
        else:
            depth = None
        blocked, dist = compute_sectors(depth, dets, meta)
        n += 1
        with S.lock:
            S.dets, S.depth, S.blocked, S.dist, S.depth_source = dets, depth, blocked, dist, source
            if n % 15 == 0:
                S.fps = 15 / max(1e-3, time.time() - t0)
        if n % 15 == 0:
            t0 = time.time()
        draw(frame, meta, dets, blocked, depth)


# ----------------------------------------------------------------- modes and targets
def set_mode(mode, target=None, keep_target=False):
    with S.lock:
        S.mode = mode
        if not keep_target:
            S.target = target
    if mode == "IDLE":
        BELT.set({})


def latest():
    with S.lock:
        return (None if S.frame is None else S.frame.copy()), S.meta, S.depth


def lock_target(label, box, meta, depth, walk_only=False):
    heading, pitch, pos = pose(meta)
    cx = (box[0] + box[2]) / 2
    world_h = heading + px_bearing(cx, meta)
    dist = box_distance(box, meta, depth)
    tpos = pos + dist * np.array([math.sin(math.radians(world_h)), -math.cos(math.radians(world_h))])
    return {"label": label, "pos": tpos, "dist": dist, "box": box, "box_heading": heading, "box_pitch": pitch,
            "box_t": time.time(), "in_view": True, "walk_only": walk_only}


def start_reach(tgt):
    LAST["inside_since"] = 0.0
    BELT.set({})
    set_mode("REACH", tgt)
    say("It's right in front of you. Reach out, and the ring will guide your hand.")


def start_find(target, walk_only=False):
    frame, meta, depth = latest()
    if frame is None:
        say("I'm not getting any video yet.")
        return
    items = BRAIN.find(frame, target)
    if items:
        tgt = lock_target(target, items[0]["box"], meta, depth, walk_only)
        rel = wrap(heading_to(pose(meta)[2], tgt["pos"])[0] - pose(meta)[0])
        if tgt["dist"] < REACH_M and not walk_only:
            start_reach(tgt)
            return
        set_mode("GOTO", tgt)
        verb = "are" if target.endswith("s") else "is"
        say(f"Your {target} {verb} {direction_words(rel)}, about {max(1, round(tgt['dist']))} meters away. Follow the belt."
            if not walk_only else f"The {target} {verb} {direction_words(rel)}. Follow the belt.")
        return
    mem = recall(target)
    if mem:
        heading, _, pos = pose(meta)
        rel = wrap(heading_to(pos, mem["pos"])[0] - heading)
        set_mode("GOTO", {"label": target, "pos": mem["pos"], "dist": None, "box": None, "in_view": False,
                          "walk_only": walk_only})
        say(f"I last saw your {target} {direction_words(rel)}, {int(time.time() - mem['t'])} seconds ago. Follow the belt.")
        return
    set_mode("SEARCH", {"label": target, "pos": None, "box": None, "in_view": False, "walk_only": walk_only,
                        "started": time.time()})
    say(f"I can't see your {target} yet. Turn slowly and I'll keep looking.")


def start_follow(desc):
    frame, meta, _ = latest()
    with S.lock:
        people = [d for d in S.dets if d["name"] == "person" and d["id"] is not None]
    if not people:
        say("I don't see anyone to follow.")
        return
    pick = max(people, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    if desc and desc not in ("me", "person", "them", "someone"):
        found = BRAIN.find(frame, desc)
        if found:
            fb = found[0]["box"]
            pick = max(people, key=lambda d: iou(d["box"], fb))
    set_mode("FOLLOW", {"label": "person", "follow_id": pick["id"], "box": None})
    say("Following. Keep the buzz in front of you.")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def recall(target):
    words = set(target.lower().split())
    with S.lock:
        hits = [(k, v) for k, v in S.memory.items() if words & set(k.split())]
    return max(hits, key=lambda kv: kv[1]["t"])[1] if hits else None


def handle_command(text):
    with S.lock:
        S.last_heard = text
    it = BRAIN.intent(text)
    intent, target = it.get("intent", ""), (it.get("target") or "").strip()
    print(f"COMMAND {text!r} -> {intent} {target!r}")
    if intent == "stop":
        set_mode("IDLE")
        say("Stopped.")
    elif intent == "walk":
        set_mode("WALK")
        say("Walking. I'll steer you around obstacles.")
    elif intent == "describe":
        frame, _, _ = latest()
        say(BRAIN.describe(frame))
    elif intent == "follow":
        start_follow(target)
    elif intent in ("find", "where"):
        start_find(target or "keys")
    elif intent == "goto":
        start_find(target or "door", walk_only=True)
    else:
        say("Sorry, I didn't catch that.")


# ----------------------------------------------------------------- control loop (10 Hz)
def pick_hand(hands):
    best = None
    for h in hands or []:
        pt = h.get("index") or h.get("wrist")
        if not pt:
            continue
        if best is None or h.get("side") == RING_HAND:
            best = pt
    return best


def reach_step(meta, tgt, hands):
    if not tgt or tgt.get("box") is None:
        BELT.set({})
        return
    heading, pitch, _ = pose(meta)
    fx, fy, _, _ = cam(meta)
    dx = -fx * math.tan(math.radians(wrap(heading - tgt["box_heading"])))
    dy = fy * math.tan(math.radians(pitch - tgt["box_pitch"]))
    x0, y0, x1, y1 = tgt["box"]
    bx0, by0, bx1, by1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
    ocx, ocy = (bx0 + bx1) / 2, (by0 + by1) / 2
    hand = pick_hand(hands)
    if hand is None:
        BELT.set({RING: (60, 1.0)})
        say_hint("Reach toward it. I can't see your hand yet.", 4)
        return
    hx, hy = hand
    pad = 20
    if bx0 - pad <= hx <= bx1 + pad and by0 - pad <= hy <= by1 + pad:
        BELT.set({RING: (100, 0)})
        if LAST["inside_since"] == 0:
            LAST["inside_since"] = time.time()
        elif time.time() - LAST["inside_since"] > 0.6:
            BELT.pattern("ARRIVE")
            say(f"Got it. That's your {tgt['label']}.")
            set_mode("IDLE")
        return
    LAST["inside_since"] = 0.0
    diag = max(40.0, math.hypot(bx1 - bx0, by1 - by0))
    dn = math.hypot(ocx - hx, ocy - hy) / diag
    BELT.set({RING: (90, max(2.0, min(12.0, 12.0 - 3.0 * dn)))})
    if abs(ocx - hx) > abs(ocy - hy):
        say_hint("left" if ocx < hx else "right", 2.5)
    else:
        say_hint("forward" if ocy < hy else "back toward you", 2.5)


def control_loop():
    while True:
        time.sleep(0.1)
        if BELT is None:
            continue
        with S.lock:
            mode, meta, tgt = S.mode, S.meta, S.target
            blocked, dist, dets = list(S.blocked), list(S.dist), list(S.dets)
        if meta is None or mode == "IDLE":
            BELT.set({})
            continue
        if mode == "REACH":
            reach_step(meta, tgt, meta.get("hands", []))
            continue
        heading, pitch, pos = pose(meta)
        goal, gdist = 0.0, None
        if mode in ("GOTO", "SEARCH"):
            if not tgt or tgt.get("pos") is None:
                BELT.set({})                                       # searching: stand still, turn slowly
                continue
            th, gdist = heading_to(pos, tgt["pos"])
            goal = wrap(th - heading)
        elif mode == "FOLLOW":
            p = next((d for d in dets if d["id"] == (tgt or {}).get("follow_id")), None)
            if p is None:
                BELT.set({})
                continue
            goal = px_bearing((p["box"][0] + p["box"][2]) / 2, meta)
            gdist = floor_dist(p["box"][3], meta)
            if gdist < 1.2:
                BELT.set({})
                continue
        if mode == "GOTO" and gdist is not None and gdist < REACH_M:
            if tgt.get("walk_only"):
                BELT.pattern("ARRIVE")
                say(f"You're at the {tgt['label']}.")
                set_mode("IDLE")
            elif tgt.get("in_view"):
                start_reach(tgt)
            continue
        steer, near = goal, 9.9
        half_fov = abs(px_bearing(0, meta))
        if abs(goal) < half_fov:
            free = [s for s in range(16) if not blocked[s]]
            if not free:
                if time.time() - LAST["sweep"] > 1.5:
                    LAST["sweep"] = time.time()
                    BELT.pattern("SWEEP")
                continue
            best = min(free, key=lambda s: abs(px_bearing((s + 0.5) * W / 16, meta) - goal))
            steer, near = px_bearing((best + 0.5) * W / 16, meta), dist[best]
            if abs(steer - goal) > 3 and abs(steer) < 30:
                steer = math.copysign(45, steer)          # dodge: make the side cue unmistakable
        closeness = min(near, gdist if gdist is not None else 9.9)
        hz = 2.0 if closeness > 3 else (4.0 if closeness > 1.8 else 7.0)
        BELT.set(motors_for(steer, 80, hz))


# ----------------------------------------------------------------- Gemini background loops
def refresher_loop():
    """re-finds the current target so the box/position stay fresh (GOTO, SEARCH, REACH)."""
    while True:
        local_ok = LOCAL is not None and LOCAL.model is not None
        time.sleep(REFRESH_LOCAL_S if local_ok else REFRESH_GEMINI_S)
        if BRAIN is None or (BRAIN.client is None and not local_ok):
            continue
        with S.lock:
            mode, tgt = S.mode, S.target
        if mode not in ("GOTO", "SEARCH", "REACH") or not tgt:
            continue
        frame, meta, depth = latest()
        if frame is None:
            continue
        if mode == "GOTO" and tgt.get("pos") is not None:
            heading, _, pos = pose(meta)
            if abs(wrap(heading_to(pos, tgt["pos"])[0] - heading)) > abs(px_bearing(0, meta)):
                continue                                           # target not in view: nothing to refresh
        items = BRAIN.find(frame, tgt["label"], gemini_ok=time.time() - LAST["gemini"] > REFRESH_GEMINI_S)
        if not items:
            if mode == "SEARCH" and time.time() - tgt.get("started", time.time()) > 25:
                say(f"I still can't find your {tgt['label']}. Stopping the search.")
                set_mode("IDLE")
            continue
        new = lock_target(tgt["label"], items[0]["box"], meta, depth, tgt.get("walk_only", False))
        with S.lock:
            still = S.mode == mode and S.target is tgt
        if not still:
            continue
        if mode == "SEARCH":
            heading, _, pos = pose(meta)
            set_mode("GOTO", new)
            say(f"Found your {tgt['label']}, {direction_words(wrap(heading_to(pos, new['pos'])[0] - heading))}. Follow the belt.")
        elif mode == "REACH":
            with S.lock:
                for k in ("box", "box_heading", "box_pitch", "box_t"):
                    tgt[k] = new[k]
        else:
            with S.lock:
                tgt.update(new)


def memory_loop():
    """logs where notable objects are, so 'where did I leave...' works later.
    Local find-anything scan every few seconds; Gemini adds a smarter scan now and then."""
    last_gemini = 0.0
    while True:
        time.sleep(LOCAL_MEMORY_EVERY_S)
        if BRAIN is None:
            continue
        with S.lock:
            mode = S.mode
        if mode in ("REACH", "SEARCH"):
            continue
        frame, meta, depth = latest()
        if frame is None:
            continue
        items = []
        if LOCAL is not None and LOCAL.model is not None:
            items += LOCAL.detect(frame, MEMORY_CLASSES, conf=0.25)
        if BRAIN.client is not None and time.time() - last_gemini > MEMORY_EVERY_S:
            last_gemini = time.time()
            items += BRAIN.scan(frame)
        for it in items:
            if not it["label"]:
                continue
            t = lock_target(it["label"], it["box"], meta, depth)
            with S.lock:
                S.memory[it["label"]] = {"pos": t["pos"], "t": time.time()}


# ----------------------------------------------------------------- web: phone socket + dashboard
app = FastAPI()


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    PHONE["ws"], PHONE["loop"] = ws, asyncio.get_running_loop()
    print("Phone connected")
    meta, expect = None, None
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "frame":
                    meta, expect = data, "jpeg"
                elif data.get("type") == "cmd":
                    threading.Thread(target=handle_command, args=(data.get("text", ""),), daemon=True).start()
            elif msg.get("bytes") is not None and meta is not None:
                b = msg["bytes"]
                if expect == "jpeg":
                    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if img.shape[1] != W or img.shape[0] != H:
                        img = cv2.resize(img, (W, H))
                    m = json.loads(json.dumps(meta))
                    if FLIP_IMAGE:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                        for hd in m.get("hands", []):
                            for k in ("wrist", "index"):
                                if hd.get(k):
                                    hd[k] = [W - hd[k][0], H - hd[k][1]]
                    with S.lock:
                        S.frame, S.meta, S.frame_id = img, m, S.frame_id + 1
                    expect = "depth" if meta.get("depth") else None
                elif expect == "depth":
                    d = np.frombuffer(b, np.float16).astype(np.float32).reshape(meta.get("dh", 96), meta.get("dw", 128))
                    d = cv2.resize(d, (W, H))
                    if FLIP_IMAGE:
                        d = cv2.rotate(d, cv2.ROTATE_180)
                    with S.lock:
                        S.phone_depth, S.phone_depth_t = d, time.time()
                    expect = None
    except WebSocketDisconnect:
        pass
    finally:
        PHONE["ws"] = None
        print("Phone disconnected")


def mjpeg(getter):
    async def gen():
        while True:
            jpg = getter()
            if jpg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.07)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video/camera")
async def video_camera():
    return mjpeg(lambda: S.annotated)


@app.get("/video/depth")
async def video_depth():
    return mjpeg(lambda: S.depth_vis)


@app.get("/state")
async def state():
    with S.lock:
        tgt = S.target or {}
        return JSONResponse({
            "mode": S.mode, "fps": round(S.fps, 1), "motors": {str(k): v for k, v in S.motors.items()},
            "heard": S.last_heard, "said": S.last_said, "target": tgt.get("label"), "finder": S.last_finder,
            "depth": S.depth_source, "phone": PHONE["ws"] is not None,
            "memory": sorted([[k, int(time.time() - v["t"])] for k, v in S.memory.items()], key=lambda kv: kv[1])[:8]})


@app.post("/cmd")
async def cmd(payload: dict = Body(...)):
    threading.Thread(target=handle_command, args=(payload.get("text", ""),), daemon=True).start()
    return {"ok": True}


@app.get("/")
async def index():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>NavBelt</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:0;background:#111;color:#eee;font:18px system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:12px 16px;flex-wrap:wrap}
button{font:inherit;padding:8px 16px;border-radius:8px;border:1px solid #555;background:#222;color:#eee;cursor:pointer}
button.on{background:#0a84ff;border-color:#0a84ff}
main{display:flex;gap:16px;padding:0 16px 16px;flex-wrap:wrap}
.views{display:flex;gap:8px;flex:1;min-width:320px}
.views img{width:100%;max-width:960px;border-radius:8px;background:#000}
aside{width:300px}
#status{font-size:22px;line-height:1.6}
.big{font-size:30px;font-weight:600;color:#0a84ff}
input{font:inherit;padding:8px;border-radius:8px;border:1px solid #555;background:#222;color:#eee;width:100%;box-sizing:border-box}
</style></head><body>
<header><span class="big">NavBelt</span>
<button id="b-cam" class="on" onclick="view('cam')">Camera</button>
<button id="b-depth" onclick="view('depth')">Depth</button>
<button id="b-split" onclick="view('split')">Split</button>
<span id="phone"></span></header>
<main><div class="views"><img id="cam" src="/video/camera"><img id="depth" src="/video/depth" style="display:none"></div>
<aside><svg id="belt" viewBox="-110 -110 220 220" width="300" height="300"></svg>
<div id="status"></div>
<p><input id="cmd" placeholder="Type a command, e.g. find my keys" onkeydown="if(event.key==='Enter')send()"></p>
<div id="memory"></div></aside></main>
<script>
const ANG={1:0,2:45,3:90,4:120,5:-120,6:-90,7:-45};
function view(v){document.getElementById('cam').style.display=v==='depth'?'none':'';
document.getElementById('depth').style.display=v==='cam'?'none':'';
for(const k of ['cam','depth','split'])document.getElementById('b-'+k).className=k===v?'on':'';}
function send(){const i=document.getElementById('cmd');fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:i.value})});i.value='';}
function belt(m){let s='<circle r="80" fill="none" stroke="#444" stroke-width="6"/><text y="-92" text-anchor="middle" fill="#888" font-size="14">front</text>';
for(const k in ANG){const a=ANG[k]*Math.PI/180,x=80*Math.sin(a),y=-80*Math.cos(a),on=m[k];
s+=`<circle cx="${x}" cy="${y}" r="${on?16:11}" fill="${on?'#0a84ff':'#333'}"/><text x="${x}" y="${y+5}" text-anchor="middle" fill="#fff" font-size="13">${k}</text>`;}
const r=m['8'];s+=`<circle r="${r?24:16}" fill="${r?'#ff9f0a':'#333'}"/><text y="5" text-anchor="middle" fill="#fff" font-size="13">ring</text>`;
document.getElementById('belt').innerHTML=s;}
async function tick(){try{const s=await (await fetch('/state')).json();belt(s.motors);
document.getElementById('phone').textContent=s.phone?'phone connected':'phone NOT connected';
document.getElementById('status').innerHTML=`Mode: <b>${s.mode}</b>${s.target?' ('+s.target+')':''}<br>Heard: ${s.heard||'-'}<br>Said: ${s.said||'-'}<br>Found by: ${s.finder||'-'}<br>${s.fps} fps, depth: ${s.depth}`;
document.getElementById('memory').innerHTML=s.memory.length?'<b>Last seen</b><br>'+s.memory.map(m=>`${m[0]} - ${m[1]} s ago`).join('<br>'):'';}catch(e){}}
setInterval(tick,200);belt({});
</script></body></html>"""


if __name__ == "__main__":
    BELT = Belt()
    LOCAL = LocalFinder()
    BRAIN = Brain()
    for fn in (perception_loop, control_loop, refresher_loop, memory_loop):
        threading.Thread(target=fn, daemon=True).start()
    print("Dashboard: http://orin.local:8765/   phone socket: ws://<this-ip>:8765/stream")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
