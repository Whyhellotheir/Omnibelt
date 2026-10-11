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
CAM_HEIGHT_M=1.0 belt cradle (default; set 1.3 for the chest), USE_DEPTH=0 to skip the depth model,
RING_HAND=left if the ring is on the left hand, YOLO_ENGINE=path, GEMINI_MODEL=name, USE_PLANNER=0 to
use dodging only (no map / route planning),
USE_OPEN_VOCAB=0 to turn off the find-anything detector, OPEN_VOCAB_CONF=0.25 if it finds wrong things.
GEMINI_FIND_CONF=0.6: Gemini finds it rates below this are ignored (raise if it finds things that aren't there).
Gemini models: GEMINI_MODEL then GEMINI_FALLBACK=a,b,... answer what the user asks (find, ask, describe);
GEMINI_LIGHT (default gemini-3.5-flash-lite) does background scans/re-finds and command parsing, so they never use
the Flash models' small daily quota. GEMINI_TIMEOUT_S=12 gives up on a slow model.
GEMINI_LIGHT_FIND_CONF=0.8: stricter confidence for background re-finds (Flash-Lite guesses more).
Gemini quota: GEMINI_MAX_PER_MIN=4 caps background calls per minute, GEMINI_MEMORY_EVERY_S=60 sets the
memory scan interval (0 = off), GEMINI_REFRESH_S=3 sets how often Gemini re-finds a target the local
detector misses. Spoken requests (find, describe, unusual commands) are never skipped.

Finding objects: the local find-anything detector (YOLO-World) answers first, in a fraction of a
second; Gemini is only asked when the local detector finds nothing (and for describe + memory).
Path planning: depth + the phone's ARKit position build a floor map; A* plans a route to the goal around
everything on it (including obstacles no longer in view); the dodge layer still checks every frame.
Threads: perception (YOLO + depth + map update every frame) -> planner (2.5 Hz) -> control (10 Hz motors) ;
         refresher (re-finds the target) ; memory (logs where objects were seen).
"""
import asyncio, collections, glob, json, math, os, re, sys, threading, time
import urllib.error
import urllib.request
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling modules below
import memory as memory_mod
import settings as settings_mod
import depth_trt
import finder
import geometry
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response

# ----------------------------------------------------------------- settings
YOLO_ENGINE = os.environ.get("YOLO_ENGINE", "yolo11n.engine")
USE_DEPTH = os.environ.get("USE_DEPTH", "1") == "1"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
# Two tiers, to fit the free-tier quotas (each Flash model ~20 requests/day and 5/min; Flash-Lite ~500/day, 15/min):
#  - smart (GEMINI_MODEL, then GEMINI_FALLBACK in order): what the user asked for - find, ask, describe.
#    Each model has its own daily quota, so the chain pools them.
#  - light (GEMINI_LIGHT): everything in the background - memory scan, re-finding a target - plus parsing commands.
#    Background calls never touch the smart tier's quota.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK = [m.strip() for m in os.environ.get("GEMINI_FALLBACK",
                   "gemini-3.7-flash,gemini-3.6-flash,gemini-3-flash").split(",") if m.strip()]
GEMINI_LIGHT = [m.strip() for m in os.environ.get("GEMINI_LIGHT", "gemini-3.5-flash-lite").split(",") if m.strip()]
GEMINI_TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "12"))
THINKS = [dict(thinking_budget=0), dict(thinking_level="minimal"), dict(thinking_level="low"), None]
FLIP_IMAGE = os.environ.get("FLIP_IMAGE", "0") == "1"
CAM_HEIGHT_M = float(os.environ.get("CAM_HEIGHT_M", "1.0"))   # phone on the belt at the navel (chest cradle: 1.3)
RING_HAND = os.environ.get("RING_HAND", "right")
OBSTACLE_M = 1.0          # closer than this = blocked
REACH_M = 0.9             # closer than this = switch from belt to ring
MEMORY_EVERY_S = float(os.environ.get("GEMINI_MEMORY_EVERY_S", "60"))  # Gemini memory scan; 0 = off
LOCAL_MEMORY_EVERY_S = 3    # find-anything memory scan (fast, local)
REFRESH_LOCAL_S = 0.4       # how often the target is re-found with the local detector
REFRESH_GEMINI_S = float(os.environ.get("GEMINI_REFRESH_S", "3"))     # ...and with Gemini when that misses
# Gemini quota: background calls (memory scan, re-finding) stop once this many calls were made in the
# last minute, and pause for a minute after a rate-limit error. Spoken requests always go through.
GEMINI_MAX_PER_MIN = int(os.environ.get("GEMINI_MAX_PER_MIN", "4"))
# find-anything detector (YOLO-World): give it words, it finds them in every frame, on the Orin
USE_OPEN_VOCAB = os.environ.get("USE_OPEN_VOCAB", "1") == "1"
OPEN_VOCAB_MODEL = os.environ.get("OPEN_VOCAB_MODEL", "yolov8s-worldv2.pt")
OPEN_VOCAB_CONF = float(os.environ.get("OPEN_VOCAB_CONF", "0.15"))
GEMINI_FIND_CONF = float(os.environ.get("GEMINI_FIND_CONF", "0.6"))   # Gemini finds below this are ignored
GEMINI_LIGHT_FIND_CONF = float(os.environ.get("GEMINI_LIGHT_FIND_CONF", "0.8"))  # stricter for Flash-Lite re-finds
MEMORY_CLASSES = ["keys", "wallet", "phone", "glasses", "cup", "bottle", "backpack", "bag",
                  "remote", "laptop", "door", "chair"]
# body angle of each belt motor in degrees, + = to the wearer's right. Measure on the wearer!
MOTOR_ANGLES = {1: 0, 2: 45, 3: 90, 4: 120, 5: -120, 6: -90, 7: -45}
# path planner: builds a floor map from depth + ARKit position, plans a route (A*) to the goal,
# and the dodge layer still checks every frame. USE_PLANNER=0 = reactive dodging only.
USE_PLANNER = os.environ.get("USE_PLANNER", "1") == "1"
MAP_CELL_M = 0.15          # map resolution
MAP_SIZE_M = 12.0          # map covers 12 x 12 m around the start point
MAP_RANGE_M = 4.0          # only trust depth out to this distance
BODY_RADIUS_M = 0.35       # keep the route this far from obstacles
LOOKAHEAD_M = 0.8          # steer toward the point this far along the route
OBST_MIN_H, OBST_MAX_H = 0.15, 1.9   # only things between ankle and head height count as obstacles
RING = 8
VERSION = "nav-accuracy w11"
FIELD = {"mirror_belt": False, "mirror_hands": False, "voice": "quiet"}   # dashboard switches
ORIENT = {"force": os.environ.get("ORIENT", "auto"), "k": 0}   # auto | portrait | landscape; k = current rotation
W, H = 640, 480
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")

INTENT_PROMPT = ("You control a navigation belt for a blind user. Convert the spoken request into JSON "
                 '{"intent": "find|where|follow|goto|describe|ask|stop|walk", "target": "short noun phrase or empty"}. '
                 "find/where = locate an object; goto = walk to a place or thing (door, exit, chair); "
                 "target is the bare name without my/the/a, e.g. 'keys', 'red backpack'; "
                 "follow = follow a person (target = their description or empty); "
                 "ask = any other question about what the camera sees, e.g. what kind, what color, how many, "
                 "read this, is it ... (target empty). Request: ")
FIND_PROMPT = ('Find "{target}" in this image. A blind person will walk to whatever you return, so only return '
               'something that really is a {target}, not just a similar shape, color or object. Return a JSON '
               'list of {{"label": what the object actually is, "box_2d": [ymin, xmin, ymax, xmax], '
               '"confidence": 0.0-1.0}} with coordinates normalized to 0-1000, best match first. '
               "If it is not clearly visible, return [] - an empty list is the right answer when unsure.")
SCAN_PROMPT = ("List up to 8 objects in this image that a blind person might want to find later "
               "(keys, wallet, phone, glasses, cup, bottle, bag, backpack, remote, laptop, door, chair). "
               'Return a JSON list of {"label": short lowercase name, "box_2d": [ymin, xmin, ymax, xmax]} normalized to 0-1000.')
ANSWER_PROMPT = ("You are the eyes of a blind person wearing this camera. Answer their question about what the "
                 "camera sees in one or two short, spoken sentences. If the image doesn't show enough to tell, "
                 "say so briefly. Question: ")
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
        self.path, self.path_t = None, 0.0
        self.map_vis = None
        self.cols = None          # per-column nearest obstacle distance (64 columns), for the map
        self.timing = {}          # ms per stage + fps, for the dashboard
        self.depth_scale = 1.0
        self.planes = []          # ARKit horizontal planes from the phone (world coords)
        self.pending = None       # {"type": "goto", "target": label, "expires": t} after "Want me to take you there?"


S = State()
MEM = memory_mod.Memory()
EVENTS = collections.deque(maxlen=200)          # transcript for the dashboard: (i, t, who, text)
EVENTS_LOCK = threading.Lock()


def log_event(who, text):
    with EVENTS_LOCK:
        i = EVENTS[-1][0] + 1 if EVENTS else 1
        EVENTS.append((i, time.time(), who, text))
PHONE = {"ws": None, "loop": None}
LAST = {"sweep": 0.0, "hint": 0.0, "inside_since": 0.0, "gemini": 0.0}
GEMINI_LOCK = threading.Lock()
BELT = None
BRAIN = None
LOCAL = None
DET = None                                   # finder.DetectorThread: fixed-vocab YOLO-World on the newest frame
STRONG_DET = None                            # finder.StrongDetector: the big model, only when the fast one misses
TEACH = finder.TeachStore(os.path.join(os.path.dirname(os.path.abspath(__file__)), "taught"))


class WorldConfirmer:
    """a sighting counts after `need` of the last `n` looks put the object within `radius_m` of the same world
    spot. World space (not pixels) so it keeps working while the wearer walks or turns."""
    def __init__(self, n=5, need=3, radius_m=0.5, max_age=2.5):
        self.n, self.need, self.r, self.max_age, self.hist = n, need, radius_m, max_age, {}
        self.lock = threading.Lock()

    def confirmed(self, label, pos, now):
        with self.lock:
            h = self.hist.setdefault(label, collections.deque(maxlen=self.n))
            h.append((now, None if pos is None else np.asarray(pos, float)))
            if pos is None:
                return False
            return sum(1 for t, q in h if q is not None and now - t <= self.max_age
                       and np.hypot(*(q - h[-1][1])) <= self.r) >= self.need

    def reset(self, label=None):
        with self.lock:
            self.hist.clear() if label is None else self.hist.pop(label, None)


CONF = WorldConfirmer()                      # a target counts after 3 of the last 5 looks agree (in world space)
DEPTH_SCALE = geometry.DepthScale()          # pins the depth model to real metres (ARKit points, else the floor)
REACH_GATE = geometry.ReachGate(reach_m=0.9)    # belt -> ring hand-over at arm's length
TEACH_STATE = {"ready": False, "error": ""}      # CLIP image model, loaded at startup (not on the first "this is my ...")
REACH_DBG = {}                                   # why reach did / didn't fire, for the dashboard
ROTCV = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
GRID = None


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
    a = wrap(-angle if FIELD["mirror_belt"] else angle)
    if abs(a) > 150:
        return {4: (duty, hz), 5: (duty, hz)}
    m = min(MOTOR_ANGLES, key=lambda k: abs(wrap(MOTOR_ANGLES[k] - a)))
    return {m: (duty, hz)}


# ----------------------------------------------------------------- talking back
def say(text):
    print("SAY:", text)
    log_event("said", text)
    with S.lock:
        S.last_said = text
    send_phone({"type": "say", "text": text})


def send_phone(obj):
    ws, loop = PHONE["ws"], PHONE["loop"]
    if ws is not None and loop is not None:
        asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps(obj)), loop)


def say_hint(text, every):
    quiet = FIELD["voice"] == "quiet"
    gap = every * (3.0 if quiet and text == LAST.get("hint_text") else 1.6 if quiet else 1.0)
    if time.time() - LAST["hint"] > gap:
        LAST["hint"], LAST["hint_text"] = time.time(), text
        say(text)


# ----------------------------------------------------------------- Gemini brain
def clean_target(target):
    """'my dinosaur?' -> 'dinosaur', so replies read 'Your dinosaur is ...' and not 'Your my dinosaur'."""
    t = re.sub(r"[^a-z0-9' -]", " ", (target or "").lower())
    t = re.sub(r"^(?:(?:my|the|a|an|your|our|his|her|their|this|that|these|those|some|nearest|closest|"
               r"near|nearby)\s+)+", "", t.strip())
    t = re.sub(r"(?:\s+(?:please|now|thanks|thank you|for me|go|gone|at|again|real quick|quickly|is|are))+$", "", t)
    return " ".join(t.split())


def simple_intent(text):
    t = re.sub(r"[^a-z0-9' ]", " ", text.lower()).strip()      # drop ?!., so "my keys?" still matches
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
        it["target"] = clean_target(it["target"])
        if 0 < len(it["target"].split()) <= 3:          # longer = probably extra words; let Gemini parse it
            return it
    if it["intent"] == "follow" and re.search(r"follow (me|them|him|her|us)\b", t):
        return it
    if not it["intent"] and re.match(r"(what|which|who|how|why|is|are|does|do|can you (read|tell)|read|tell me)\b", t):
        return {"intent": "ask", "target": ""}          # a question about the scene ("where" questions are finds)
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
        self.models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK if m != GEMINI_MODEL]   # smart tier
        self.light = list(GEMINI_LIGHT)                                                    # light tier
        self.think, self.model_paused = {}, {}            # per model: THINKS index it accepts, paused-until time
        self.calls, self.paused_until, self.intents = [], 0.0, {}   # call times (last minute), 429 pause, cache
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            from google import genai
            from google.genai import types
            # no SDK retries and a short timeout: a busy model hands over to the next one instead of stalling
            self.client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=int(GEMINI_TIMEOUT_S * 1000), retry_options=types.HttpRetryOptions(attempts=1)))
            self.types = types
            print("Gemini ready - asks:", ", then ".join(self.models), "| background:", ", then ".join(self.light))
        else:
            print("GEMINI_API_KEY not set - finding uses the local detector only; no describe")

    def _ask(self, parts, as_json=True, background=False, models=None):
        """Tries each model in `models` (default: the smart tier). background=True calls use the light tier and are
        skipped (returns None) when over the per-minute budget or while every model is paused."""
        models = models or (self.light if background else self.models)
        now = time.time()
        with GEMINI_LOCK:
            self.calls = [t for t in self.calls if now - t < 60]
            if background and (now < self.paused_until or len(self.calls) >= GEMINI_MAX_PER_MIN):
                return None
            self.calls.append(now)
        kw = dict(response_mime_type="application/json" if as_json else "text/plain", temperature=0.2)
        last = None
        for model in models:
            if time.time() < self.model_paused.get(model, 0):
                continue                                 # busy, out of quota or unknown a moment ago
            while True:
                # skip Gemini's 'thinking' step (much faster). Older Flash takes a budget of 0, Gemini 3 a level;
                # a model that rejects the setting answers with a bare 400, so we step down to the next one.
                i = self.think.get(model, 0)
                cfg = dict(kw, thinking_config=self.types.ThinkingConfig(**THINKS[i])) if THINKS[i] else kw
                t0 = time.time()
                try:
                    with GEMINI_LOCK:
                        r = self.client.models.generate_content(model=model, contents=parts,
                                                                config=self.types.GenerateContentConfig(**cfg))
                except Exception as e:
                    last, code = e, getattr(e, "code", None)
                    if code in (401, 403):
                        raise                            # bad key: no model will work
                    if code == 400 and i < len(THINKS) - 1:
                        self.think[model] = i + 1
                        print(f"{model} rejected the thinking setting; retrying with", THINKS[i + 1])
                        continue
                    busy = code in (408, 429, 500, 503, 504) or "RESOURCE_EXHAUSTED" in str(e) \
                        or "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()
                    daily = code == 429 and "PerDay" in str(e)   # out of today's quota: no point asking again soon
                    self.model_paused[model] = time.time() + (3600 if daily or not busy else 60)
                    if daily:
                        print(f"{model} is out of today's quota - skipping it for an hour")
                    print(f"{model} failed ({code or type(e).__name__}) after {time.time() - t0:.1f} s"
                          + (" - trying the next model" if model != models[-1] else ""))
                    break
                LAST["gemini"] = time.time()
                print(f"Gemini ({model}) answered in {time.time() - t0:.1f} s "
                      f"({len(self.calls)} calls in the last minute)")
                return json.loads(r.text) if as_json else r.text.strip()
        self.paused_until = time.time() + 60             # every model failed: background calls wait a minute
        print("No Gemini model answered - background scans paused for 60 s")
        raise last or RuntimeError("every Gemini model is paused")

    def _img(self, frame):
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        return self.types.Part.from_bytes(data=jpg, mime_type="image/jpeg")

    def intent(self, text):
        fast = fast_intent(text)
        if fast is not None:
            return fast
        if not self.client:
            return simple_intent(text)
        key = text.strip().lower()
        if key in self.intents:                          # same words as before: no need to ask again
            return self.intents[key]
        try:
            out = self._ask([INTENT_PROMPT + text], models=self.light + self.models)   # Lite is fine at this
            if isinstance(out, dict):
                self.intents[key] = out
            return out if isinstance(out, dict) else simple_intent(text)
        except Exception as e:
            print("intent error:", e)
            return simple_intent(text)

    def _boxes(self, frame, prompt, background=False, models=None):
        if not self.client or frame is None:
            return []
        try:
            items = self._ask([self._img(frame), prompt], background=background, models=models)
        except Exception as e:
            print("gemini error:", e)
            return []
        out = []
        for it in items if isinstance(items, list) else []:
            try:
                y0, x0, y1, x1 = it["box_2d"]
                if float(it.get("confidence", 1.0)) < (GEMINI_LIGHT_FIND_CONF if background else GEMINI_FIND_CONF):
                    print(f"Gemini unsure ({it.get('confidence')}) about {it.get('label')!r} - ignored")
                    continue
                out.append({"label": str(it.get("label", "")).lower(),
                            "box": [x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000]})
            except Exception:
                continue
        return out

    def find(self, frame, target, gemini_ok=True, background=False):
        """local detector first; Gemini only if it finds nothing. Items from the detector thread carry the
        "meta" of the frame they were found in (use it for the position, not the newest meta)."""
        items = local_find(frame, target)
        if items:
            with S.lock:
                S.last_finder = f"YOLO-World ({items[0]['conf']:.2f})"
            return items
        if not gemini_ok:
            return []
        items = self._boxes(frame, FIND_PROMPT.format(target=target), background)
        if items:
            with S.lock:
                S.last_finder = "Gemini"
        return items

    def scan(self, frame):
        return self._boxes(frame, SCAN_PROMPT, background=True)

    def answer(self, frame, question):
        if self.client and frame is not None:
            try:
                return self._ask([self._img(frame), ANSWER_PROMPT + question], as_json=False,
                                 models=self.models + self.light)      # Lite has a far bigger daily quota
            except Exception as e:
                print("answer error:", e)
        return local_scene()

    def describe(self, frame):
        if self.client and frame is not None:
            try:
                return self._ask([self._img(frame), DESCRIBE_PROMPT], as_json=False, models=self.models + self.light)
            except Exception as e:
                print("describe error:", e)
        return local_scene()


def local_scene():
    """offline answer from what the detectors see right now: 'I see a chair ahead and your keys slightly left.'"""
    with S.lock:
        dets, meta = list(S.dets), S.meta
    items = [(d["name"], d["box"]) for d in dets if d.get("conf", 1) > 0.4]
    if DET is not None:
        vd = DET.latest()[0]
        items += [(d["label"], d["box"]) for d in vd if d["label"] in finder.MEMORY_LABELS and d.get("conf", 0) > 0.25]
    if not items or meta is None:
        return "I don't see anything I recognize right now."
    seen, parts = set(), []
    for name, b in sorted(items, key=lambda it: -(it[1][3] - it[1][1])):        # biggest (nearest) first
        if name in seen or len(parts) >= 4:
            continue
        seen.add(name)
        rel = px_bearing((b[0] + b[2]) / 2, meta)
        where = "ahead" if abs(rel) < 12 else ("slightly " if abs(rel) < 30 else "to your ") + ("right" if rel > 0 else "left")
        parts.append(f"{'your ' if name in finder.DEMO else 'a '}{name} {where}")
    return "I see " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + "."


SEAT_CLASSES = ("chair", "couch", "bench")


def empty_seats(dets):
    """chairs/couches/benches with nobody sitting on them (no person box over the seat's middle)."""
    people = [d["box"] for d in dets if d["name"] == "person"]
    out = []
    for d in dets:
        if d["name"] not in SEAT_CLASSES:
            continue
        x0, y0, x1, y1 = d["box"]
        cx, cy = (x0 + x1) / 2, y0 + 0.6 * (y1 - y0)
        taken = any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] + 0.1 * (p[3] - p[1]) for p in people)
        if not taken:
            out.append(d)
    return sorted(out, key=lambda d: -d["box"][3])                              # nearest (lowest in frame) first


def start_seat():
    frame, meta, depth = latest()
    if frame is None:
        say("I'm not getting any video yet.")
        return
    with S.lock:
        dets = list(S.dets)
    seats = empty_seats(dets)
    if not seats:
        set_mode("SEARCH", {"label": "empty seat", "kind": "seat", "pos": None, "box": None, "in_view": False,
                            "walk_only": True, "started": time.time()})
        say("I don't see an empty seat yet. Turn slowly and I'll keep looking.")
        return
    tgt = dict(lock_target("empty seat", seats[0]["box"], meta, depth, walk_only=True), kind="seat")
    heading, _, pos = pose(meta)
    set_mode("GOTO", tgt)
    say(f"There's an empty {seats[0]['name']} {direction_words(wrap(heading_to(pos, tgt['pos'])[0] - heading))}, "
        f"about {max(1, round(tgt['dist']))} meters away. Follow the belt.")


def tile_detect(frame):
    """look again, zoomed in: the two lower corners (table/floor items) and the horizon band (far items). The
    detector scales each crop up to its input size, so a crop of the full-resolution frame gains real detail."""
    h, w = frame.shape[:2]
    th, tw = int(h * 2 / 3), int(w * 0.55)
    crops = [(0, h - th, tw, th), (w - tw, h - th, tw, th),                      # lower left / lower right
             (int(w * 0.2), int(h * 0.28), int(w * 0.6), int(h * 0.45))]         # middle band: far away
    out = []
    for x0, y0, cw, ch in crops:
        for d in DET.det.detect(np.ascontiguousarray(frame[y0:y0 + ch, x0:x0 + cw])):
            b = d["box"]
            out.append(dict(d, box=[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]))
    return out


EXIT_WORDS = ("exit", "exit door", "way out", "emergency exit", "the exit")
DET_FAST_HZ = float(os.environ.get("DET_FAST_HZ", "8"))     # detector looks/s while finding or guiding
DET_IDLE_HZ = float(os.environ.get("DET_IDLE_HZ", "3"))     # ...and otherwise (memory stays fresh, GPU left to steering)
LEARNED_SIM = float(os.environ.get("LEARNED_SIM", "0.72"))   # how alike a crop must be to an auto-learned item
WHERE_CLAUSE = re.compile(r"\s+(?:on|in|at|near|by|under|next to|beside|behind|inside|over|from)\s+(?:the|a|my|this|that)?\s*\S.*$")


def open_queries(target):
    """words to try with the open-vocabulary detector, most specific first."""
    if target in EXIT_WORDS:
        return ["exit sign", "exit door"]
    core = WHERE_CLAUSE.sub("", target).strip() or target           # "green dragon on the desk" -> "green dragon"
    qs = [core]
    if len(core.split()) > 1:
        qs.append(core.split()[-1])                                   # ...then just "dragon"
    return qs


def local_find(frame, target, after_t=None, zoom=False):
    """fixed-vocabulary detector thread (distractor-aware, taught items re-ranked); open-vocab for other words."""
    if DET is None:
        return LOCAL.detect(frame, [target]) if LOCAL is not None and LOCAL.model is not None else []
    got = DET.wait_newer(after_t, 0.6) if after_t is not None else None
    dets, meta, _, t, fr = got + (DET.frame,) if got is not None else DET.latest(with_frame=True)
    if fr is None or time.time() - t > 1.0:
        return []
    if finder.canonical(target) or (TEACH.has(target) and TEACH.classes(target)):   # vocabulary / taught-by-hand
        best = finder.pick_target(dets, target, TEACH, fr)
        if best is None and zoom:
            best = finder.pick_target(tile_detect(fr), target, TEACH, fr)   # small item: look again, zoomed in
            if best is not None:
                best = dict(best, zoomed=True)
        if best is None and zoom and STRONG_DET is not None:              # fast model missed: ask the big one once
            best = finder.pick_target(STRONG_DET.detect(fr), target, TEACH, fr)
            if best is not None:
                best = dict(best, strong=True)
                with S.lock:
                    S.last_finder = "YOLO-World L"
        if best is not None and finder.canonical(target) == "door":
            doors = [d for d in dets if d["label"] == "door"]
            best = max(doors, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), default=best)
        return [dict(best, meta=meta, label=target)] if best else []
    if TEACH.has(target):                                # learned from a Gemini find: match its look
        props = list(dets)
        for q in open_queries(target):
            try:
                props += DET.det.detect_open(fr, q, conf=0.08)
            except Exception as e:
                print("open-vocab error:", e)
        props = [d for d in props if (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) < 0.5 * fr.shape[0] * fr.shape[1]]
        if props:
            sims = TEACH.similarity_boxes(target, fr, [d["box"] for d in props])
            i = int(np.argmax(sims))
            if sims[i] >= LEARNED_SIM:
                return [dict(props[i], meta=meta, label=target, sim=sims[i])]
        return []
    found = []
    for q in open_queries(target):                       # most specific first, then the plain noun
        try:
            found = DET.det.detect_open(fr, q)
        except Exception as e:
            print("open-vocab error:", e)
            return []
        if found:
            break
    if not found and target in EXIT_WORDS:               # no exit sign in view: the nearest door will do
        found = [d for d in dets if d["label"] == "door"]
    if target in EXIT_WORDS or target == "door":        # "nearest": biggest box = closest
        found = sorted(found, key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    else:
        found = sorted(found, key=lambda d: -d["conf"])
    return [dict(found[0], meta=meta, label=target)] if found else []


def confirmed_find(frame, target, budget_s=0.8):
    """for a spoken find: watch a few detector results and lock only on a confirmed sighting (or the best one
    seen if time runs out), so a single wrong box doesn't send the wearer to the wrong object."""
    CONF.reset(target)
    if DET is not None:
        DET.set_rate(DET_FAST_HZ)                       # a spoken find wants every look it can get, now
    end, best, t = time.time() + budget_s, None, time.time() - 0.01
    while time.time() < end:
        items = local_find(frame, target, after_t=t, zoom=True)
        t = time.time()
        pos = lock_target(target, items[0]["box"], items[0].get("meta") or S.meta, S.depth)["pos"] if items else None
        if CONF.confirmed(target, pos, t) and items:
            return items
        if items and (best is None or items[0].get("score", items[0]["conf"]) > best[0].get("score", best[0]["conf"])):
            best = items
        if DET is None:
            break
    return best or []


# ----------------------------------------------------------------- perception thread
N_COLS = 64


def obstacle_columns(depth, dets, meta):
    """nearest obstacle distance in each of 64 image columns. With depth, only points between ankle and
    head height count (so the floor never shows up as a wall). Without depth, YOLO box bottoms are used."""
    if depth is not None:
        return geometry.obstacle_columns_any(depth, meta, CAM_HEIGHT_M, N_COLS, (OBST_MIN_H, OBST_MAX_H),
                                             flip=FLIP_IMAGE)
    cols = np.full(N_COLS, 9.9, np.float32)
    for det in dets:
        x0, y0, x1, y1 = det["box"]
        dist = floor_dist(y1, meta) if y1 < H - 3 else 0.8
        c0, c1 = max(0, int(x0 * N_COLS / W)), min(N_COLS - 1, int(x1 * N_COLS / W))
        cols[c0:c1 + 1] = np.minimum(cols[c0:c1 + 1], dist)
    return cols


def compute_sectors(depth, dets, meta):
    cols = obstacle_columns(depth, dets, meta)
    dist = [float(cols[s * 4:(s + 1) * 4].min()) for s in range(16)]
    return [x < OBSTACLE_M for x in dist], dist, cols


# ----------------------------------------------------------------- floor map + A* planner
class FloorMap:
    """top-down occupancy grid in ARKit world coordinates (x right, z backward; start point = centre)."""
    def __init__(self):
        self.n = int(MAP_SIZE_M / MAP_CELL_M)
        self.val = np.zeros((self.n, self.n), np.float32)      # >0 occupied evidence, <0 free evidence
        self.lock = threading.Lock()
        self.half = MAP_SIZE_M / 2

    def reset(self):
        with self.lock:
            self.val[:] = 0

    def to_cell(self, p):
        return int((p[0] + self.half) / MAP_CELL_M), int((p[1] + self.half) / MAP_CELL_M)   # (col, row)

    def to_world(self, c):
        return np.array([(c[0] + 0.5) * MAP_CELL_M - self.half, (c[1] + 0.5) * MAP_CELL_M - self.half])

    def inside(self, c):
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n

    def update(self, meta, cols):
        heading, _, pos = pose(meta)
        with self.lock:
            self.val *= 0.997                                  # slowly forget, so moved things disappear
            for i, d in enumerate(cols):
                bearing = px_bearing((i + 0.5) * W / N_COLS, meta)
                wh = math.radians(heading + bearing)
                direction = np.array([math.sin(wh), -math.cos(wh)])
                hit = d < MAP_RANGE_M
                reach = min(float(d), MAP_RANGE_M)
                for t in np.arange(0.3, reach - MAP_CELL_M, MAP_CELL_M):       # free space along the ray
                    c = self.to_cell(pos + t * direction)
                    if self.inside(c):
                        self.val[c[1], c[0]] = max(-3.0, self.val[c[1], c[0]] - 0.3)
                if hit:
                    c = self.to_cell(pos + reach * direction)
                    if self.inside(c):
                        self.val[c[1], c[0]] = min(5.0, self.val[c[1], c[0]] + 1.0)
            c = self.to_cell(pos)                               # where you stand is free
            r = int(0.4 / MAP_CELL_M)
            if self.inside(c):
                self.val[max(0, c[1] - r):c[1] + r + 1, max(0, c[0] - r):c[0] + r + 1] = np.minimum(
                    self.val[max(0, c[1] - r):c[1] + r + 1, max(0, c[0] - r):c[0] + r + 1], -1.0)

    def blocked_grid(self):
        with self.lock:
            occ = (self.val > 1.5).astype(np.uint8)
        k = 2 * int(math.ceil(BODY_RADIUS_M / MAP_CELL_M)) + 1
        return cv2.dilate(occ, np.ones((k, k), np.uint8)), occ

    def plan(self, start, goal):
        """A* over the inflated grid; returns a list of world points from start to goal, or None."""
        import heapq
        blocked, occ = self.blocked_grid()
        clearance = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 3) * MAP_CELL_M
        s, g = self.to_cell(start), self.to_cell(goal)
        if not (self.inside(s) and self.inside(g)):
            return None
        blocked[s[1], s[0]] = 0
        if blocked[g[1], g[0]]:                                 # goal inside an obstacle (e.g. the door frame):
            free = np.argwhere(blocked == 0)                    # aim for the nearest free cell instead
            if not len(free):
                return None
            k = np.argmin((free[:, 1] - g[0]) ** 2 + (free[:, 0] - g[1]) ** 2)
            g = (int(free[k, 1]), int(free[k, 0]))
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for dx, dy, w in steps:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < self.n and 0 <= ny < self.n) or blocked[ny, nx]:
                    continue
                near = max(0.0, 0.8 - float(clearance[ny, nx]))          # prefer routes away from obstacles
                nc = cost[cur] + w * (1.0 + 3.0 * near)
                if nc < cost.get((nx, ny), 1e9):
                    cost[(nx, ny)] = nc
                    came[(nx, ny)] = cur
                    heapq.heappush(openq, (nc + math.hypot(g[0] - nx, g[1] - ny), (nx, ny)))
        if g not in came:
            return None
        path, c = [], g
        while c is not None:
            path.append(self.to_world(c))
            c = came[c]
        return path[::-1]

    def render(self, meta, path, target, pins=()):
        with self.lock:
            v = self.val.copy()
        img = np.full((self.n, self.n, 3), 130, np.uint8)
        img[v < -0.5] = (235, 235, 235)
        img[v > 1.5] = (25, 25, 25)
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)
        sc = 480 / self.n
        px = lambda p: (int((p[0] + self.half) / MAP_CELL_M * sc), int((p[1] + self.half) / MAP_CELL_M * sc))
        if path:
            cv2.polylines(img, [np.array([px(p) for p in path], np.int32)], False, (0, 140, 255), 3)
        if target is not None:
            cv2.circle(img, px(target), 9, (40, 40, 220), -1)
        for pin in pins:                                  # remembered objects
            c = px((pin["x"], pin["z"]))
            col = (150, 150, 150) if pin.get("stale") else (200, 90, 220)
            cv2.drawMarker(img, c, col, cv2.MARKER_DIAMOND, 16, 3)
            cv2.putText(img, pin["label"], (c[0] + 9, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3)
            cv2.putText(img, pin["label"], (c[0] + 9, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if meta is not None:
            heading, _, pos = pose(meta)
            p = px(pos)
            tip = px(pos + 0.6 * np.array([math.sin(math.radians(heading)), -math.cos(math.radians(heading))]))
            cv2.circle(img, p, 8, (60, 200, 60), -1)
            cv2.line(img, p, tip, (60, 200, 60), 3)
        cv2.putText(img, "black = obstacle, orange = route, red = goal, green = you, purple = memory", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def tracking_ok(meta):
    return meta is not None and meta.get("T") is not None and meta.get("tracking", True)


def planner_loop():
    while True:
        time.sleep(0.4)
        if GRID is None:
            continue
        with S.lock:
            mode, tgt, meta = S.mode, S.target, S.meta
        path = None
        if USE_PLANNER and tracking_ok(meta) and mode == "GOTO" and tgt and tgt.get("pos") is not None:
            try:
                path = GRID.plan(pose(meta)[2], tgt["pos"])
            except Exception as e:
                print("planner error:", e)
        with S.lock:
            S.path, S.path_t = path, time.time()
        if tracking_ok(meta):
            S.map_vis = GRID.render(meta, path, tgt.get("pos") if tgt else None, MEM.pins(time.time()))


def route_bearing(meta, path):
    """pure pursuit: direction (body angle) to the route point LOOKAHEAD_M ahead of the wearer."""
    heading, _, pos = pose(meta)
    look = path[-1]
    for p in path:
        if np.hypot(*(p - pos)) >= LOOKAHEAD_M:
            look = p
            break
    return wrap(heading_to(pos, look)[0] - heading)


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
    worker = None
    if USE_DEPTH:
        try:                                            # TensorRT engines if built, else HF fp16; own thread
            worker = depth_trt.DepthWorker(depth_trt.make_depth(os.path.dirname(os.path.abspath(__file__))))
        except Exception as e:
            print("Depth model off:", e)
    mem_log("tracker + depth loaded")
    CORE_READY.set()
    last_id, n, t0, depth, dmeta = -1, 0, time.time(), None, None
    ms = collections.defaultdict(float)
    while True:
        with S.lock:
            fid, frame, meta = S.frame_id, S.frame, S.meta
            pd, pdt = S.phone_depth, S.phone_depth_t
        if frame is None or fid == last_id:
            time.sleep(0.005)
            continue
        last_id = fid
        ta = time.time()
        if worker is not None:
            worker.submit(frame, meta, fid)             # never blocks; the worker keeps only the newest frame
        if DET is not None:
            DET.submit(frame, meta, fid)                # fixed-vocab YOLO-World, ~8 Hz, own thread
        res = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
            for b, c, cf, i in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), ids):
                dets.append({"box": b, "name": model.names[c], "conf": cf, "id": i})
        tb = time.time()
        source = "none"
        if pd is not None and time.time() - pdt < 0.5:
            depth, dmeta, source = pd, meta, "LiDAR"
        elif worker is not None:
            got = worker.latest()
            if got is not None and got[0].shape[:2] == frame.shape[:2]:
                depth, dmeta, _, _, ms["depth_ms"] = got[0], got[1], got[2], got[3], got[4]
                depth = scale_depth(depth, dmeta)
                source = "estimated"
            else:
                depth = None
        else:
            depth = None
        gmeta = dmeta if depth is not None and dmeta is not None else meta   # geometry uses the depth frame's pose
        blocked, dist, cols = compute_sectors(depth, dets, gmeta)
        tc = time.time()
        if USE_PLANNER and GRID is not None and tracking_ok(gmeta):
            GRID.update(gmeta, cols)
        td = time.time()
        n += 1
        with S.lock:
            S.dets, S.depth, S.blocked, S.dist, S.depth_source = dets, depth, blocked, dist, source
            if n % 15 == 0:
                S.fps = 15 / max(1e-3, time.time() - t0)
        if n % 15 == 0:
            t0 = time.time()
        draw(frame, meta, dets, blocked, depth)
        te = time.time()
        for k, v in (("yolo_ms", tb - ta), ("sectors_ms", tc - tb), ("map_ms", td - tc), ("draw_ms", te - td)):
            ms[k] = 0.9 * ms[k] + 100 * v                # EMA of milliseconds
        if n % 5 == 0:
            with S.lock:
                S.timing = {k: round(v, 1) for k, v in ms.items()}
                if DET is not None:
                    S.timing["world_ms"] = round(DET.ms, 1)
            if DET is not None:                         # full-speed detector only while it matters
                DET.set_rate(DET_FAST_HZ if S.mode in ("SEARCH", "GOTO", "REACH") else DET_IDLE_HZ)
        if n % 75 == 0:
            print("timing:", dict({k: round(v, 1) for k, v in ms.items()}, world_ms=round(DET.ms, 1) if DET else None), f"fps {S.fps:.1f}",
                  f"depth {source} scale {DEPTH_SCALE.scale:.2f} ({DEPTH_SCALE.source or 'none'})")


def scale_depth(depth, meta):
    """per-frame metric scale correction: ARKit feature points when the phone sends them, else the floor."""
    if tracking_ok(meta):
        DEPTH_SCALE.update(depth, meta, CAM_HEIGHT_M)
    with S.lock:
        S.depth_scale = DEPTH_SCALE.scale
    return DEPTH_SCALE.apply(depth)


# ----------------------------------------------------------------- modes and targets
def set_mode(mode, target=None, keep_target=False):
    with S.lock:
        S.mode = mode
        if not keep_target:
            S.target = target
    REACH_GATE.reset()
    if mode == "IDLE":
        BELT.set({})


def latest():
    with S.lock:
        return (None if S.frame is None else S.frame.copy()), S.meta, S.depth


def lock_target(label, box, meta, depth, walk_only=False):
    heading, pitch, pos = pose(meta)
    if depth is not None and depth.shape[:2] != (int(meta.get("H", H)), int(meta.get("W", W))):
        depth = None                                     # depth from the other orientation: don't mix
    with S.lock:
        planes = list(S.planes)
    est = geometry.target_estimate(box, meta, depth, CAM_HEIGHT_M, planes or None, flip=FLIP_IMAGE)
    return {"label": label, "pos": np.array(est["pos_xz"], float), "dist": est["dist"], "height": est["height"],
            "dist_source": est["source"], "box": box, "box_heading": heading, "box_pitch": pitch,
            "box_t": time.time(), "in_view": True, "walk_only": walk_only}


def smooth_pos(tgt, new):
    """blend the last few position estimates (closer sightings count more, outliers >1 m dropped), so a far
    target doesn't jump around and the belt points steadily."""
    hist = (tgt.get("hist") or [])[-4:] + [(np.asarray(new["pos"], float), float(new.get("dist") or 1.5))]
    P = np.array([q for q, _ in hist])
    w = np.array([1.0 / max(d, 0.5) ** 2 for _, d in hist])
    keep = np.hypot(*(P - np.median(P, axis=0)).T) < 1.0
    if keep.any():
        new["pos"] = (P[keep] * w[keep, None]).sum(axis=0) / w[keep].sum()
    new["hist"] = hist
    return new


def start_reach(tgt):
    LAST["inside_since"], LAST["reach_t"] = 0.0, time.time()
    BELT.set({})
    set_mode("REACH", tgt)
    say("It's right in front of you. Reach out, and the ring will guide your hand.")


def gemini_online():
    return BRAIN.client is not None and time.time() > getattr(BRAIN, "paused_until", 0)


def auto_learn(target, frame, box):
    """remember what Gemini's find looks like, so local tracking follows that exact object from now on."""
    if not TEACH_STATE["ready"]:
        return
    try:
        TEACH.teach(target, frame, box)
        log_event("info", f"Learned what your {target} looks like")
    except Exception as e:
        print("auto-learn failed:", e)


def start_find(target, walk_only=False, ask_first=False, desc=None):
    own = "the" if walk_only else "your"
    frame, meta, depth = latest()
    if frame is None:
        say("I'm not getting any video yet.")
        return
    if ask_first and recall(target):                   # a memory question: answer from memory even if not in view
        items = []
    else:
        unusual = not finder.canonical(target) and not TEACH.has(target) and target not in EXIT_WORDS
        items = []
        if unusual and gemini_online():                  # Gemini knows "green dragon"; the local model may not
            say(f"Looking for {own} {target}.")
            items = BRAIN._boxes(frame, FIND_PROMPT.format(target=desc or target), models=BRAIN.models + BRAIN.light)
            if items:
                with S.lock:
                    S.last_finder = "Gemini"
                threading.Thread(target=auto_learn, args=(target, frame.copy(), items[0]["box"]), daemon=True).start()
        if not items:
            items = confirmed_find(frame, target)
        if not items and not unusual and gemini_online():
            items = BRAIN._boxes(frame, FIND_PROMPT.format(target=desc or target), models=BRAIN.models + BRAIN.light)
    if items:
        meta = items[0].get("meta") or meta
        tgt = lock_target(target, items[0]["box"], meta, depth, walk_only)
        if not walk_only and tracking_ok(meta):
            for _ in range(2):
                MEM.add(target, tgt["pos"], tgt.get("height"), items[0].get("conf", 0.5), cam_pos_xz=pose(meta)[2])
        rel = wrap(heading_to(pose(meta)[2], tgt["pos"])[0] - pose(meta)[0])
        if tgt["dist"] < REACH_GATE.reach_m and tgt.get("dist_source") != "floor-guess" and not walk_only:
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
        if ask_first:                                   # "where did I leave...": answer, then offer to guide
            set_mode("IDLE")
            with S.lock:
                S.pending = {"type": "goto", "target": target, "walk_only": walk_only, "expires": time.time() + 12,
                             "mem": dict(mem)}
            say(MEM.describe(mem, heading, pos, time.time()) + " Want me to take you there?")
            return
        rel = wrap(heading_to(pos, mem["pos"])[0] - heading)
        set_mode("GOTO", {"label": target, "pos": mem["pos"], "dist": None, "box": None, "in_view": False,
                          "box_t": 0.0, "walk_only": walk_only})
        say(MEM.describe(mem, heading, pos, time.time()) + " Follow the belt." if not walk_only
            else f"I last saw {own} {target} {direction_words(rel)}. Follow the belt.")
        return
    set_mode("SEARCH", {"label": target, "pos": None, "box": None, "in_view": False, "walk_only": walk_only,
                        "started": time.time()})
    say(f"I can't see {own} {target} yet. Turn slowly and I'll keep looking.")


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
    return MEM.recall(target)


YES_RE = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok(ay)?|please|go ahead|do it|take me|let'?s go)\b")
NO_RE = re.compile(r"^\W*(no|nope|nah|cancel|never ?mind|not now|stop)\b")
WHERE_RE = re.compile(r"\b(where|did i leave|did i put|last see|last saw)\b")


TEACH_RE = re.compile(r"^\W*(?:this is|these are|here(?: is|'s| are)|remember this as|learn what|learn|memorize)\s+(?:my |the )?(.+?)(?: looks? like)?[.!?]*$")
FORGET_RE = re.compile(r"^\W*forget (?:my |the |what )?(.+?)(?: looks? like)?[.!?]*$")


LOW_MEM = os.environ.get("LOW_MEM", "0") == "1"   # no strong model; item-teaching model loads on first use
CORE_READY = threading.Event()                       # set when the perception models are loaded


def mem_log(tag):
    """free memory after each model loads, so a crash log shows where memory ran out."""
    try:
        info = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        avail = int(info["MemAvailable"].split()[0]) // 1024
        print(f"[mem] {tag}: {avail} MB available")
    except Exception:
        pass


def warmup():
    """load the optional models one by one once the core runs, instead of all at once at startup."""
    CORE_READY.wait(180)
    if STRONG_DET is not None:
        STRONG_DET.detect(np.zeros((480, 640, 3), np.uint8))
        mem_log("strong detector loaded")
    if not LOW_MEM:
        warm_teach()
        mem_log("teach model loaded")


def warm_teach():
    """load the CLIP image model in the background at startup, so teaching answers at once."""
    free = finder.mem_available_mb()
    if free is not None and free < 900:                 # better no teaching than a crashed server
        TEACH_STATE["error"] = f"only {free} MB free"
        print("Teach model skipped:", TEACH_STATE["error"])
        return
    try:
        TEACH.embed([np.zeros((32, 32, 3), np.uint8)])
        TEACH_STATE["ready"] = True
        print("Teach model ready")
    except Exception as e:
        TEACH_STATE["error"] = str(e)[:120]
        print("Teach model failed to load:", e)


def teach_item(label):
    """'this is my wallet': store what the item looks like from a few frames (CLIP), so finds prefer it."""
    if not label or DET is None:
        say("I can't learn items right now. The detector isn't running.")
        return
    if not TEACH_STATE["ready"] and LOW_MEM and not TEACH_STATE["error"]:
        say("Loading the item model, one moment.")
        warm_teach()                                    # low-memory mode: load on first use
    if not TEACH_STATE["ready"]:
        say("The item learning model isn't installed on the Orin." if TEACH_STATE["error"]
            else "I'm still loading the item model. Try again in a moment.")
        return
    DET.set_rate(DET_FAST_HZ)
    say(f"Hold your {label} in front of the camera.")
    time.sleep(1.5)                                     # give them time to hold it up
    n, t = 0, time.time()
    for _ in range(4):
        got = DET.wait_newer(t, 1.0)
        if got is None:
            continue
        dets, meta, _, t = got
        frame = DET.frame
        wh = (frame.shape[1], frame.shape[0])
        d = finder.pick_teach_box(dets, (meta or {}).get("hands"), wh)
        box = d["box"] if d else finder.default_teach_box((meta or {}).get("hands"), wh)
        try:
            n = TEACH.teach(label, frame, box, det_class=d["label"] if d else None)
        except Exception as e:
            print("teach error:", e)
            say("I couldn't learn that one. Is the image model installed?")
            return
        time.sleep(0.25)
    say(f"Got it. I'll know your {label} now." if n else f"I couldn't see your {label}. Try again closer to the camera.")


WAKE_RE = re.compile(r"^\W*(?:hey |ok |okay )?(?:omni[ -]?belt|omni|belt)\b[\s,.!?]*")
FIXES = {"fine": "find", "fined": "find", "fight": "find", "wallets": "wallet", "walla": "wallet", "wallace": "wallet",
         "walt": "wallet", "key": "keys", "keyes": "keys", "kees": "keys", "keith": "keys", "bottles": "bottle",
         "botel": "bottle", "model": "bottle", "rich": "reach", "breach": "reach", "speech": "reach",
         "yeah": "yes", "yep": "yes", "nope": "no", "dory": "door", "gin": "keys"}
LEXICON = ["find", "where", "take", "follow", "stop", "walk", "describe", "reach", "yes", "forget", "remember", "learn",
           "keys", "wallet", "bottle", "water", "door", "chair", "glasses", "backpack", "laptop", "phone", "leave"]


def fix_words(text):
    """undo common speech-to-text slips for our small vocabulary ('fine my walla' -> 'find my wallet')."""
    import difflib
    t = WAKE_RE.sub("", text.lower().strip())
    out = []
    for w in t.split():
        core = re.sub(r"[^a-z']", "", w)
        if core in FIXES:
            w = FIXES[core]
        elif len(core) >= 5 and core not in LEXICON:
            m = difflib.get_close_matches(core, LEXICON, n=1, cutoff=0.8)
            w = m[0] if m else w
        out.append(w)
    return " ".join(out)


REACH_RE = re.compile(r"^\W*(?:reach|reach mode|i'?m here|i am here|it'?s (?:close|here|right here)|grab it|i'?m close)\b")


SEAT_RE = re.compile(r"\b(seat|somewhere to sit|place to sit|empty chair|where can i sit|sit down)\b")


def mentioned_item(t):
    """an item the user cares about, named anywhere in the sentence (vocabulary or something remembered)."""
    words = re.sub(r"[^a-z ]", " ", t).split()
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if words[i] in ("my", "the", "a", "it", "is", "see", "you", "do", "can", "any"):
                continue                                 # phrases must start on a content word
            lab = finder.canonical(phrase)
            if lab in finder.MEMORY_LABELS:
                return lab
            hit = MEM.recall(phrase) if len(phrase) > 3 else None
            if hit:
                return hit["label"]
    return None


def handle_command(text):
    text = fix_words(text)
    with S.lock:
        S.last_heard = text
        pending, S.pending = S.pending, None
    log_event("heard", text)
    t = text.lower().strip()
    if REACH_RE.match(t):                               # manual hand-over to the ring
        with S.lock:
            tgt = S.target
        if tgt and tgt.get("box") is not None:
            start_reach(tgt)
        elif tgt:
            say(f"Point the belt at your {tgt['label']} so I can see it, then say reach again.")
        else:
            say("What should I reach for? Say find, then the item.")
        return
    if pending and time.time() < pending["expires"]:
        if YES_RE.match(t):
            mem = pending.get("mem")
            if mem and mem.get("pos") is not None:   # go straight to the remembered spot; GOTO re-finds it live
                _, meta, _ = latest()
                heading, _, pos = pose(meta or {})
                set_mode("GOTO", {"label": pending["target"], "pos": mem["pos"], "dist": None, "box": None,
                                  "in_view": False, "box_t": 0.0, "walk_only": pending.get("walk_only", False)})
                say(MEM.describe(mem, heading, pos, time.time()) + " Follow the belt.")
            else:
                start_find(pending["target"], pending.get("walk_only", False))
            return
        if NO_RE.match(t):
            say("Okay.")
            return
    m = TEACH_RE.match(t)
    if m:
        teach_item(clean_target(m.group(1)))
        return
    m = FORGET_RE.match(t)
    if m:
        lab = clean_target(m.group(1))
        say(f"Okay, I forgot what your {lab} looks like." if TEACH.forget(lab) else f"I hadn't learned your {lab}.")
        return
    if SEAT_RE.search(t):
        start_seat()
        return
    it = BRAIN.intent(text)
    intent, target = it.get("intent", ""), clean_target(it.get("target"))
    if intent in ("", "ask", "describe"):                 # "do you see my wallet", "is my bottle around"
        item = mentioned_item(t)
        if item:
            intent, target = "find", item
    print(f"COMMAND {text!r} -> {intent} {target!r}")
    if intent == "stop":
        set_mode("IDLE")
        send_phone({"type": "hush"})                     # cut whatever the phone is saying
        say("Stopped.")
    elif intent == "walk":
        set_mode("WALK")
        say("Walking. I'll steer you around obstacles.")
    elif intent == "describe":
        frame, _, _ = latest()
        say(BRAIN.describe(frame))
    elif intent == "ask":
        frame, _, _ = latest()
        say(BRAIN.answer(frame, text))
    elif intent == "follow":
        start_follow(target)
    elif intent in ("find", "where"):
        desc, target = target, WHERE_CLAUSE.sub("", target).strip() or target   # "green dragon on the desk" -> "green dragon"
        if target in EXIT_WORDS or finder.canonical(target) == "door":
            start_find(target, walk_only=True)                     # places: walk there, no reach
            return
        start_find(target or "keys", ask_first=intent == "where" or bool(WHERE_RE.search(t)), desc=desc)
    elif intent == "goto":
        start_find(target or "door", walk_only=True)
    else:
        say("Sorry, I didn't catch that. Try: find my keys.")


# ----------------------------------------------------------------- control loop (10 Hz)
def pick_hand(hands):
    best = pick_hand_raw(hands)
    if best is not None and FIELD["mirror_hands"]:
        best = [W - best[0], best[1]]
    return best


def pick_hand_raw(hands):
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
    pad = max(25.0, 0.25 * max(bx1 - bx0, by1 - by0))   # a little forgiving: fingertips at the item's edge count
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
        heading, pitch, pos = pose(meta)
        if mode == "REACH":
            d = heading_to(pos, tgt["pos"])[1] if tgt and tgt.get("pos") is not None else None
            if REACH_GATE.should_exit(tgt, d, time.time()):
                set_mode("GOTO", tgt)                     # lost it and walked away: back to the belt
                say("Let's get a little closer. Follow the belt.")
                continue
            if time.time() - LAST.get("reach_t", time.time()) > 25:
                set_mode("IDLE")                          # don't buzz forever
                say("Stopping. Say find again if you need it.")
                continue
            reach_step(meta, tgt, meta.get("hands", []))
            continue
        goal, gdist = 0.0, None
        if mode in ("GOTO", "SEARCH"):
            if not tgt or tgt.get("pos") is None:
                BELT.set({})                                       # searching: stand still, turn slowly
                continue
            th, gdist = heading_to(pos, tgt["pos"])
            goal = wrap(th - heading)
            with S.lock:
                path, path_t = S.path, S.path_t
            if path and len(path) > 1 and time.time() - path_t < 1.5:
                goal = route_bearing(meta, path)               # follow the planned route around obstacles
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
        if mode == "GOTO" and gdist is not None and tgt.get("walk_only") and gdist < REACH_M:
            BELT.pattern("ARRIVE")
            say(f"You're at the {tgt['label']}.")
            set_mode("IDLE")
            continue
        if mode == "GOTO" and gdist is not None and not tgt.get("walk_only") and tgt.get("box") is None:
            LAST["near_since"] = (LAST.get("near_since") or time.time()) if gdist < 0.6 else 0
            if LAST["near_since"] and time.time() - LAST["near_since"] > 3:     # remembered spot reached, item unseen
                LAST["near_since"] = 0
                BELT.pattern("ARRIVE")
                set_mode("IDLE")
                say(f"You're there. Your {tgt['label']} should be right in front of you.")
                continue
        if mode == "GOTO" and gdist is not None and not tgt.get("walk_only"):
            now = time.time()
            gd = gdist if tgt.get("dist_source") != "floor-guess" else None   # table items: trust only the box size
            box = tgt.get("box")
            fresh = box is not None and now - tgt.get("box_t", 0) < 1.2
            hand = pick_hand(meta.get("hands", []))
            # the wearer's hand shows up next to the item = they're within reach
            hand_near = fresh and hand is not None and \
                box[0] - 0.6 * (box[2] - box[0]) - 60 <= hand[0] <= box[2] + 0.6 * (box[2] - box[0]) + 60
            # item low in the frame and not tiny = close, whatever the depth scale says
            low_big = fresh and box[3] > 0.72 * H and (box[3] - box[1]) > 0.08 * H
            LAST["low_big"] = LAST.get("low_big", 0) + 1 if low_big else 0
            gate = REACH_GATE.update(tgt, gd, box, H, now)
            REACH_DBG.update(dist=None if gd is None else round(gd, 2), src=tgt.get("dist_source"),
                             box_age=None if box is None else round(now - tgt.get("box_t", 0), 1),
                             streak=REACH_GATE.streak, hand=hand is not None, low=LAST["low_big"])
            if gate or hand_near or LAST["low_big"] >= 3:
                LAST["low_big"] = 0
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
        local_ok = DET is not None or (LOCAL is not None and LOCAL.model is not None)
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
        if tgt.get("kind") == "seat":                   # "find me a seat": re-find an empty chair, not a word
            with S.lock:
                seats = empty_seats(list(S.dets))
            if not seats:
                if mode == "SEARCH" and time.time() - tgt.get("started", time.time()) > 25:
                    say("I still can't find an empty seat. Stopping the search.")
                    set_mode("IDLE")
                continue
            if tgt.get("pos") is not None:
                hd, _, ps = pose(meta)
                want = wrap(heading_to(ps, tgt["pos"])[0] - hd)
                seats.sort(key=lambda d: abs(px_bearing((d["box"][0] + d["box"][2]) / 2, meta) - want))
            new = dict(lock_target("empty seat", seats[0]["box"], meta, depth, True), kind="seat")
            with S.lock:
                still = S.mode == mode and S.target is tgt
            if still and mode == "SEARCH":
                heading, _, pos = pose(meta)
                set_mode("GOTO", new)
                say(f"Found an empty {seats[0]['name']}, {direction_words(wrap(heading_to(pos, new['pos'])[0] - heading))}. Follow the belt.")
            elif still:
                with S.lock:
                    tgt.update(smooth_pos(tgt, new))
            continue
        # Gemini only while searching or reaching; in GOTO the map already knows where the target is
        gemini_ok = (mode != "GOTO" or tgt.get("pos") is None) and time.time() - LAST["gemini"] > REFRESH_GEMINI_S
        items = BRAIN.find(frame, tgt["label"], gemini_ok=gemini_ok, background=True)
        local = bool(items) and "meta" in items[0]
        if local:
            cand = lock_target(tgt["label"], items[0]["box"], items[0]["meta"], depth, tgt.get("walk_only", False))
            ok = CONF.confirmed(tgt["label"], cand["pos"], time.time())
            near = tgt.get("pos") is not None and np.hypot(*(cand["pos"] - tgt["pos"])) < 0.8
            if not (ok or (mode != "SEARCH" and near)):
                items = []                              # a new spot needs 3-of-5 agreement; the known spot doesn't
        else:
            CONF.confirmed(tgt["label"], None, time.time())
        if not items:
            if mode == "SEARCH" and time.time() - tgt.get("started", time.time()) > 25:
                say(f"I still can't find your {tgt['label']}. Stopping the search.")
                set_mode("IDLE")
            continue
        meta = items[0].get("meta") or meta
        new = lock_target(tgt["label"], items[0]["box"], meta, depth, tgt.get("walk_only", False))
        with S.lock:
            still = S.mode == mode and S.target is tgt
        if not still:
            continue
        if not tgt.get("walk_only") and tracking_ok(meta):                 # keep memory current while guiding
            MEM.add(tgt["label"], new["pos"], new.get("height"), items[0].get("conf", 0.5), cam_pos_xz=pose(meta)[2])
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
                tgt.update(smooth_pos(tgt, new))


def memory_loop():
    """logs where notable objects are, so 'where did I leave...' works later. Reads the detector thread's
    newest result twice a second (no extra GPU work); an object is written only after 3 of 5 looks agree.
    Gemini adds a smarter scan now and then when the scene changed."""
    last_gemini, last_thumb, last_t, last_neg = 0.0, None, 0.0, 0.0
    while True:
        time.sleep(0.5 if DET is not None else LOCAL_MEMORY_EVERY_S)
        if BRAIN is None:
            continue
        items = []
        if DET is not None:
            dets, meta, _, t, frame = DET.latest(with_frame=True)
            if frame is None or t <= last_t:
                continue
            last_t, depth = t, S.depth
            items = list({d["label"]: d for d in finder.memory_items(dets)}.values())   # Memory confirms by place
        else:
            frame, meta, depth = latest()
            if frame is None:
                continue
            if LOCAL is not None and LOCAL.model is not None:
                items += LOCAL.detect(frame, MEMORY_CLASSES, conf=0.25)
        if BRAIN.client is not None and MEMORY_EVERY_S > 0 and time.time() - last_gemini > MEMORY_EVERY_S:
            thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 24), interpolation=cv2.INTER_AREA)
            if last_thumb is None or np.abs(thumb.astype(np.int16) - last_thumb).mean() > 12:
                last_gemini, last_thumb = time.time(), thumb.astype(np.int16)   # scene changed: worth a scan
                items += BRAIN.scan(frame)
        if not tracking_ok(meta):
            continue                                   # positions are only trustworthy while ARKit tracks
        heading, _, pos = pose(meta)
        now, seen = time.time(), set()
        for it in items:
            if not it["label"]:
                continue
            t = lock_target(it["label"], it["box"], meta, depth)
            x0, y0, x1, y1 = [int(v) for v in it["box"]]
            crop = frame[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            thumb = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes() if crop.size else None
            if MEM.add(it["label"], t["pos"], t.get("height"), it.get("conf", 1.0), cam_pos_xz=pos, t=now, thumb=thumb):
                seen.add(it["label"])
        if now - last_neg > 2.0:                       # "should be in view but isn't": 3 misses -> stale
            last_neg, half_fov = now, abs(px_bearing(0, meta))
            for pin in MEM.pins(now):
                if pin["label"] not in seen:
                    MEM.negative(pin["label"], pos, heading, half_fov, now)


# ----------------------------------------------------------------- web: phone socket + dashboard
app = FastAPI()


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    PHONE["ws"], PHONE["loop"] = ws, asyncio.get_running_loop()
    print("Phone connected")
    meta, expect, frame_k, bad_text, pushed = None, None, 0, False, False
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except ValueError:
                    if not bad_text:
                        print("ignoring non-JSON text from phone:", msg["text"][:80])
                    bad_text = True
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "frame":
                    meta, expect = data, "jpeg"
                    want = SETTINGS.get()["camera"]
                    if not pushed and want != "auto" and data.get("camera") not in (None, want):
                        send_phone({"type": "camera_mode", "zoom": want})   # e.g. 0.75x, the best-tested lens
                        log_event("info", f"Asked the phone for the {want} lens")
                    pushed = True
                elif data.get("type") == "cmd":
                    threading.Thread(target=handle_command, args=(data.get("text", ""),), daemon=True).start()
                elif data.get("type") == "planes":
                    with S.lock:
                        S.planes = [pl for pl in data.get("planes", []) if pl.get("T") and pl.get("poly")]
            elif msg.get("bytes") is not None and meta is not None:
                b = msg["bytes"]
                if expect == "jpeg":
                    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    m = json.loads(json.dumps(meta))
                    frame_k = 0
                    if "rot" not in m and m.get("T"):    # phone didn't rotate: make the image upright here
                        force = ORIENT["force"]
                        if force == "auto":
                            frame_k = geometry.pick_orientation(m["T"], ORIENT["k"])
                        else:
                            frame_k = 1 if force == "portrait" else 0
                        ORIENT["k"] = frame_k
                        img, m = geometry.apply_rotation(img, m, frame_k)
                    elif FLIP_IMAGE and "rot" not in m:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                        for hd in m.get("hands", []):
                            for k in ("wrist", "thumb", "index", "middle", "ring", "little"):
                                if hd.get(k):
                                    hd[k] = [img.shape[1] - hd[k][0], img.shape[0] - hd[k][1]]
                    set_frame_size(img.shape[1], img.shape[0])
                    m["W"], m["H"] = img.shape[1], img.shape[0]
                    with S.lock:
                        S.frame, S.meta, S.frame_id = img, m, S.frame_id + 1
                    expect = "depth" if meta.get("depth") else None
                elif expect == "depth":
                    d = np.frombuffer(b, np.float16).astype(np.float32).reshape(meta.get("dh", 96), meta.get("dw", 128))
                    if frame_k:
                        d = cv2.rotate(cv2.resize(d, (H, W) if frame_k in (1, 3) else (W, H)), ROTCV[frame_k])
                    d = cv2.resize(d, (W, H))
                    if FLIP_IMAGE and "rot" not in meta and not frame_k:
                        d = cv2.rotate(d, cv2.ROTATE_180)
                    with S.lock:
                        S.phone_depth, S.phone_depth_t = d, time.time()
                    expect = None
    except WebSocketDisconnect:
        pass
    finally:
        PHONE["ws"] = None
        print("Phone disconnected")


def set_frame_size(w, h):
    """the processed frame is 640x480 landscape or 480x640 portrait; everything reads W, H from here."""
    global W, H
    if (w, h) != (W, H):
        W, H = w, h
        print(f"Frame size now {w}x{h}")


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


@app.get("/video/map")
async def video_map():
    return mjpeg(lambda: S.map_vis)


@app.get("/state")
async def state():
    with S.lock:
        tgt = S.target or {}
        return JSONResponse({
            "mode": S.mode, "fps": round(S.fps, 1), "motors": {str(k): v for k, v in S.motors.items()},
            "heard": S.last_heard, "said": S.last_said, "target": tgt.get("label"), "finder": S.last_finder,
            "depth": S.depth_source, "phone": PHONE["ws"] is not None,
            "planner": ("route" if S.path else "dodge only") if USE_PLANNER else "off",
            "timing": dict(S.timing, fps=round(S.fps, 1)), "depth_source": S.depth_source,
            "depth_scale": round(S.depth_scale, 3), "reach": dict(REACH_DBG),
            "teach_ready": TEACH_STATE["ready"], "version": VERSION, "depth_scale_source": DEPTH_SCALE.source or "none",
            "pending": bool(S.pending),
            "memory": [dict(p, thumb=p["label"] in MEM.thumbs) for p in MEM.pins(time.time())],
            "taught": taught_list()})


def taught_list():
    try:
        return [{"label": lab, "n": len(TEACH.items[lab]["emb"])} for lab in TEACH.labels()]
    except Exception:
        return [{"label": lab, "n": 1} for lab in TEACH.labels()]


@app.get("/taught/{label}.jpg")
async def taught_thumb(label: str):
    path = os.path.join(TEACH.dir, finder._slug(TEACH.key(label)) + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return Response(content=open(path, "rb").read(), media_type="image/jpeg")


@app.get("/thumb/{label}.jpg")
async def thumb(label: str):
    jpg = MEM.thumbs.get(label)
    if jpg is None:
        raise HTTPException(status_code=404)
    return Response(content=jpg, media_type="image/jpeg")


@app.websocket("/ws/events")
async def events(ws: WebSocket):
    await ws.accept()
    with EVENTS_LOCK:
        hist = list(EVENTS)
    last = hist[-1][0] if hist else 0
    try:
        await ws.send_text(json.dumps([{"i": i, "t": t, "who": w, "text": x} for i, t, w, x in hist]))
        while True:
            await asyncio.sleep(0.2)
            with EVENTS_LOCK:
                new = [e for e in EVENTS if e[0] > last]
            for i, t, w, x in new:
                await ws.send_text(json.dumps({"i": i, "t": t, "who": w, "text": x}))
                last = i
    except (WebSocketDisconnect, RuntimeError):
        pass


def apply_setting(key, value):
    """called by Settings when the dashboard changes something."""
    global USE_PLANNER
    if key == "planner":
        USE_PLANNER = bool(value)
    elif key == "camera":
        send_phone({"type": "camera_mode", "zoom": value})   # the phone restarts ARKit: world coords reset
        MEM.clear()
        DEPTH_SCALE.reset()
        with S.lock:
            S.planes = []
        if GRID is not None:
            GRID.reset()
        set_mode("IDLE")
        log_event("info", f"Camera set to {value}: tracking, map and memory reset")
    elif key == "orient":
        ORIENT["force"] = value
    elif key in FIELD:
        FIELD[key] = value


SETTINGS = settings_mod.Settings(on_change=apply_setting)


@app.get("/settings")
async def get_settings():
    return JSONResponse(SETTINGS.get())


@app.post("/settings")
async def post_settings(payload: dict = Body(...)):
    results = [await asyncio.to_thread(SETTINGS.set, k, v) for k, v in payload.items()]  # nvpmodel can block 20 s
    ok = all(r[0] for r in results) if results else False
    return {"ok": ok, "message": "; ".join(r[1] for r in results)}


@app.post("/cmd")
async def cmd(payload: dict = Body(...)):
    threading.Thread(target=handle_command, args=(payload.get("text", ""),), daemon=True).start()
    return {"ok": True}


def request_speech(text):
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
           "?output_format=mp3_44100_128")
    body = json.dumps({"text": text, "model_id": "eleven_flash_v2_5"}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    })
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


@app.post("/tts")
async def text_to_speech(payload: dict = Body(...)):
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        raise HTTPException(status_code=400, detail="text must contain 1 to 500 characters")
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise HTTPException(status_code=503, detail="ElevenLabs is not configured")
    try:
        audio = await asyncio.to_thread(request_speech, text.strip())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        print("ElevenLabs error:", error)
        raise HTTPException(status_code=502, detail="ElevenLabs request failed") from error
    return Response(content=audio, media_type="audio/mpeg")


DASHBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


@app.get("/")
async def index():
    if os.path.exists(DASHBOARD_FILE):                   # judge dashboard; the built-in page is the fallback
        return HTMLResponse(open(DASHBOARD_FILE, encoding="utf-8").read())
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
<button id="b-map" onclick="view('map')">Map</button>
<span id="phone"></span></header>
<main><div class="views"><img id="cam" src="/video/camera"><img id="depth" src="/video/depth" style="display:none"><img id="map" src="/video/map" style="display:none;max-width:480px"></div>
<aside><svg id="belt" viewBox="-110 -110 220 220" width="300" height="300"></svg>
<div id="status"></div>
<p><input id="cmd" placeholder="Type a command, e.g. find my keys" onkeydown="if(event.key==='Enter')send()"></p>
<div id="memory"></div></aside></main>
<script>
const ANG={1:0,2:45,3:90,4:120,5:-120,6:-90,7:-45};
function view(v){const show={cam:['cam'],depth:['depth'],split:['cam','depth'],map:['cam','map']}[v];
for(const k of ['cam','depth','map'])document.getElementById(k).style.display=show.includes(k)?'':'none';
for(const k of ['cam','depth','split','map'])document.getElementById('b-'+k).className=k===v?'on':'';}
function send(){const i=document.getElementById('cmd');fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:i.value})});i.value='';}
function belt(m){let s='<circle r="80" fill="none" stroke="#444" stroke-width="6"/><text y="-92" text-anchor="middle" fill="#888" font-size="14">front</text>';
for(const k in ANG){const a=ANG[k]*Math.PI/180,x=80*Math.sin(a),y=-80*Math.cos(a),on=m[k];
s+=`<circle cx="${x}" cy="${y}" r="${on?16:11}" fill="${on?'#0a84ff':'#333'}"/><text x="${x}" y="${y+5}" text-anchor="middle" fill="#fff" font-size="13">${k}</text>`;}
const r=m['8'];s+=`<circle r="${r?24:16}" fill="${r?'#ff9f0a':'#333'}"/><text y="5" text-anchor="middle" fill="#fff" font-size="13">ring</text>`;
document.getElementById('belt').innerHTML=s;}
async function tick(){try{const s=await (await fetch('/state')).json();belt(s.motors);
document.getElementById('phone').textContent=s.phone?'phone connected':'phone NOT connected';
document.getElementById('status').innerHTML=`Mode: <b>${s.mode}</b>${s.target?' ('+s.target+')':''}<br>Heard: ${s.heard||'-'}<br>Said: ${s.said||'-'}<br>Found by: ${s.finder||'-'}<br>Path: ${s.planner}<br>${s.fps} fps, depth: ${s.depth}`;
document.getElementById('memory').innerHTML=s.memory.length?'<b>Last seen</b><br>'+s.memory.map(m=>`${m[0]} - ${m[1]} s ago`).join('<br>'):'';}catch(e){}}
setInterval(tick,200);belt({});
</script></body></html>"""


if __name__ == "__main__":
    BELT = Belt()
    try:                                                # fixed-vocab detector thread; old LocalFinder as fallback
        DET = finder.DetectorThread(finder.WorldDetector())
        if not LOW_MEM:
            STRONG_DET = finder.StrongDetector()          # loaded later by warmup(), after the core models
            DET.det.strong = STRONG_DET                   # unusual words share it (one L copy, not two)
        print("Detector thread ready:", len(finder.VOCAB), "words")
        mem_log("fast detector loaded")
    except Exception as e:
        print("Detector thread off, using the old finder:", e)
        LOCAL = LocalFinder()
    BRAIN = Brain()
    GRID = FloorMap()
    for fn in (perception_loop, control_loop, refresher_loop, memory_loop, planner_loop):
        threading.Thread(target=fn, daemon=True).start()
    threading.Thread(target=warmup, daemon=True).start()   # extras load one at a time, after the core is up
    print("Dashboard: http://orin.local:8765/   phone socket: ws://<this-ip>:8765/stream")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
