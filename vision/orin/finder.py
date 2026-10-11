"""Reliable object finding for the demo: one fixed YOLO-World vocabulary (set once, never switched), a
background detector thread, distractor-aware target picking, multi-frame confirmation, and "teach" mode
(CLIP embeddings of the user's own wallet/keys/bottle so a look-alike phone is not taken for the wallet).

GPU imports (ultralytics, torch, clip) are lazy, so the logic here runs and is tested without them.

TensorRT (optional, ~3x faster on the Orin). Run once on the Orin from vision/orin/:
    python3 finder.py export
which does: YOLOWorld(pt).set_classes(VOCAB); .save("yolow_vocab.pt");
            YOLO("yolow_vocab.pt").export(format="engine", half=True, imgsz=640)  -> yolow_vocab.engine
WorldDetector loads yolow_vocab.engine (next to this file) when it exists, else the .pt + set_classes.
Re-export whenever VOCAB changes (the class order is baked into the engine). FINDER_ENGINE=0 ignores it."""
import os, re, threading, time
from collections import deque
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(HERE, "yolow_vocab.engine")
VOCAB_PT = os.path.join(HERE, "yolow_vocab.pt")

DEMO = ["keys", "wallet", "water bottle"]
MEMORY = ["glasses", "cup", "backpack", "bag", "laptop", "door", "chair"]
# look-alikes: in the vocabulary only so they soak up confusions (a phone gets "phone", not "wallet")
DISTRACTORS = ["phone", "credit card", "remote control", "notebook", "book", "glasses case", "earbuds case",
               "computer mouse", "pen"]
VOCAB = DEMO + MEMORY + DISTRACTORS
MEMORY_LABELS = DEMO + MEMORY          # what the memory scan is allowed to write
SYNONYMS = {
    "key": "keys", "car keys": "keys", "car key": "keys", "house keys": "keys", "keychain": "keys",
    "key chain": "keys", "key ring": "keys", "keyring": "keys", "fob": "keys", "key fob": "keys",
    "purse": "wallet", "billfold": "wallet", "card holder": "wallet", "cardholder": "wallet",
    "money clip": "wallet",
    "bottle": "water bottle", "water": "water bottle", "flask": "water bottle", "thermos": "water bottle",
    "hydro flask": "water bottle", "hydroflask": "water bottle", "drink": "water bottle",
    "tumbler": "water bottle",
    "spectacles": "glasses", "sunglasses": "glasses", "eyeglasses": "glasses", "specs": "glasses",
    "mug": "cup", "coffee": "cup", "coffee cup": "cup", "coffee mug": "cup",
    "bookbag": "backpack", "book bag": "backpack", "rucksack": "backpack", "school bag": "backpack",
    "handbag": "bag", "tote": "bag", "tote bag": "bag", "duffel": "bag", "duffel bag": "bag",
    "computer": "laptop", "macbook": "laptop", "notebook computer": "laptop",
    "doorway": "door", "exit": "door", "seat": "chair",
    "cell phone": "phone", "cellphone": "phone", "mobile": "phone", "mobile phone": "phone",
    "iphone": "phone", "smartphone": "phone",
    "card": "credit card", "debit card": "credit card", "id card": "credit card",
    "remote": "remote control", "tv remote": "remote control", "clicker": "remote control",
    "airpods": "earbuds case", "earbuds": "earbuds case", "airpods case": "earbuds case",
    "mouse": "computer mouse", "pencil": "pen", "marker": "pen", "journal": "notebook",
}
_FILLER = {"my", "the", "a", "an", "your", "our", "his", "her", "their", "this", "that", "some"}


def canonical(target):
    """spoken words -> vocabulary label, or None if the fixed vocabulary has no word for it."""
    if not target:
        return None
    t = re.sub(r"[^a-z ]", " ", str(target).lower())
    words = [w for w in t.split() if w not in _FILLER]
    for i in range(len(words)):                     # longest tail first: "my black car keys" -> "car keys"
        p = " ".join(words[i:])
        for q in (p, p[:-1] if p.endswith("s") else None, p + "s"):
            if q in VOCAB:
                return q
            if q in SYNONYMS:
                return SYNONYMS[q]
    return None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _dominated(d, dets, ok_labels, thr=0.5):
    """another-label box sits on top of d and is more confident (the wallet the detector thinks is a phone)."""
    return any(o is not d and o["label"] not in ok_labels and o["conf"] > d["conf"] and iou(o["box"], d["box"]) > thr
               for o in dets)


def suppress(dets, thr=0.5):
    """drops every box beaten by a more confident, differently-labelled box on the same spot."""
    return [d for d in dets if not _dominated(d, dets, {d["label"]}, thr)]


def memory_items(dets, conf=0.25, labels=None):
    """dets worth writing to the 'where did I leave it' memory (distractor labels are never written)."""
    labels = set(labels or MEMORY_LABELS)
    return [d for d in suppress([d for d in dets if d["conf"] >= conf * 0.5]) if d["label"] in labels
            and d["conf"] >= conf]


def _parse(r, names):
    out = []
    if r.boxes is None or not len(r.boxes):
        return out
    rn = getattr(r, "names", None) or {}
    for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        lab = rn.get(c) if isinstance(rn, dict) else None
        out.append({"label": lab or (names[c] if c < len(names) else names[0]), "box": [float(v) for v in b],
                    "conf": float(cf)})
    return sorted(out, key=lambda d: -d["conf"])


class WorldDetector:
    """YOLO-World with one fixed vocabulary (no set_classes churn). detect(frame) -> [{label, box, conf}]."""
    def __init__(self, model_path="yolov8s-worldv2.pt", conf=0.10, imgsz=640, model=None, engine=None):
        self.model_path, self.conf, self.imgsz = model_path, conf, imgsz
        self.vocab, self.lock, self.engine = list(VOCAB), threading.Lock(), False
        self.open_model, self.open_cls, self.open_lock = None, None, threading.Lock()
        self.model = model
        if model is not None:
            return
        eng = engine or ENGINE_PATH
        try:
            if os.path.exists(eng) and os.environ.get("FINDER_ENGINE", "1") == "1":
                from ultralytics import YOLO
                self.model, self.engine = YOLO(eng, task="detect"), True
            else:
                from ultralytics import YOLOWorld
                self.model = YOLOWorld(model_path)
                self.model.set_classes(self.vocab)       # once, for the lifetime of the server
            self._predict(self.model, np.zeros((480, 640, 3), np.uint8), self.conf)   # warm-up
            print(f"Finder ready: {'TensorRT ' + eng if self.engine else model_path}, {len(self.vocab)} classes")
        except Exception as e:
            print("Finder detector off:", e)
            self.model = None

    def _predict(self, model, frame, conf):
        kw = dict(conf=conf, verbose=False, imgsz=self.imgsz, agnostic_nms=False, max_det=60)
        return model.predict(frame, **kw)[0]

    def detect(self, frame, conf=None):
        if self.model is None or frame is None:
            return []
        with self.lock:
            r = self._predict(self.model, frame, self.conf if conf is None else conf)
        return _parse(r, self.vocab)

    def detect_open(self, frame, target, conf=None):
        """separate YOLO-World instance for words outside VOCAB (its own lock; never touches the fixed one)."""
        if frame is None or not target:
            return []
        with self.open_lock:
            try:
                if self.open_model is None:
                    from ultralytics import YOLOWorld
                    self.open_model = YOLOWorld(self.model_path)
                if self.open_cls != target:
                    self.open_model.set_classes([target])
                    self.open_cls = target
                r = self.open_model.predict(frame, conf=max(0.15, self.conf) if conf is None else conf,
                                            verbose=False, imgsz=self.imgsz)[0]
            except Exception as e:
                print("open-vocab detect failed:", e)
                return []
        return [dict(d, label=target) for d in _parse(r, [target])]


def export_engine(model_path="yolov8s-worldv2.pt", imgsz=640):
    from ultralytics import YOLO, YOLOWorld
    m = YOLOWorld(model_path)
    m.set_classes(list(VOCAB))
    m.save(VOCAB_PT)
    out = YOLO(VOCAB_PT).export(format="engine", half=True, imgsz=imgsz)
    if os.path.abspath(out) != ENGINE_PATH:
        os.replace(out, ENGINE_PATH)
    print("saved", ENGINE_PATH)


class DetectorThread:
    """runs detector.detect on the newest submitted frame, at most `hz` times a second. Older frames that
    arrive while it is busy are dropped. submit() and latest() never block on the GPU."""
    def __init__(self, detector, hz=8.0):
        self.det, self.period = detector, 1.0 / hz
        self.cv = threading.Condition()
        self.pending, self.frame, self.result = None, None, ([], None, None, 0.0)
        self.processed, self.dropped, self.running = 0, 0, True
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame, meta=None, frame_id=None):
        with self.cv:
            if self.pending is not None:
                self.dropped += 1
            self.pending = (frame, meta, frame_id)
            self.cv.notify_all()

    def latest(self, with_frame=False):
        """(dets, meta, frame_id, t) of the newest finished detection; with_frame adds the frame they came from."""
        with self.cv:
            return self.result + (self.frame,) if with_frame else self.result

    def wait_newer(self, after_t, timeout=1.0):
        """blocks (caller's choice) until a result newer than after_t exists; returns latest() or None."""
        end = time.time() + timeout
        with self.cv:
            while self.result[3] <= after_t:
                left = end - time.time()
                if left <= 0:
                    return None
                self.cv.wait(left)
            return self.result

    def stop(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()

    def _run(self):
        last = 0.0
        while True:
            with self.cv:
                while self.pending is None and self.running:
                    self.cv.wait()
                if not self.running:
                    return
            wait = self.period - (time.time() - last)
            if wait > 0:
                time.sleep(wait)                         # rate limit; a newer frame may replace pending meanwhile
            with self.cv:
                job, self.pending = self.pending, None
            last = time.time()
            try:
                dets = self.det.detect(job[0])
            except Exception as e:
                print("detector error:", e)
                dets = []
            with self.cv:
                self.result, self.frame = (dets, job[1], job[2], time.time()), job[0]
                self.processed += 1
                self.cv.notify_all()


def pick_target(dets, target, teach=None, frame=None, sim_min=None, overlap=0.5):
    """best detection of `target` or None. A candidate under a more confident box of another label is dropped
    (wallet 0.2 under phone 0.4 = it's a phone). If the user taught this item, candidates also include the
    detector classes the taught crops were seen as, are re-ranked by 0.5*conf + 0.5*CLIP similarity, and any
    below sim_min (default teach.sim_min) are dropped."""
    lab = canonical(target)
    taught = teach is not None and frame is not None and teach.has(target)
    want = ({lab} if lab else set()) | (teach.classes(target) if taught else set())
    if not want:
        return None                                      # not in the vocabulary and not taught: use detect_open
    cands = [dict(d) for d in dets if d["label"] in want and not _dominated(d, dets, want, overlap)]
    if not cands:
        return None
    if taught:
        sims = teach.similarity_boxes(target, frame, [d["box"] for d in cands])
        thr = teach.sim_min if sim_min is None else sim_min
        for d, s in zip(cands, sims):
            d["sim"], d["score"] = s, 0.5 * d["conf"] + 0.5 * s
        cands = [d for d in cands if d["sim"] >= thr]
    else:
        for d in cands:
            d["score"] = d["conf"]
    return max(cands, key=lambda d: d["score"]) if cands else None


class Confirmer:
    """an object counts only after it shows up in roughly the same place in `need` of the last `n` looks.
    Call confirmed(label, box_or_None, now) once per new detection result (None = not seen this time)."""
    def __init__(self, n=5, need=3, iou=0.3, max_age=2.0):
        self.n, self.need, self.iou, self.max_age = n, need, iou, max_age
        self.hist = {}

    def confirmed(self, label, box, now=None):
        now = time.time() if now is None else now
        h = self.hist.setdefault(label, deque(maxlen=self.n))
        h.append((now, box))
        if box is None:
            return False
        return sum(1 for t, b in h if b is not None and now - t <= self.max_age and iou(b, box) > self.iou) >= self.need

    def reset(self, label=None):
        if label is None:
            self.hist.clear()
        else:
            self.hist.pop(label, None)


# ----------------------------------------------------------------- teach mode (CLIP image embeddings)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def square_pad(img):
    h, w = img.shape[:2]
    s = max(h, w)
    t, l = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, t, s - h - t, l, s - w - l, cv2.BORDER_REPLICATE)


def clip_preprocess(crop, size=224):
    """BGR crop -> 3x224x224 float32, padded to square (replicated edges), CLIP mean/std."""
    sq = square_pad(crop)
    sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)


def crop_box(frame, box, margin=0.05):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    x1, y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
    return frame[y0:y1, x0:x1] if x1 - x0 >= 4 and y1 - y0 >= 4 else None


class ClipEmbedder:
    """CLIP ViT-B/32 image encoder (the weights YOLO-World already cached in ~/.cache/clip), fp16 on CUDA."""
    def __init__(self, device=None):
        import torch, clip
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        if self.device != "cpu":
            self.model.half()
        self.lock = threading.Lock()

    def __call__(self, crops):
        x = np.stack([clip_preprocess(c) for c in crops])
        with self.lock, self.torch.no_grad():
            t = self.torch.from_numpy(x).to(self.device, dtype=self.model.visual.conv1.weight.dtype)
            return self.model.encode_image(t).float().cpu().numpy()


def _norm(e):
    e = np.asarray(e, np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)


def _slug(label):
    return re.sub(r"[^a-z0-9_-]+", "_", label.lower()).strip("_") or "item"


class TeachStore:
    """'this is my wallet': keeps up to `max_per` CLIP embeddings per label in <dir>/<label>.npz plus a
    <dir>/<label>.jpg thumbnail for the dashboard. embed(crops)->(k,d) is injectable (default ClipEmbedder,
    loaded on first use)."""
    def __init__(self, dir, embed=None, max_per=6, sim_min=0.75):
        self.dir, self._embed, self.max_per, self.sim_min = dir, embed, max_per, sim_min
        self.lock, self.items = threading.RLock(), {}    # label -> {"emb": (k,d), "classes": [str]}
        os.makedirs(dir, exist_ok=True)
        for f in sorted(os.listdir(dir)):
            if f.endswith(".npz"):
                try:
                    z = np.load(os.path.join(dir, f), allow_pickle=False)
                    self.items[str(z["label"])] = {"emb": _norm(z["emb"]), "classes": [str(c) for c in z["classes"]]}
                except Exception as e:
                    print("teach: could not load", f, e)

    @staticmethod
    def key(label):
        return canonical(label) or " ".join(re.sub(r"[^a-z0-9 ]", " ", str(label).lower()).split())

    def embed(self, crops):
        if self._embed is None:
            self._embed = ClipEmbedder()
        return _norm(self._embed(crops))

    def labels(self):
        with self.lock:
            return sorted(self.items)

    def has(self, label):
        with self.lock:
            return self.key(label) in self.items

    def classes(self, label):
        with self.lock:
            it = self.items.get(self.key(label))
            return set(it["classes"]) if it else set()

    def teach(self, label, frame, box, det_class=None):
        """adds the crop (and its mirror image) as references; returns how many are stored for the label."""
        k = self.key(label)
        crop = crop_box(frame, box)
        if not k or crop is None:
            return 0
        new = self.embed([crop, cv2.flip(crop, 1)])
        with self.lock:
            it = self.items.setdefault(k, {"emb": np.zeros((0, new.shape[1]), np.float32), "classes": []})
            it["emb"] = np.concatenate([it["emb"], new])[-self.max_per:]
            if det_class and det_class not in it["classes"]:
                it["classes"].append(det_class)
            path = os.path.join(self.dir, _slug(k))
            with open(path + ".tmp", "wb") as f:
                np.savez(f, label=np.array(k), emb=it["emb"], classes=np.array(it["classes"], dtype=str))
            os.replace(path + ".tmp", path + ".npz")
            cv2.imwrite(path + ".jpg", cv2.resize(square_pad(crop), (160, 160), interpolation=cv2.INTER_AREA))
            return len(it["emb"])

    def similarity(self, label, crops):
        """max cosine similarity of each crop to the label's references (None if the label isn't taught)."""
        with self.lock:
            it = self.items.get(self.key(label))
            refs = None if it is None else it["emb"]
        if refs is None or not len(refs):
            return None
        if not crops:
            return []
        return (self.embed(crops) @ refs.T).max(axis=1).tolist()

    def similarity_boxes(self, label, frame, boxes):
        crops = [crop_box(frame, b) for b in boxes]
        ok = [i for i, c in enumerate(crops) if c is not None]
        sims = self.similarity(label, [crops[i] for i in ok]) or []
        out = [0.0] * len(boxes)
        for i, s in zip(ok, sims):
            out[i] = float(s)
        return out

    def forget(self, label):
        k = self.key(label)
        with self.lock:
            gone = self.items.pop(k, None) is not None
            for ext in (".npz", ".jpg"):
                p = os.path.join(self.dir, _slug(k) + ext)
                if os.path.exists(p):
                    os.remove(p)
        return gone


def pick_teach_box(dets, hands, frame_wh):
    """the detection the user is holding up: the one overlapping a hand box (grown 10%) the most, else the
    largest box nearest the image centre. Background-sized boxes (>60% of the frame) never count."""
    W, H = frame_wh
    dets = [d for d in dets if (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) < 0.6 * W * H]
    if not dets:
        return None
    best, best_s = None, 0.0
    for h in hands or []:
        x, y, w, hh = h["bbox"]
        hb = [x - 0.1 * w, y - 0.1 * hh, x + 1.1 * w, y + 1.1 * hh]
        for d in dets:
            b = d["box"]
            inter = max(0.0, min(b[2], hb[2]) - max(b[0], hb[0])) * max(0.0, min(b[3], hb[3]) - max(b[1], hb[1]))
            s = inter / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))      # share of the object inside the hand area
            if s > best_s:
                best, best_s = d, s
    if best is not None:
        return best
    diag = (W * W + H * H) ** 0.5

    def central_size(d):
        b = d["box"]
        c = ((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2)
        return (b[2] - b[0]) * (b[3] - b[1]) / (W * H) * (1 - 2 * (c[0] ** 2 + c[1] ** 2) ** 0.5 / diag)
    return max(dets, key=central_size)


def default_teach_box(hands, frame_wh):
    """no detection at all: the grown hand box, else the middle third of the image."""
    W, H = frame_wh
    if hands:
        x, y, w, h = max(hands, key=lambda h: h["bbox"][2] * h["bbox"][3])["bbox"]
        return [max(0, x - 0.25 * w), max(0, y - 0.25 * h), min(W, x + 1.25 * w), min(H, y + 1.25 * h)]
    return [W / 3, H / 3, 2 * W / 3, 2 * H / 3]


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["export"]:
        export_engine(sys.argv[2] if len(sys.argv) > 2 else "yolov8s-worldv2.pt")
    else:
        print("usage: python3 finder.py export [yolov8s-worldv2.pt]")
