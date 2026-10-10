"""AI track. Finds an object in a photo with Gemini and draws its box; also tests intent parsing.
    pip install google-genai opencv-python
    export GEMINI_API_KEY=...        (Windows PowerShell: $env:GEMINI_API_KEY="...")
    python gemini_find_test.py desk.jpg "keys"
    python gemini_find_test.py --intent "uh where did I leave my wallet"
    python gemini_find_test.py --models          # list the models your key can use (for GEMINI_MODEL / FALLBACK)
    python gemini_find_test.py --scan desk.jpg   # the server's memory scan (its exact SCAN_PROMPT) -> scan.jpg
Saves found.jpg.

Speed settings (env vars):
    GEMINI_MODEL       main model for finds (default gemini-3.8-flash)
    GEMINI_FALLBACK    comma-separated models to try when the main one is busy or out of quota
                       (default gemini-3.7-flash,gemini-3.6-flash,gemini-3-flash; "" to disable)
    GEMINI_LIGHT       model for --scan and --intent (default gemini-3.5-flash-lite), like the server
    GEMINI_TIMEOUT_S   give up on one model after this many seconds (default 12)
Thinking is turned off (or set to its minimum), which is what made a find take ~1 minute."""
import json, os, re, sys, time
import cv2
from google import genai
from google.genai import errors, types

# same two tiers as the server: Flash models (~20 requests/day each) for finds, Flash-Lite (~500/day) for
# --scan and --intent, which the server runs in the background / for parsing commands
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACKS = [m.strip() for m in os.environ.get("GEMINI_FALLBACK",
                                               "gemini-3.7-flash,gemini-3.6-flash,gemini-3-flash").split(",") if m.strip()]
LIGHT = [m.strip() for m in os.environ.get("GEMINI_LIGHT", "gemini-3.5-flash-lite").split(",") if m.strip()]
TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "12"))
# No SDK retries: on "high usage" (503/429) we move straight to the next model instead of waiting.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000),
                                                     retry_options=types.HttpRetryOptions(attempts=1)))

# Request settings to try, fastest first. Models differ in what they accept (older Flash takes a thinking
# budget of 0, Gemini 3 takes a level, some reject a response schema), and a model that doesn't accept a
# setting answers with a bare "400 INVALID_ARGUMENT", so on a 400 we step down to the next one.
OPTIONS = [(types.ThinkingConfig(thinking_budget=0), True), (types.ThinkingConfig(thinking_level="minimal"), True),
           (types.ThinkingConfig(thinking_level="low"), True), (None, True), (None, False)]
option_choice = {}                                           # model -> index into OPTIONS that it accepts


def busy(e):
    """True for errors where another model may still answer: overloaded, rate-limited, or timed out."""
    code = getattr(e, "code", None)
    return code in (408, 429, 500, 503, 504) or "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()


quota_hits = []                                              # (model, quota name, retry delay) from 429s


def quota_info(e):
    """From a 429: which quota ran out (e.g. ...PerDay...-FreeTier) and how long Google says to wait."""
    quota, delay = "", ""
    for d in ((e.details or {}).get("error", {}) or {}).get("details", []) if isinstance(e.details, dict) else []:
        for v in d.get("violations", []):
            quota = v.get("quotaId") or v.get("quotaMetric") or quota
            if v.get("quotaValue"):
                quota += f" (limit {v['quotaValue']})"
        delay = d.get("retryDelay", delay)
    return quota, delay


def try_model(model, parts, schema, max_tokens):
    """Returns parsed JSON, or None if this model can't answer right now."""
    i = option_choice.get(model, 0)
    while i < len(OPTIONS):
        think, use_schema = OPTIONS[i]
        cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                          response_schema=schema if use_schema else None,
                                          temperature=0.2, max_output_tokens=max_tokens, thinking_config=think)
        t0 = time.time()
        try:
            r = client.models.generate_content(model=model, contents=parts, config=cfg)
        except errors.ClientError as e:
            if e.code == 400:                                # this model rejects one of the settings
                i += 1
                continue
            if e.code in (401, 403):                         # bad or missing key: no model will work
                raise SystemExit(f"Gemini refused the API key ({e.code}): {e.message}")
            if e.code == 429:                                # out of quota for this model
                quota, delay = quota_info(e)
                quota_hits.append((model, quota, delay))
                print(f"  {model}: out of quota - {quota or e.message}" + (f", retry in {delay}" if delay else ""))
                return None
            print(f"  {model}: {e.code} {e.status} after {time.time() - t0:.1f} s, trying next model")
            return None                                      # 404 unknown model, ...
        except Exception as e:
            if not busy(e):
                raise
            print(f"  {model}: {getattr(e, 'code', '') or type(e).__name__} (busy) after "
                  f"{time.time() - t0:.1f} s, trying next model")
            return None
        option_choice[model] = i
        text = (r.text or "").strip()
        if text.startswith("```"):                           # tolerate ```json ... ``` around the answer
            text = text.strip("`").removeprefix("json").strip()
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            finish = r.candidates[0].finish_reason if r.candidates else "?"
            print(f"  {model} replied in {time.time() - t0:.1f} s but the answer was unreadable "
                  f"(finish: {finish}): {text[:120]!r}, trying next model")
            return None
        print(f"  {model} answered in {time.time() - t0:.1f} s")
        return out
    print(f"  {model}: rejected every request setting (400), trying next model")
    return None


def ask_json(parts, schema=None, max_tokens=2048, models=None):   # roomy: thinking tokens count toward this
    models = models or [MODEL] + [m for m in FALLBACKS if m != MODEL]
    for round_ in range(2):                                  # "high usage" errors usually clear in a second or two
        if round_:
            if len(quota_hits) == len(models):               # every model is out of quota: 2 s won't fix that
                break
            print("  every model was busy; retrying once in 2 s")
            time.sleep(2)
        for model in models:
            out = try_model(model, parts, schema, max_tokens)
            if out is not None:
                return out
    if quota_hits:
        daily = any("PerDay" in q for _, q, _ in quota_hits)
        raise SystemExit(
            "Out of Gemini quota. " + (
                "A daily limit ran out: it resets at midnight Pacific time (3 a.m. Eastern). " if daily else
                "A per-minute limit ran out: wait a minute and try again. ") +
            ("These are free-tier limits; turning on billing for the key's project in AI Studio raises them a lot. "
             if any("FreeTier" in q for _, q, _ in quota_hits) else "") +
            "A key from another project also has its own quota.")
    raise SystemExit("Gemini did not answer. Try again in a minute, or set GEMINI_FALLBACK to a model "
                     "listed in AI Studio.")


INTENT_SCHEMA = {"type": "OBJECT", "properties": {
    "intent": {"type": "STRING", "enum": ["find", "where", "follow", "goto", "describe", "stop", "walk"]},
    "target": {"type": "STRING"}}, "required": ["intent", "target"]}
FIND_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "label": {"type": "STRING"},
    "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    "confidence": {"type": "NUMBER"}}, "required": ["label", "box_2d", "confidence"]}}
SCAN_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "label": {"type": "STRING"}, "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}}}, "required": ["label", "box_2d"]}}
MIN_CONF = float(os.environ.get("GEMINI_FIND_CONF", "0.6"))      # same setting as the server

if sys.argv[1] == "--models":                                 # which models this key can use
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []) and "gemini" in m.name:
            print(m.name.removeprefix("models/"))
    sys.exit()

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Can't read {path}")
    h, w = img.shape[:2]
    small = cv2.resize(img, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA) if w > 640 else img
    return img, types.Part.from_bytes(data=cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(),
                                      mime_type="image/jpeg")


def draw(img, items, default_label):
    h, w = img.shape[:2]
    for it in items:
        if len(it.get("box_2d", [])) != 4:
            continue
        y0, x0, y1, x1 = it["box_2d"]
        p0, p1 = (int(x0 / 1000 * w), int(y0 / 1000 * h)), (int(x1 / 1000 * w), int(y1 / 1000 * h))
        cv2.rectangle(img, p0, p1, (0, 200, 255), 3)
        cv2.putText(img, it.get("label", default_label), (p0[0], max(20, p0[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


if sys.argv[1] == "--scan":
    # read SCAN_PROMPT straight from the server, so this tests exactly what the belt sends
    import ast
    server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision", "orin", "navbelt_server.py")
    prompt = next(ast.literal_eval(n.value) for n in ast.parse(open(server, encoding="utf-8").read()).body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SCAN_PROMPT")
    img, part = load_image(sys.argv[2])
    t0 = time.time()
    items = ask_json([part, prompt], SCAN_SCHEMA, models=LIGHT)           # the server scans with the light tier
    print(f"{time.time() - t0:.1f} s, {len(items)} objects the belt would remember:")
    for it in items:
        print("  ", it.get("label"), it.get("box_2d"))
    draw(img, items, "?")
    cv2.imwrite("scan.jpg", img)
    print("saved scan.jpg")
    sys.exit()

if sys.argv[1] == "--intent":
    t0 = time.time()
    print(ask_json([
        "You control a navigation belt for a blind user. Convert the request into JSON "
        '{"intent": "find|where|follow|goto|describe|stop|walk", "target": "short noun phrase or empty"}. '
        f"Request: {sys.argv[2]}"], INTENT_SCHEMA, models=LIGHT + [MODEL] + FALLBACKS), f"({time.time() - t0:.1f} s)")
    sys.exit()

path, target = sys.argv[1], sys.argv[2]
# This script finds one named object. Accept "find my keys" / "where are my keys" (-> "keys"), but send questions
# and other commands to voice_test.py, which sorts them into find / describe / ask like the belt does.
t = re.sub(r"[^a-z0-9' ]", " ", target.lower()).strip()
t = re.sub(r"^(?:(?:can you|could you|please|help me|uh|um)\s+)*(?:find|locate|look for|where (?:is|are|did i (?:leave|put))|"
           r"where's|where'd)\s+", "", t)
t = re.sub(r"^(?:(?:my|the|a|an|your|this|that)\s+)+", "", t)
t = re.sub(r"(?:\s+(?:please|now|go|at|again))+$", "", t).strip()
if not t or re.match(r"(what|which|who|how|why|is|are|does|do|can|could|describe|tell|read|take me|go to|follow|stop|walk)\b", t):
    sys.exit(f'"{target}" is a question or command, not an object name. This script only finds objects, e.g. '
             f'"keys" or "find my keys". To hear how the belt answers it, run:\n'
             f'    python tests\\voice_test.py {path} "{target}" --no-audio')
if t != target:
    print(f'Looking for "{t}"')
target = t
img, part = load_image(path)
t0 = time.time()
items = ask_json([part,
    f'Find "{target}" in this image. A blind person will walk to whatever you return, so only return something '
    f'that really is a {target}, not just a similar shape, color or object. Return a JSON list of at most 3 '
    '{"label": what the object actually is, "box_2d": [ymin, xmin, ymax, xmax], "confidence": 0.0-1.0} '
    "with coordinates normalized to 0-1000. If it is not clearly visible, return [] - an empty list is the "
    "right answer when unsure."], FIND_SCHEMA)
print(f"{time.time() - t0:.1f} s  ->", items)
for it in [i for i in items if float(i.get("confidence", 1)) < MIN_CONF]:
    print(f"  ignored (confidence {it['confidence']} < {MIN_CONF}): {it.get('label')}")
items = [i for i in items if float(i.get("confidence", 1)) >= MIN_CONF]
if not items:
    print(f'Not found: no "{target}" in the image.')
draw(img, items, target)
cv2.imwrite("found.jpg", img)
print("saved found.jpg")
