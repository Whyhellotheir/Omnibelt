"""AI track. Finds an object in a photo with Gemini and draws its box; also tests intent parsing.
    pip install google-genai opencv-python
    export GEMINI_API_KEY=...        (Windows PowerShell: $env:GEMINI_API_KEY="...")
    python gemini_find_test.py desk.jpg "keys"
    python gemini_find_test.py --intent "uh where did I leave my wallet"
Saves found.jpg."""
import json, os, sys, time
import cv2
from google import genai
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")   # newest Flash listed in AI Studio
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def ask_json(parts):
    r = client.models.generate_content(model=MODEL, contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2))
    return json.loads(r.text)

if sys.argv[1] == "--intent":
    t0 = time.time()
    print(ask_json([
        "You control a navigation belt for a blind user. Convert the request into JSON "
        '{"intent": "find|where|follow|goto|describe|stop|walk", "target": "short noun phrase or empty"}. '
        f"Request: {sys.argv[2]}"]), f"({time.time() - t0:.1f} s)")
    sys.exit()

path, target = sys.argv[1], sys.argv[2]
img = cv2.imread(path)
h, w = img.shape[:2]
jpg = cv2.imencode(".jpg", cv2.resize(img, (640, int(640 * h / w))))[1].tobytes()
t0 = time.time()
items = ask_json([types.Part.from_bytes(data=jpg, mime_type="image/jpeg"),
    f'Find "{target}" in this image. Return a JSON list of {{"label": string, "box_2d": [ymin, xmin, ymax, xmax]}} '
    "with coordinates normalized to 0-1000. Return [] if it is not visible."])
print(f"{time.time() - t0:.1f} s  ->", items)
for it in items:
    y0, x0, y1, x1 = it["box_2d"]
    p0, p1 = (int(x0 / 1000 * w), int(y0 / 1000 * h)), (int(x1 / 1000 * w), int(y1 / 1000 * h))
    cv2.rectangle(img, p0, p1, (0, 200, 255), 3)
    cv2.putText(img, it.get("label", target), (p0[0], max(20, p0[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
cv2.imwrite("found.jpg", img)
print("saved found.jpg")
