"""AI track. Hear what the belt would say: a photo + a spoken command go through the real server logic
(vision/orin/navbelt_server.py: intent -> find / describe / ...), and every say(...) line is spoken
with ElevenLabs. No phone, Orin or belt needed.
    pip install google-genai opencv-python requests fastapi uvicorn
    set GEMINI_API_KEY, ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID (or pass --voice)
    python voice_test.py desk.jpg "find my mouse"
    python voice_test.py desk.jpg "what's in front of me"
    python voice_test.py desk.jpg "take me to the door" --voice 21m00Tcm4TlvDq8ikWAM
    python voice_test.py desk.jpg "describe" --no-audio        # text only, no ElevenLabs
    python voice_test.py desk.jpg "find my keys" --local       # also use YOLO-World first, like the Orin
Saves voice_<time>_<n>.mp3 per line (and plays it on Windows)."""
import argparse, importlib.util, os, sys, time
import cv2
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "..", "..", "vision", "orin", "navbelt_server.py")

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("command", help='what the user says, e.g. "find my keys"')
ap.add_argument("--voice", default=os.environ.get("ELEVENLABS_VOICE_ID"), help="ElevenLabs voice ID")
ap.add_argument("--no-audio", action="store_true", help="print the lines only")
ap.add_argument("--local", action="store_true", help="load YOLO-World too (downloads it the first time)")
args = ap.parse_args()

img = cv2.imread(args.image)
if img is None:
    sys.exit(f"Can't read {args.image}")

spec = importlib.util.spec_from_file_location("navbelt_server", SERVER)
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)                       # loads the code only; the server itself doesn't start


class NoBelt:                                      # stands in for the ESP32
    def set(self, *a, **k): pass
    def send(self, *a, **k): pass
    def pattern(self, *a, **k): pass


lines = []
srv.say = lambda text: (lines.append(text), print("BELT SAYS:", text))
srv.BELT = NoBelt()
srv.BRAIN = srv.Brain()
srv.LOCAL = srv.LocalFinder() if args.local else None
# the photo stands in for one phone frame: camera level, at belt height, ARKit tracking
srv.S.frame = cv2.resize(img, (srv.W, srv.H), interpolation=cv2.INTER_AREA)
srv.S.meta = {"type": "frame", "yaw": 0.0, "pitch": 0.0, "intr": [500.0, 500.0, srv.W / 2, srv.H / 2],
              "T": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, srv.CAM_HEIGHT_M, 0, 1], "tracking": True}

t0 = time.time()
srv.handle_command(args.command)
print(f"({time.time() - t0:.1f} s, mode now {srv.S.mode}, found by: {srv.S.last_finder or '-'})")
if not lines:
    print("The belt said nothing.")

key = os.environ.get("ELEVENLABS_API_KEY")
if args.no_audio or not lines:
    sys.exit()
if not key or not args.voice:
    sys.exit("No audio: set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID (or pass --voice), or use --no-audio.")
for n, text in enumerate(lines, 1):
    t0 = time.time()
    r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{args.voice}?output_format=mp3_44100_128",
                      headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                      json={"text": text, "model_id": "eleven_flash_v2_5"}, timeout=30)
    if not r.ok:
        sys.exit(f"ElevenLabs error {r.status_code}: {r.text[:200]}")
    out = f"voice_{int(time.time())}_{n}.mp3"
    open(out, "wb").write(r.content)
    print(f"spoken in {time.time() - t0:.2f} s -> {out}")
    if hasattr(os, "startfile"):                   # Windows: play it in the default player
        os.startfile(os.path.abspath(out))
