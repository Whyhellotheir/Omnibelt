"""AI track. Turns text into speech with ElevenLabs and times it.
    pip install requests
    export ELEVENLABS_API_KEY=...
    python elevenlabs_test.py --voices                  # list voice IDs, pick one
    python elevenlabs_test.py VOICE_ID "Your keys are ahead, slightly left"
Saves desk.mp3 (open it to listen)."""
import os, sys, time
import requests

KEY = os.environ["ELEVENLABS_API_KEY"]
if sys.argv[1] == "--voices":
    r = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": KEY})
    for v in r.json().get("voices", []):
        print(v["voice_id"], v["name"])
    sys.exit()
voice, text = sys.argv[1], sys.argv[2]
t0 = time.time()
r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
                  headers={"xi-api-key": KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                  json={"text": text, "model_id": "eleven_flash_v2_5"})
r.raise_for_status()
open("say.mp3", "wb").write(r.content)
print(f"{time.time() - t0:.2f} s, {len(r.content)} bytes -> desk.mp3")
