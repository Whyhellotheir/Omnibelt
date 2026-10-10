# Omnibelt (Haptic Nav Belt) — KnightHacks 9

## Start here
- **What:** a belt that steers a blind person by vibration, plus a ring that guides their hand to lost objects.
  - "The belt tells you where to walk; the ring tells you where to reach."
  - Pitch line: *the AI decides where; the reflexes decide how.*
- **Current focus (Hieu): Hardware, Stage 1A.** The goal is for all 8 motors to buzz on command from a computer while they're still loose on the bench. See [Hardware now](#hardware-now-stage-1a).
- **Source docs:** "Haptic Nav Belt — Design Document" and "Haptic Nav Belt — Build Guide" (Oct 9 2026, Hugh Nguyen). These are kept outside the repo. This file summarizes them; when they disagree, the docs win.
- **Event:** KnightHacks, about 36 hours. Judging targets:
  - the **Jetson** competition: everything safety-critical runs on the Orin worn on the belt
  - **Best use of Gemini**: understand, find, remember, describe
  - ElevenLabs voice
  - The ring is the differentiator and the centerpiece of the demo.

## Architecture
```
iPhone 17 (eyes/ears) --WebSocket :8765/stream--> Jetson Orin Nano Super (reflexes) --USB serial 115200--> ESP32 + ULN2803 (muscles) --> 7 belt motors + ring (M8)
       ^  "say" text                                   |  background thread
       |                                               v
   ElevenLabs (voice)                           Gemini Flash (brain: intent / find / memory / describe)
```
- Gemini and ElevenLabs are the only internet calls, so obstacle avoidance keeps working offline.
- **Gemini never drives the motors.** It only sets the goal that the steering loop follows.
- Local models run every frame:
  - YOLO11n (TensorRT) with ByteTrack
  - Depth Anything V2 Small, metric indoor (every 1–2 frames; there is no LiDAR)
  - Apple Vision hand pose and ARKit world tracking, both on the phone
- **Finding runs on the Nano first** (Build Guide v2, 2026-10-10):
  - **YOLO-World** (`yolov8s-worldv2.pt`) is an open-vocabulary detector. Give it words and it finds them, on the Orin GPU.
  - It answers "find my …" first, re-finds the target every 0.4 s (`REFRESH_LOCAL_S`), and scans `MEMORY_CLASSES` every 3 s to feed memory.
  - **Gemini is now the backup.** It is called only when the local detector finds nothing (at most every 1.5 s), plus for describe and a smarter memory scan every 8 s. Thinking is turned off for speed.
  - Common commands (stop/walk/describe/"find my X"/"take me to X"/"follow me") are parsed instantly by keywords in `fast_intent`. Anything else goes to Gemini.
  - The dashboard shows "Found by: YOLO-World (conf)" or "Gemini".
- Steering:
  - Take the depth band from knee to head height and split it into 16 sectors. A sector is blocked if anything in it is under 1.0 m (`OBSTACLE_M`).
  - Pick the open sector closest to the goal and buzz the nearest motor, blending two motors for in-between angles.
  - Pulse faster as things get closer. If every sector is blocked, play `SWEEP`; when the goal is reached, play `ARRIVE`.
  - Obstacles are depth points between ankle and head height (`OBST_MIN_H` 0.15 m to `OBST_MAX_H` 1.9 m), computed with exact pinhole geometry from `CAM_HEIGHT_M` and the ARKit pitch. The floor never counts as a wall.
- **Path planning** (navbelt_code (2), 2026-10-10):
  - Depth plus the ARKit pose build a 12×12 m floor map with 15 cm cells, centred on the start point. Things are forgotten after about 15–30 s.
  - A* plans a route to the goal around everything seen, including obstacles no longer in view, keeping 0.35 m clear (`BODY_RADIUS_M`). It re-plans every 0.4 s, in GOTO mode only.
  - Steering follows the route by aiming at a point 0.8 m ahead, and the dodge layer still checks every frame.
  - No route (start or goal off the map, ARKit not tracking, goal unreachable) falls back to steering straight at the goal, with dodging.
  - `USE_PLANNER=0` turns it off, leaving dodging only.
  - The dashboard has a **Map** view (black = obstacle, orange = route, red = goal, green = you) and shows Path: route / dodge only / off. The map stays blank with `fake_phone.py`, because it sends `tracking:false`.

## Repo layout (one folder per track)
| Folder | Track / owner | Contents | Build-guide path |
|---|---|---|---|
| `hardware/` | A · Hardware (**Hieu**): wiring, firmware, prints, belt, ring | `firmware/navbelt_fw/navbelt_fw.ino`, `cad/parts.py`, `tests/belt_test.py` | `esp32/…`, `cad/…`, `tests/belt_test.py` |
| `software/` | B · iOS app: ARKit stream, hands, speech, ElevenLabs | `ios/NavBelt/*.swift`, `ios/NavBelt/Info-plist-keys.txt`, `tests/test_receiver.py` | `ios/…`, `tests/test_receiver.py` |
| `vision/` | C · Orin: YOLO, depth, steering, motors, dashboard | `orin/navbelt_server.py`, `tests/fake_phone.py`, `tests/yolo_test.py`, `tests/find_test.py` (YOLO-World), `tests/depth_test.py` | `orin/…`, `tests/…` |
| `ai/` | D · AI + pitch: Gemini prompts, intent, memory, spoken lines, slides | `tests/gemini_find_test.py`, `tests/elevenlabs_test.py` | `tests/…` |

- The build guide assumes the original flat zip layout. Use the right-hand column to translate its paths.
- **Orin server settings** (env vars):
  - `GEMINI_API_KEY`, `GEMINI_MODEL`
  - `FLIP_IMAGE=1`
  - `CAM_HEIGHT_M` (1.0 belt / 1.3 chest)
  - `USE_DEPTH=0` (also the fix for out-of-memory)
  - `RING_HAND=left`
  - `USE_PLANNER=0`: dodging only, no map or route.
  - **Measure `CAM_HEIGHT_M` on the wearer to within about 10 cm.** If the Map shows phantom obstacles on open floor, the camera height is set too high or depth reads short. Raise `OBST_MIN_H` from 0.15 to 0.2 in `navbelt_server.py`.
  - `USE_OPEN_VOCAB=0` turns YOLO-World off. `OPEN_VOCAB_CONF` (code default 0.15) can be raised (e.g. 0.25) if it finds wrong things or the reach box jumps.
- **Orin setup additions in v2:**
  - `sudo apt install git tmux` and `pip install git+https://github.com/ultralytics/CLIP.git`
  - `python3 -c "import torch; print(torch.cuda.is_available())"` must print True. If it's False, every model runs on the CPU.
  - Run `yolo_test.py`, `find_test.py` and `depth_test.py` once on good Wi-Fi so the models are cached.
  - Use MAXN SUPER mode on wall power and 15W on battery.
- AI's prompts (`INTENT_PROMPT`, `SCAN_PROMPT`), its `say(...)` lines, and `DASHBOARD_HTML` all live inside `vision/orin/navbelt_server.py`. Coordinate with Vision before editing them.

## Interfaces (shared contract: change only with whole-team agreement)
**Phone → Orin** (`ws://orin.local:8765/stream`). Each frame is sent as three messages. `tracking` was added 2026-10-10: true when ARKit tracking is normal, and the map only updates then. If it's missing, it's treated as true.
1. A JSON text message.
2. The 640×480 JPEG, as a binary message.
3. A float16 depth array, only if `depth` is true.

```json
{"type":"frame","seq":123,"t":1760000000.12,"yaw":1.57,"pitch":-0.21,"T":[16 floats, ARKit camera-to-world, column-major],"intr":["fx","fy","cx","cy"],"hands":[{"side":"right","wrist":[x,y],"index":[x,y],"conf":0.9}],"depth":false,"tracking":true}
{"type":"cmd","text":"find my keys"}
```

**Orin → phone** (same socket):
- `{"type":"say","text":"..."}`
- `{"type":"status","mode":"REACH"}`

**Orin → ESP32**: USB serial at 115200 baud, one command per line.
| Command | Effect |
|---|---|
| `M <1-8> <0-100> S` | Motor steady |
| `M <1-8> <0-100> P<hz>` | Motor pulsing (e.g. `M 3 70 P4`) |
| `M <n> 0 S` / `ALL 0` | Off / all off |
| `SWEEP` / `ARRIVE` | Stop pattern / arrived pattern |
| `TEST` | Buzz M1–M8 in order |
| `WD 0` / `WD 1` | Watchdog off (hand testing) / on |
| `PING` | Replies `PONG` |

Firmware safety limits:
- At most 3 motors on at once.
- Everything auto-stops if no command arrives for 500 ms. The server re-sends 10–15 times a second.
- Opening the serial port normally resets the ESP32, so wait 2 s before sending anything. The server does this. `belt_test.py` opens the port with DTR/RTS held low so the board doesn't reset; press **EN** when it asks you to.

**Dashboard:** `http://orin.local:8765/`, served by the Orin. The design doc also describes port 8080 with `/video/camera`, `/video/depth` and `/state`. The current code serves everything on 8765.

## Hardware now (Stage 1A)
- **Parts in hand:**
  - ELEGOO ESP32-WROOM-32 (×3)
  - ULN2803APG
  - BOJACK 10×3 mm coin motors
  - 400-hole breadboard, 22 AWG wire
  - INIU 45W 10k battery and 15V trigger cable (feeds the Orin's DC jack)
- **Still needed / to check:**
  - the short USB-C ↔ USB-A *data* cable from Orin to ESP32
  - whether the battery label shows 15V⎓3A
- **Pin map.** Go by the labels printed on the board, not by position. All of these pins are on one side of the ESP32.

| ESP32 | ULN in | ULN out | Motor |
|---|---|---|---|
| GPIO13 | 1 | 18 | M1 front (navel) |
| GPIO14 | 2 | 17 | M2 front-right |
| GPIO33 | 3 | 16 | M3 right hip |
| GPIO25 | 4 | 15 | M4 back-right |
| GPIO26 | 5 | 14 | M5 back-left |
| GPIO27 | 6 | 13 | M6 left hip |
| GPIO32 | 7 | 12 | M7 front-left |
| GPIO12 | 8 | 11 | M8 ring |

- **Power:**
  - ESP32 VIN/5V → + rail
  - ESP32 GND → − rail
  - ULN pin 9 → − rail
  - ULN pin 10 → + rail. Don't skip this one; it powers the chip's protection diodes.
  - Each motor: one lead to the + rail, the other to its ULN output. Polarity doesn't matter.
- **Steps:**
  1. In Arduino IDE 2, install "esp32 by Espressif" and pick the board **ESP32 Dev Module**. Upload. If it hangs at "Connecting…", hold BOOT.
  2. Open the Serial Monitor at 115200 with line ending "Newline". Run `PING`, then `WD 0`, then `M 1 80 S`, then `M 1 0 S`, then `TEST`.
  3. Add motors one at a time, running `TEST` after each one.
  4. Close the Serial Monitor (only one program can hold the port), then run `python hardware/tests/belt_test.py COM9` on Windows with the CP210x COM port from Device Manager. On Mac/Linux, leave out the port and it's found automatically. Needs `pip install pyserial`. Press EN on the ESP32 when prompted.
- **Done when:** `TEST` buzzes M1–M8 in order and `belt_test.py` runs clean.
- **Rules:**
  - Never solder or glue a motor until it has passed `TEST`.
  - GPIO14 making M2 twitch at power-on is normal.
- **Next stages:**
  - Stage 2: solder 40 cm leads, fit the pods, set `MOTOR_ANGLES` in the server.
  - Stage 4: the ring on M8, plus an optional LED with a 150 Ω resistor on ULN pin 11.
  - 3D prints are in `PRINT_THIS.zip` (15 parts, ~300 g PLA/PETG, no supports). Print one motor pod first as a fit test. To regenerate them: `cd hardware/cad && python parts.py` (needs `pip install manifold3d trimesh numpy`), which writes `stl/`.
  - Swap ring: `stl/ring_optional/ring_SWAP_plate.stl` holds all six sizes plus 3 motor capsules on one bed. The capsule slides into any ring from the fingertip side, and the wire's pull holds it in.

## Checkpoints
1. Wave a hand at the phone and the right motor buzzes.
2. Blindfolded hallway walk with no collisions: (a) on wall power, (b) on battery and Wi-Fi.
3. "Belt, find my keys" works three times in a row, wireless, with the ring.

After checkpoint 3: freeze, rehearse the 90 s demo five times, and record a backup video.

## Rules
- **No keys in the repo.** `GEMINI_API_KEY` and `ELEVENLABS_API_KEY` come from env vars. `software/ios/NavBelt/Config.swift` keeps the `PASTE_…` placeholders in git; paste real values only in your local copy, and never commit them.
- Only edit your own track's folder. Changes to the shared contract above need the whole team's agreement.
- Pull before you push. Small commits, staged by file name.

## Open issues
- **DUTY_CAP (resolved):** `hardware/firmware/navbelt_fw/navbelt_fw.ino:30` is `0.65`, the same as the latest `navbelt_code (1).zip` and the docs. If the pulses are too weak, raise it here on purpose.
- Does the team have a Mac? Without one, the phone side becomes a browser page served by the Orin.
- Sponsor LiDAR? Battery 15V label? Demo wearer's ring size?

## Worklog
- 2026-10-10: Repo seeded from `navbelt_code.zip` and split into hardware/software/vision/ai. CLAUDE.md written from the design doc and build guide.
- 2026-10-10: Merged Jayce's flat initial commit; files moved into track folders, firmware kept at DUTY_CAP 0.65.
- 2026-10-10: Build Guide v2: YOLO-World find-anything runs locally on the Nano with Gemini as the fallback (server + `vision/tests/find_test.py`). Took the newer `belt_test.py` (Windows COM port, no reset on open) and `parts.py` (universal belt phone holder) from `navbelt_code (1).zip`.
- 2026-10-10: navbelt_code (2): floor map + A* path planning on the Orin, `tracking` flag from the phone, swap ring CAD. Fixed `parts.py` crashing at export (an ordering bug since the universal holder; all 24 STLs now export watertight). Fixed the obstacle height/distance math (the chest mount missed a 0.35 m backpack; synthetic checks in commit 3fcbcae).
