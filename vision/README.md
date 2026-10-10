# Vision (Track C · Orin)

The whole Orin program: the phone socket, YOLO11n with ByteTrack, Depth Anything V2, 16-sector steering, the motor commands and the dashboard (port 8765).

| Path | What |
|---|---|
| `orin/navbelt_server.py` | The server. Put `yolo11n.engine` next to it |
| `tests/fake_phone.py` | Stand-in phone (webcam or `walk.mp4`) |
| `tests/yolo_test.py` | Builds the TensorRT engine and checks fps |
| `tests/find_test.py` | Checks the YOLO-World find-anything detector (`desk.jpg keys wallet`, or `0 keys` for webcam; the live view needs a display) |
| `tests/depth_test.py` | Checks depth distance and speed |

Settings: `GEMINI_API_KEY`, `GEMINI_MODEL`, `FLIP_IMAGE=1`, `CAM_HEIGHT_M` (measure it!), `USE_DEPTH=0`, `RING_HAND=left`, `USE_PLANNER=0`, `USE_OPEN_VOCAB=0`, `OPEN_VOCAB_CONF` (default 0.15).

Path planning: depth + ARKit build a floor map; A* routes around obstacles in GOTO mode (dashboard **Map** view). The map stays blank with `fake_phone.py` (it sends `tracking:false`).

"Find my …" asks the local YOLO-World detector first; Gemini is the fallback.
