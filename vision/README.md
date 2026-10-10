# Vision (Track C · Orin)

The whole Orin program: the phone socket, YOLO11n with ByteTrack, Depth Anything V2, 16-sector steering, the motor commands and the dashboard (port 8765).

| Path | What |
|---|---|
| `orin/navbelt_server.py` | The server. Put `yolo11n.engine` next to it |
| `tests/fake_phone.py` | Stand-in phone (webcam or `walk.mp4`) |
| `tests/yolo_test.py` | Builds the TensorRT engine and checks fps |
| `tests/depth_test.py` | Checks depth distance and speed |

Settings: `GEMINI_API_KEY`, `GEMINI_MODEL`, `FLIP_IMAGE=1`, `CAM_HEIGHT_M`, `USE_DEPTH=0`, `RING_HAND=left`.
