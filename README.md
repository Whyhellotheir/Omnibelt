# Omnibelt

A haptic navigation belt and ring for blind and low-vision users, built at KnightHacks 9.
- **The belt** steers you around obstacles by vibration.
- **The ring** guides your hand to the object you asked for ("Belt, find my keys").

| Folder | Track | Runs on |
|---|---|---|
| [`hardware/`](hardware/) | ESP32 firmware, wiring, 3D prints, belt + ring | ESP32 (Arduino IDE) |
| [`software/`](software/) | iPhone app: camera/pose/hands stream, wake word, voice | iPhone (Xcode) |
| [`vision/`](vision/) | Orin server: YOLO, depth, steering, dashboard | Jetson Orin Nano |
| [`ai/`](ai/) | Gemini (intent/find/memory/describe) + ElevenLabs checks | Any laptop |

For architecture, interfaces and the current status, see [CLAUDE.md](CLAUDE.md).

API keys go in environment variables only. Never commit them.
