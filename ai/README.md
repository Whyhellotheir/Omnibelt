# AI (Track D · Gemini + ElevenLabs + pitch)

| Path | What |
|---|---|
| `tests/gemini_find_test.py` | Gemini finds an object by description (`desk.jpg "keys"`) or parses intent (`--intent "..."`) |
| `tests/elevenlabs_test.py` | Lists voices and tests text-to-speech |
| `tests/voice_test.py` | A photo + a spoken command through the real server logic; speaks every line the belt would say (`desk.jpg "find my mouse"`) |

The prompts (`INTENT_PROMPT`, `SCAN_PROMPT`), the spoken `say(...)` lines and `DASHBOARD_HTML` live in `../vision/orin/navbelt_server.py`. Coordinate with Vision before editing them.

Keys come from env vars only: `GEMINI_API_KEY` and `ELEVENLABS_API_KEY`.
