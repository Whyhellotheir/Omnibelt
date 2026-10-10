# Hardware (Track A)

The ESP32 + ULN2803 belt hub, 7 belt motors plus the ring motor (M8), and the 3D prints.

| Path | What |
|---|---|
| `firmware/navbelt_fw/navbelt_fw.ino` | Belt firmware. Serial at 115200; motor commands; max 3 motors on; 500 ms watchdog |
| `tests/belt_test.py` | Buzzes every motor and pattern from a computer (`pip install pyserial`; Windows: `python belt_test.py COM9`) |
| `cad/parts.py` | Regenerates every 3D print (edit the numbers at the top) |

The pin map, wiring and the Stage 1A steps are in [../CLAUDE.md](../CLAUDE.md#hardware-now-stage-1a).
