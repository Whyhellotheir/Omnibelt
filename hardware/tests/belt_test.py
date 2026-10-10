"""Buzzes every motor from a computer. Works on Windows, Mac and Linux.
    pip install pyserial
    python belt_test.py COM9            # Windows: use the CP210x port from Device Manager
    python belt_test.py                 # Mac/Linux: finds the port by itself
Close the Arduino Serial Monitor first (only one program can use the port)."""
import glob, sys, time
import serial

if len(sys.argv) > 1:
    port = sys.argv[1]
else:
    found = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB*")
    if not found:
        sys.exit("No ESP32 port found. On Windows pass it, e.g.  python belt_test.py COM9")
    port = found[0]

s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0.2
s.dtr = False            # don't hold the ESP32 in reset / boot mode while the port is open
s.rts = False
s.open()
print("Opened", port, "- press the EN button on the ESP32 now")
t0 = time.time()
while time.time() - t0 < 4:                      # show the startup message if you press EN
    line = s.readline().decode(errors="ignore").strip()
    if line:
        print("ESP32:", line)

def cmd(text):
    s.write((text + "\n").encode())
    time.sleep(0.15)
    print(f"> {text:14s} {s.read_all().decode(errors='ignore').strip()}")

cmd("PING")
cmd("TEST")
time.sleep(5.2)
print(s.read_all().decode(errors="ignore").strip())
for motor in range(1, 8):                        # point around the clock, re-sending at 10 Hz
    for _ in range(10):
        s.write(f"M {motor} 80 P6\n".encode())
        time.sleep(0.1)
    s.write(f"M {motor} 0 S\n".encode())
cmd("SWEEP"); time.sleep(1.2)
cmd("ARRIVE"); time.sleep(1.2)
cmd("ALL 0")
print("If PING answered PONG and each motor buzzed in turn, the belt electronics are done.")
