"""Run on the Orin (or a laptop) with the ESP32 plugged in by USB:  python3 belt_test.py
pip install pyserial.  If permission denied: sudo usermod -aG dialout $USER  (then log out/in)."""
import glob, time, serial

port = (glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB*"))[0]
s = serial.Serial(port, 115200, timeout=0.2)
time.sleep(2.0)                      # opening the port resets the ESP32; wait for it to boot
s.reset_input_buffer()

def cmd(line):
    s.write((line + "\n").encode()); time.sleep(0.05)
    print(f"> {line:14s} {s.read_all().decode(errors='ignore').strip()}")

cmd("PING")
cmd("TEST"); time.sleep(5.2)
# Simulate the Orin steering: point the belt around the clock, re-sending at 10 Hz (keeps the watchdog happy)
for motor in [1, 2, 3, 4, 5, 6, 7]:
    for _ in range(10):
        s.write(f"M {motor} 80 P6\n".encode()); time.sleep(0.1)
    s.write(f"M {motor} 0 S\n".encode())
cmd("SWEEP"); time.sleep(1.2)
cmd("ARRIVE"); time.sleep(1.2)
cmd("ALL 0")
print("If every motor buzzed in order M1..M7 (and M8 = ring), the belt electronics are done.")
