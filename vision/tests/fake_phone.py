"""Stand-in for the phone, for the Orin track. Sends webcam (or video file) frames to the server
in the exact phone format, with a fixed forward-looking pose and no hands.
    pip install websockets opencv-python
    python fake_phone.py                              # webcam 0 -> ws://localhost:8765/stream
    python fake_phone.py --src walk.mp4 --host orin.local
Type a command and press Enter to send it as if spoken (e.g.  find my keys)."""
import argparse, asyncio, json, threading, time
import cv2, websockets

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="0")
ap.add_argument("--host", default="localhost")
ap.add_argument("--fps", type=float, default=15)
args = ap.parse_args()
cmds = []

def read_stdin():
    while True:
        try:
            cmds.append(input())
        except EOFError:
            return
threading.Thread(target=read_stdin, daemon=True).start()

# camera-to-world pose, column-major: looking along -Z, 1.3 m above the floor
T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1.3, 0, 1]

async def main():
    cap = cv2.VideoCapture(int(args.src) if args.src.isdigit() else args.src)
    async with websockets.connect(f"ws://{args.host}:8765/stream", max_size=None) as ws:
        async def reader():
            async for m in ws:
                print("ORIN SAYS:", m)
        asyncio.create_task(reader())
        seq = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (640, 480))
            meta = {"type": "frame", "seq": seq, "t": time.time(), "yaw": 0.0, "pitch": 0.0, "T": T,
                    "intr": [500.0, 500.0, 320.0, 240.0], "hands": [], "depth": False, "tracking": False}
            await ws.send(json.dumps(meta))
            await ws.send(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes())
            while cmds:
                await ws.send(json.dumps({"type": "cmd", "text": cmds.pop(0)}))
            seq += 1
            await asyncio.sleep(1 / args.fps)

asyncio.run(main())
