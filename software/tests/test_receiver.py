"""Stand-in for the Orin, for the iOS track. Run on any laptop on the same Wi-Fi/hotspot:
    pip install fastapi uvicorn opencv-python numpy
    python test_receiver.py
Point the app at this laptop's IP (port 8765). It prints frames/sec and hand points,
saves the newest frame to latest.jpg, and answers every voice command with a 'say' message."""
import json, time
import numpy as np, cv2, uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("Phone connected")
    meta, n, t0 = None, 0, time.time()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "frame":
                    meta = data
                elif data.get("type") == "cmd":
                    print("COMMAND:", data.get("text"))
                    await ws.send_text(json.dumps({"type": "say", "text": f"I heard: {data.get('text')}"}))
            elif msg.get("bytes") and meta is not None:
                if msg["bytes"][:2] != b"\xff\xd8":
                    continue                      # depth payload (LiDAR phones only); ignored here
                img = cv2.imdecode(np.frombuffer(msg["bytes"], np.uint8), cv2.IMREAD_COLOR)
                n += 1
                if n % 15 == 0:
                    cv2.imwrite("latest.jpg", img)
                    fps = n / (time.time() - t0)
                    hands = [(h.get("side"), h.get("wrist")) for h in meta.get("hands", [])]
                    print(f"{fps:5.1f} fps  img={img.shape[1]}x{img.shape[0]}  hands={hands}")
    except WebSocketDisconnect:
        pass
    print("Phone disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
