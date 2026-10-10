# NavBelt code

Follow the "Haptic Nav Belt — Build Guide" doc step by step; it says when to use each file.

| Folder | What | Runs on |
|---|---|---|
| esp32/navbelt_fw/navbelt_fw.ino | Belt firmware (motor commands, safety limits) | ESP32 (Arduino IDE) |
| orin/navbelt_server.py | The whole Orin program: phone socket, YOLO, depth, steering, Gemini, dashboard | Jetson Orin |
| ios/NavBelt/*.swift | iPhone app: camera + pose + hands stream, wake-word voice, ElevenLabs | iPhone (Xcode) |
| tests/belt_test.py | Buzzes every motor from a computer | Orin or laptop + ESP32 |
| tests/test_receiver.py | Fake Orin for testing the phone app | Laptop |
| tests/fake_phone.py | Fake phone (webcam/video) for testing the Orin server | Laptop or Orin |
| tests/yolo_test.py, depth_test.py | Vision model checks | Orin |
| tests/gemini_find_test.py, elevenlabs_test.py | API checks | Any computer |
| cad/parts.py | Regenerates every 3D print (edit the numbers at the top) | Any computer |

Orin quick start:
    pip install fastapi uvicorn pyserial google-genai transformers pillow websockets
    export GEMINI_API_KEY=...
    python3 orin/navbelt_server.py        # dashboard at http://orin.local:8765/
