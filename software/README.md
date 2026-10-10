# Software (Track B · iOS)

The iPhone app streams camera frames, its ARKit pose and hand points to the Orin at about 15 fps. It listens for the wake word "belt" and speaks replies through ElevenLabs.

| Path | What |
|---|---|
| `ios/NavBelt/*.swift` | Drag these into a new SwiftUI Xcode project named NavBelt |
| `ios/NavBelt/Info-plist-keys.txt` | Required permissions and settings (Local Network + Allow Local Networking) |
| `tests/test_receiver.py` | Stand-in Orin on a laptop for testing the app |

Paste the ElevenLabs key into `Config.swift` **locally only**. Never commit it.
