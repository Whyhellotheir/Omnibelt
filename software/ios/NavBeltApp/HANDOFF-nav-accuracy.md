# Handoff: iOS nav-accuracy changes (for Helena)

Written on Windows without a compiler. Please build and run through the checklist below. Nothing is committed.

## What changed

### Config.swift
New constants: `autoRotate` (true), `maxFeaturePoints` (150), `planesInterval` (0.5 s), `maxPlaneVertices` (32), `depthW/depthH` (128/96), `replyWindowSeconds` (8).

### BeltStreamer.swift
- **Camera profiles**: new `enum CameraProfile` = `auto | 1x | 0.75x | 0.5x`, saved in UserDefaults key `cameraProfile`. The default is `auto`, which keeps the old behavior: 1x, auto-switching to 0.5x on "reach out" / "right in front of you" and on `{"type":"camera_mode","mode":"search"}`.
  - `1x` = `builtInWideAngleCamera` format. `0.5x` = `builtInUltraWideCamera` format. `0.75x` = ultra-wide with a centered 2/3 crop before scaling to 640x480. If ultra-wide isn't available, it falls back to 1x and sends `"camera":"1x"`.
  - Orin `{"type":"camera_mode","zoom":"1x"|"0.75x"|"0.5x"}` sets and persists the profile, then restarts the session with `[.resetTracking, .removeExistingAnchors]`. Picking a profile in the UI does the same.
  - `ARWorldTrackingConfiguration.supportedVideoFormats` is logged once at Start to the console (`print`) and to a "ARKit video formats" disclosure in the UI.
- **Orientation**: `k` is computed from `camera.transform` (phi = atan2(R[1,0], R[1,1]), 55° hysteresis, kept when |(ax,ay)| < 0.5), following the contract. When k≠0, the JPEG is rotated with `CIImage.oriented(.right/.left/.down)` for k = 1/3/2. intr, hands, fp and T's camera axes are mapped with the contract formulas, and `"rot": k` is always sent while `Config.autoRotate` is on. In portrait, `W`/`H` = 480x640.
- **Hands**: Vision runs on the full-resolution buffer with the matching `CGImagePropertyOrientation`, so it sees upright hands. Points are mapped upright → native → crop/scale → rotation into sent pixels. Points outside the crop are dropped, and a hand with no in-image points is dropped. bbox is computed from the mapped points.
- **Feature points**: `frame.rawFeaturePoints` are captured in `session(_:didUpdate:)` and projected in `process`. Only points with 0.25 < z < 6 m that land inside the image are kept. The list is subsampled evenly to ≤150 and sent as `"fp": [[u, v, z]]`, with u,v rounded to 0.1 px and z to 0.01 m.
- **Planes**: about every 0.5 s (frame timestamps), a text message `{"type":"planes","planes":[{"id","cls","T","poly"}]}` is sent for horizontal `ARPlaneAnchor`s. `T` = anchor transform (16, column-major). `poly` = `geometry.boundaryVertices` as `[x,z]` (anchor-local, 1 mm rounding, ≤32 vertices). `cls` ∈ floor|table|seat|ceiling|none. **Server note:** a planes message can arrive between a frame's meta text and its JPEG binary, so the server must not reset `meta`/`expect` on it (the current handler ignores unknown types, which is correct).
- **Depth** (LiDAR only, so not on the iPhone 17): resampled onto the sent-image grid with crop and rotation applied. `dw,dh` = 128x96, or 96x128 when rotated.
- The meta still includes `type, seq, t, tracking, yaw, pitch, T, intr, hands, depth, dw, dh, camera`, and also `W, H` plus the legacy `width, height`. New: `fp`, `rot`. `yaw`/`pitch` are still raw `eulerAngles` and are not rotated.
- The status line is now `Streaming N fps · <camera> · rot k · hands n · fp n`.

### VoiceIO.swift
- **Reply window**: if a `say` text ends with `?`, then after the question finishes playing (AVAudioPlayer `duration` + 0.3 s, or a length estimate for the Apple-voice fallback), an 8 s window opens and recognition restarts so the question's own audio isn't in the transcript. The next final utterance is sent as a `cmd` without the wake word. The UI shows `↩ yes`. Speech that starts inside the window extends it by 2.5 s so it isn't cut off. Echo guard: a transcript longer than 6 characters that is contained in the spoken question is ignored.
- The wake-word variants are unchanged: `omnibelt`, `omni belt`, `omni-belt`. The ElevenLabs `/tts` path is unchanged.

### ContentView.swift
A segmented "Camera" picker (Auto / 1x / 0.75x / 0.5x) and an "ARKit video formats" disclosure.

## Build
1. Open `software/ios/NavBeltApp/NavBeltApp.xcodeproj` in Xcode 15+ (iOS 17 target), select the iPhone and press Run.
2. No new frameworks are needed. The new imports `ImageIO` and `simd` are system modules.

## On-device test checklist
- [ ] App launches. Picker shows **Auto** on first install. Expand "ARKit video formats" and confirm UltraWide entries are listed (note resolutions here: ____).
- [ ] Pick **1x / 0.75x / 0.5x**: status shows `Camera … · tracking reset`, then the streaming line shows the matching camera label. Kill and relaunch the app: the choice persists.
- [ ] **0.75x** on the Orin dashboard: the field of view sits between 0.5x and 1x and the image is centered (the same object stays in the middle when you switch 0.5x → 0.75x).
- [ ] **Portrait mount** (phone upright on the belt): status shows `rot 1` and the Orin dashboard image is upright, 480x640. Turn the phone back to landscape: `rot 0` and the image is upright. Upside-down landscape gives `rot 2`.
- [ ] Hands in portrait: hold a hand in front. Status `hands 1` and the dashboard hand overlay lands on the hand.
- [ ] `fp` count in status is >0 (up to 150) after moving the phone a bit to give ARKit texture.
- [ ] Server log / a debug print shows `{"type":"planes",...}` messages about every 0.5 s once a floor or table is detected. The server must add a handler; today it silently ignores them.
- [ ] From the Orin: `{"type":"camera_mode","zoom":"0.5x"}` switches the picker to 0.5x and resets tracking. `{"type":"camera_mode","mode":"search"}` only does something in Auto.
- [ ] Reply window: make the Orin say something ending in `?` (or temporarily call `voice.say("Should I continue?")`). After it finishes speaking, say "yes" with no wake word. "Heard" shows `↩ yes` and the server gets `{"type":"cmd","text":"yes"}`. After 8 s of silence, "yes" alone does nothing again.
- [ ] Wake word still works: "Omnibelt, find my keys".

## Lines I'm not 100% sure compile (or behave) — check these first
1. `image.oriented(Self.exif(k))` with `k=1 → .right`: I expect `.right` to rotate the buffer **90° clockwise** (the standard portrait back-camera fix), which matches contract k=1 = `cv2.ROTATE_90_CLOCKWISE`. If the portrait image arrives upside down on the dashboard, swap `.right`/`.left` in `exif(_:)` **and** check that hands/fp still align. The Vision handler uses the same function, so they stay consistent either way.
2. After `.oriented(...)` I translate by `-image.extent.origin` in case the extent isn't at (0,0). This should be harmless.
3. Vision normalized coordinates with a non-`.up` orientation are assumed to be relative to the **oriented (upright)** image (`FrameGeom.nativeFromUpright`). If hand points land rotated or mirrored in portrait only, this is the cause.
4. `for case let p as ARPlaneAnchor in anchors where p.alignment == .horizontal` and the `switch p.classification { case .floor: … default: … }` (the `.none(Status)` case is covered by `default`).
5. `let c = inv * simd_float4(p.x, p.y, p.z, 1)` (`simd_float4x4 * simd_float4`) and `-t.columns.0` (unary minus on `SIMD4<Float>`).
6. `private struct FrameGeom` at file scope is used in a `private static func` signature inside the class. This should be fine (file-private).
7. `@Published var profile: CameraProfile = CameraProfile.load()` and the ContentView `Picker` with `Binding(get:set:)` + `ForEach(CameraProfile.allCases)` (the enum is `Identifiable`). `setProfile` updates `profile` asynchronously on main, so the segment may lag by one runloop.
8. `.font(.caption2.monospaced())` (iOS 15+).
9. VoiceIO: `guard let self, self.active else { return }` inside a Timer closure (Swift 5.7+ shorthand, already used elsewhere in the project).

## Known limitations / notes for the orchestrator
- `OrinCameraView` (not edited) uses a fixed 4:3 aspect, so a portrait stream will be letterboxed or squeezed in the in-app preview. Change it to `3/4` or make it flexible if that matters.
- The server currently resizes every JPEG to its global `W,H` (640x480) and applies `FLIP_IMAGE`. With `rot` present it should use `meta["W"], meta["H"]` and skip its own rotation.
- In Auto, the legacy lens switch still does `session.run(config)` **without** resetting tracking (old behavior). Fixed profiles always reset.
- `lens` (camera label/crop flag) is snapshotted per frame in `didUpdate`. For a few frames right after a switch, the label can be stale. A fixed-profile switch also resets tracking, so those frames have `tracking:false`.

## Review fixes (please build + check on device)
- **F9 VoiceIO reply window**: `say(_:)` now closes any open reply window first (`replyUntil = nil`, `replyTimer?.invalidate()`), so the belt's own short cues ("left", "forward") can't be sent back as commands. A no-wake-word reply is only accepted while nothing is playing (`player?.isPlaying != true`, `!synth.isSpeaking`) and if it is at most 4 words; the echo check stays. Test: after "Want me to take you there?", "yes" still works; a long sentence from someone nearby does not.
- **F10 VoiceIO**: speech inside the reply window extends it by 2.5 s **at most once** per window (`replyExtended`, reset when the window opens). Before, every partial result extended it forever.
- **F11 BeltStreamer.runSession**: for the wanted lens, a 4:3 ARKit format is preferred (`abs(w/h - 4/3) < 0.01` on `imageResolution`), then the first format of that device, then the old 1x fallback. Check the "ARKit video formats" list shows a 4:3 entry is picked (status/stream resolution should be 4:3).
- Server side: the dashboard camera control now includes **auto** (default) and sends `{"type":"camera_mode","zoom":"auto"}`; re-selecting the current lens no longer resets anything.

## Wave 3: speech reliability (rebuild needed)
- **Wake word is now "Belt"** (`Config.wakeWords`: belt, omnibelt, omni belt, omni-belt, omni). A short real word is recognized far more reliably than "Omnibelt".
- **Vocabulary hints:** `req.contextualStrings = Config.speechHints`, which biases recognition toward keys, wallet, water bottle, "where did I leave", and so on.
- **Server recognition by default:** `Config.onDeviceSpeech = false` gives more accurate results over the hotspot's cellular. Set it to true if there's no internet.
- **Listening cue:** a system click (sound 1113) plays the moment the wake word is heard.
- **Smarter end of speech:** it sends after 0.7 s of silence when the sentence ends in a known word (keys, wallet, stop, …), otherwise after 1.6 s. Fewer cut-off commands.
- **No self-triggers:** anything heard while the phone is speaking is ignored, unless it contains "stop". The belt's own "Follow the belt." can't fire a command.
- **Compile notes:** this adds `import AudioToolbox` (for `AudioServicesPlaySystemSound`). No other new APIs.
- **Test:** say "Belt, find my wallet" in a noisy room. You should hear the click, and the dashboard transcript should show "find my wallet".
