import Foundation

/// Edit these before running.
enum Config {
    /// Orin's name on the hotspot, or its IP (iPhone hotspot IPs look like 172.20.10.x).
    /// For the iOS-only test, use the laptop running tests/test_receiver.py.
    static var defaultHost = "orin.local"
    static let port = 8765
    static let fps: Double = 15
    /// Landscape (pre-rotation) size of the sent image. Portrait frames are sent as frameHeight x frameWidth.
    static let frameWidth = 960                         // more detail for far objects (Orin zooms into crops of it)
    static let frameHeight = 720
    static let jpegQuality: CGFloat = 0.75
    static let wakeWord = "belt"                        // "Belt, find my keys" (short real word = recognized reliably)
    /// Any of these starts a command; the longest/latest match wins ("omnibelt" still works).
    static let wakeWords = ["belt", "omnibelt", "omni belt", "omni-belt", "omni"]
    /// Words the recognizer should expect (biases it away from "wall it", "fine my keys", ...).
    static let speechHints = ["Belt", "Omnibelt", "find my keys", "find my wallet", "find my water bottle",
                              "where did I leave my keys", "where did I leave my wallet", "take me to the door",
                              "this is my wallet", "these are my keys", "this is my water bottle",
                              "reach", "stop", "walk", "describe", "yes", "no", "keys", "wallet", "water bottle"]
    /// A command ending in one of these is complete: send after a short pause instead of waiting longer.
    static let completeEndings = ["keys", "wallet", "bottle", "door", "stop", "walk", "reach", "yes", "no",
                                  "describe", "phone", "glasses", "chair", "cup", "backpack", "laptop", "me"]
    /// On-device recognition is faster but less accurate; server recognition (via the hotspot's cellular) is better.
    static let onDeviceSpeech = false
    /// Short phrases generated once at start and played instantly afterwards.
    static let cachedPhrases = ["left", "right", "forward", "back toward you", "Stopped.",
                                "Walking. I'll steer you around obstacles."]

    /// Rotate the image (and intr/hands/fp/T) upright on the phone and send "rot". Off = always native landscape, no "rot".
    static let autoRotate = true
    /// Max ARKit feature points per frame ("fp").
    static let maxFeaturePoints = 150
    /// Planes message period (seconds) and max boundary vertices per plane.
    static let planesInterval: Double = 0.5
    static let maxPlaneVertices = 32
    /// Downsampled depth grid (landscape). Swapped when the frame is rotated to portrait.
    static let depthW = 128
    static let depthH = 96
    /// After the Orin asks a question (say text ends with "?"), the next utterance is a command without the wake word.
    static let replyWindowSeconds: Double = 8
}
