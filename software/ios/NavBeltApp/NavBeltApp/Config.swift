import Foundation

/// Edit these before running.
enum Config {
    /// Orin's name on the hotspot, or its IP (iPhone hotspot IPs look like 172.20.10.x).
    /// For the iOS-only test, use the laptop running tests/test_receiver.py.
    static var defaultHost = "orin.local"
    static let port = 8765
    static let fps: Double = 15
    /// Landscape (pre-rotation) size of the sent image. Portrait frames are sent as frameHeight x frameWidth.
    static let frameWidth = 640
    static let frameHeight = 480
    static let jpegQuality: CGFloat = 0.75
    static let wakeWord = "omnibelt"                    // "Omnibelt, find my keys"
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
