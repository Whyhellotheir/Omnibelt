import Foundation

/// Edit these before running.
enum Config {
    /// Orin's name on the hotspot, or its IP (iPhone hotspot IPs look like 172.20.10.x).
    /// For the iOS-only test, use the laptop running tests/test_receiver.py.
    static var defaultHost = "orin.local"
    static let port = 8765
    static let fps: Double = 15
    static let elevenLabsKey = "PASTE_ELEVENLABS_KEY"   // empty/unchanged = Apple's built-in voice
    static let elevenLabsVoiceID = "PASTE_VOICE_ID"     // from tests/elevenlabs_test.py --voices
    static let wakeWord = "belt"                        // "Belt, find my keys"
    /// Short phrases generated once at start and played instantly afterwards.
    static let cachedPhrases = ["left", "right", "forward", "back toward you", "Stopped.",
                                "Walking. I'll steer you around obstacles."]
}
