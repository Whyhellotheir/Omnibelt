import Foundation
import Speech
import AVFoundation
import Combine

/// Voice in: always-on speech recognition; anything said after the wake word ("Omnibelt") is sent as a command.
/// Voice out: ElevenLabs (cached phrases play instantly), falling back to Apple's built-in voice.
final class VoiceIO: NSObject, ObservableObject {
    @Published var lastHeard = ""
    @Published var lastSaid = ""

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var debounce: Timer?
    private var restartTimer: Timer?
    private var recognitionSession = 0
    private var active = false
    private var host = Config.defaultHost
    private weak var streamer: BeltStreamer?
    /// Reply window: after the Orin asks a question, the next utterance is sent without the wake word.
    private var replyUntil: Date?
    private var replyTimer: Timer?

    private var player: AVAudioPlayer?
    private let synth = AVSpeechSynthesizer()
    private var cache: [String: Data] = [:]

    func start(streamer: BeltStreamer, host: String) {
        self.streamer = streamer
        self.host = host
        active = true
        let audio = AVAudioSession.sharedInstance()
        try? audio.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth, .duckOthers])
        try? audio.setActive(true)
        SFSpeechRecognizer.requestAuthorization { status in
            guard status == .authorized else {
                DispatchQueue.main.async { self.lastHeard = "Speech permission denied" }
                return
            }
            audio.requestRecordPermission { granted in
                DispatchQueue.main.async {
                    if granted { self.listen() } else { self.lastHeard = "Microphone permission denied" }
                }
            }
        }
        for phrase in Config.cachedPhrases {
            fetch(phrase) { data in if let d = data { DispatchQueue.main.async { self.cache[phrase.lowercased()] = d } } }
        }
    }

    func stop() {
        active = false
        replyTimer?.invalidate()
        replyTimer = nil
        replyUntil = nil
        stopListening()
    }

    // MARK: - Listening
    private func listen() {
        guard active, let recognizer = recognizer, recognizer.isAvailable else { return }
        stopListening()
        recognitionSession += 1
        let session = recognitionSession
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition { req.requiresOnDeviceRecognition = true }
        request = req
        let input = engine.inputNode
        input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { [weak self] buffer, _ in
            self?.request?.append(buffer)
        }
        engine.prepare()
        try? engine.start()
        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self = self else { return }
            DispatchQueue.main.async {
                guard self.active, session == self.recognitionSession else { return }
                if let result = result {
                    let text = result.bestTranscription.formattedString.lowercased()
                    self.lastHeard = text
                    // Speech that starts inside the reply window keeps it open until the utterance finishes.
                    if let until = self.replyUntil, Date() < until {
                        self.replyUntil = max(until, Date().addingTimeInterval(2.5))
                    }
                    self.debounce?.invalidate()
                    self.debounce = Timer.scheduledTimer(withTimeInterval: 1.2, repeats: false) { _ in
                        guard session == self.recognitionSession else { return }
                        self.finish(text)
                    }
                }
                if error != nil {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        guard session == self.recognitionSession else { return }
                        self.listen()
                    }
                }
            }
        }
        // Apple limits one recognition task to about a minute: restart regularly.
        restartTimer?.invalidate()
        restartTimer = Timer.scheduledTimer(withTimeInterval: 50, repeats: false) { [weak self] _ in self?.listen() }
    }

    private func stopListening() {
        recognitionSession += 1
        debounce?.invalidate()
        debounce = nil
        restartTimer?.invalidate()
        restartTimer = nil
        if engine.isRunning { engine.stop() }
        engine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
    }

    private func finish(_ text: String) {
        let trimmed = text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
        let wakePhrases = [Config.wakeWord, "omni belt", "omni-belt"]
        let wakeRange = wakePhrases.compactMap {
            trimmed.range(of: $0, options: .backwards)
        }.max { $0.lowerBound < $1.lowerBound }

        if let wakeRange {
            let command = trimmed[wakeRange.upperBound...]
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
            if !command.isEmpty {
                streamer?.sendCommand(command)
                lastHeard = "→ " + command
                replyUntil = nil
            }
        } else if let until = replyUntil, Date() < until, !trimmed.isEmpty, !isEcho(trimmed) {
            streamer?.sendCommand(trimmed)          // e.g. "yes" / "no" right after a question
            lastHeard = "↩ " + trimmed
            replyUntil = nil
        }
        listen()                                    // fresh task, so old words aren't re-sent
    }

    /// True if the transcript is just our own spoken question picked up by the mic.
    private func isEcho(_ heard: String) -> Bool {
        func norm(_ s: String) -> String {
            s.lowercased().components(separatedBy: CharacterSet.alphanumerics.union(.whitespaces).inverted).joined()
        }
        let h = norm(heard).trimmingCharacters(in: .whitespaces)
        return h.count > 6 && norm(lastSaid).contains(h)
    }

    /// Opens the reply window once our question has finished playing, with a fresh recognition task.
    private func openReplyWindow(after delay: TimeInterval) {
        replyTimer?.invalidate()
        replyTimer = Timer.scheduledTimer(withTimeInterval: max(0.1, delay), repeats: false) { [weak self] _ in
            guard let self, self.active else { return }
            self.replyUntil = Date().addingTimeInterval(Config.replyWindowSeconds)
            self.listen()
        }
    }

    // MARK: - Speaking
    func say(_ text: String) {
        lastSaid = text
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("?")
        if let data = cache[text.lowercased()] { play(data, question: question); return }
        fetch(text) { data in
            DispatchQueue.main.async {
                if let d = data {
                    self.play(d, question: question)
                } else {
                    self.synth.speak(AVSpeechUtterance(string: text))
                    if question { self.openReplyWindow(after: Double(text.count) / 14.0 + 0.5) }   // rough speech length
                }
            }
        }
    }

    private func play(_ data: Data, question: Bool = false) {
        player = try? AVAudioPlayer(data: data)
        player?.play()
        if question { openReplyWindow(after: (player?.duration ?? 2) + 0.3) }
    }

    private func fetch(_ text: String, completion: @escaping (Data?) -> Void) {
        guard let url = URL(string: "http://\(host):\(Config.port)/tts") else {
            completion(nil)
            return
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("audio/mpeg", forHTTPHeaderField: "Accept")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["text": text])
        URLSession.shared.dataTask(with: req) { data, response, _ in
            let ok = (response as? HTTPURLResponse)?.statusCode == 200
            completion(ok ? data : nil)
        }.resume()
    }
}
