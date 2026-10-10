import ARKit
import Vision
import CoreImage
import UIKit

/// Streams camera frames, ARKit pose and hand points to the Orin over a WebSocket,
/// and hands any {"type":"say"} reply to VoiceIO.
final class BeltStreamer: NSObject, ObservableObject, ARSessionDelegate {
    @Published var running = false
    @Published var status = "Idle"

    private let session = ARSession()
    private let urlSession = URLSession(configuration: .default)
    private var socket: URLSessionWebSocketTask?
    private var host = Config.defaultHost
    private let ciContext = CIContext()
    private let work = DispatchQueue(label: "navbelt.encode")
    private var lastSent: TimeInterval = 0
    private var busy = false
    private var seq = 0
    private var sentSince = Date()
    private var sentCount = 0
    private weak var voice: VoiceIO?

    func start(host: String, voice: VoiceIO) {
        self.host = host
        self.voice = voice
        connect()
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)          // LiDAR phones only
        }
        session.delegate = self
        session.delegateQueue = DispatchQueue(label: "navbelt.ar")
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        running = true
    }

    func stop() {
        running = false
        session.pause()
        socket?.cancel(with: .goingAway, reason: nil)
        status = "Stopped"
    }

    // MARK: - WebSocket
    private func connect() {
        guard let url = URL(string: "ws://\(host):\(Config.port)/stream") else { return }
        socket = urlSession.webSocketTask(with: url)
        socket?.maximumMessageSize = 8 * 1024 * 1024
        socket?.resume()
        DispatchQueue.main.async { self.status = "Connecting to \(self.host)…" }
        receive()
    }

    private func receive() {
        socket?.receive { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .failure(let error):
                DispatchQueue.main.async { self.status = "Disconnected: \(error.localizedDescription)" }
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { if self.running { self.connect() } }
            case .success(let message):
                if case .string(let text) = message,
                   let data = text.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   obj["type"] as? String == "say", let say = obj["text"] as? String {
                    DispatchQueue.main.async { self.voice?.say(say) }
                }
                self.receive()
            }
        }
    }

    func sendCommand(_ text: String) {
        sendJSON(["type": "cmd", "text": text])
    }

    private func sendJSON(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let text = String(data: data, encoding: .utf8) else { return }
        socket?.send(.string(text)) { _ in }
    }

    // MARK: - Frames
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = frame.timestamp
        guard !busy, now - lastSent >= 1.0 / Config.fps else { return }
        lastSent = now
        busy = true
        let pixels = frame.capturedImage
        let camera = frame.camera
        let depth = frame.sceneDepth?.depthMap
        work.async { [weak self] in
            self?.process(pixels: pixels, camera: camera, depth: depth, time: now)
            self?.busy = false
        }
    }

    private func process(pixels: CVPixelBuffer, camera: ARCamera, depth: CVPixelBuffer?, time: TimeInterval) {
        let outW: CGFloat = 640, outH: CGFloat = 480
        let res = camera.imageResolution                          // e.g. 1920x1440, landscape
        let sx = outW / res.width, sy = outH / res.height

        // 1. JPEG 640x480
        let scaled = CIImage(cvPixelBuffer: pixels).transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        guard let cg = ciContext.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: outW, height: outH)),
              let jpeg = UIImage(cgImage: cg).jpegData(compressionQuality: 0.5) else { return }

        // 2. Hands (Apple Vision), in 640x480 pixel coordinates, origin top-left
        var hands: [[String: Any]] = []
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2
        try? VNImageRequestHandler(cvPixelBuffer: pixels, orientation: .up).perform([request])
        for hand in request.results ?? [] {
            guard let points = try? hand.recognizedPoints(.all) else { continue }
            func px(_ joint: VNHumanHandPoseObservation.JointName) -> [Double]? {
                guard let p = points[joint], p.confidence > 0.3 else { return nil }
                return [Double(p.location.x) * Double(outW), Double(1 - p.location.y) * Double(outH)]
            }
            var h: [String: Any] = ["conf": Double(hand.confidence)]
            switch hand.chirality {
            case .left: h["side"] = "left"
            case .right: h["side"] = "right"
            default: h["side"] = "unknown"
            }
            if let w = px(.wrist) { h["wrist"] = w }
            if let i = px(.indexTip) { h["index"] = i }
            hands.append(h)
        }

        // 3. Pose + intrinsics
        let t = camera.transform
        let T: [Float] = [t.columns.0.x, t.columns.0.y, t.columns.0.z, t.columns.0.w,
                          t.columns.1.x, t.columns.1.y, t.columns.1.z, t.columns.1.w,
                          t.columns.2.x, t.columns.2.y, t.columns.2.z, t.columns.2.w,
                          t.columns.3.x, t.columns.3.y, t.columns.3.z, t.columns.3.w]
        let k = camera.intrinsics
        let intr = [Double(k[0][0] * Float(sx)), Double(k[1][1] * Float(sy)),
                    Double(k[2][0] * Float(sx)), Double(k[2][1] * Float(sy))]

        // 4. Optional LiDAR depth, downsampled to 128x96 float16
        var depthData: Data? = nil
        if let d = depth { depthData = Self.downsample(depth: d, toW: 128, toH: 96) }

        seq += 1
        let meta: [String: Any] = ["type": "frame", "seq": seq, "t": time,
                                   "yaw": Double(camera.eulerAngles.y), "pitch": Double(camera.eulerAngles.x),
                                   "T": T.map { Double($0) }, "intr": intr, "hands": hands,
                                   "depth": depthData != nil, "dw": 128, "dh": 96]
        sendJSON(meta)
        socket?.send(.data(jpeg)) { _ in }
        if let dd = depthData { socket?.send(.data(dd)) { _ in } }

        sentCount += 1
        if Date().timeIntervalSince(sentSince) > 1 {
            let fps = Double(sentCount) / Date().timeIntervalSince(sentSince)
            let hc = hands.count
            DispatchQueue.main.async { self.status = String(format: "Streaming %.0f fps · hands: %d", fps, hc) }
            sentCount = 0
            sentSince = Date()
        }
    }

    private static func downsample(depth: CVPixelBuffer, toW: Int, toH: Int) -> Data {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        let rowBytes = CVPixelBufferGetBytesPerRow(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return Data() }
        var out = [Float16](repeating: 0, count: toW * toH)
        for y in 0..<toH {
            let row = base.advanced(by: (y * h / toH) * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in 0..<toW { out[y * toW + x] = Float16(row[x * w / toW]) }
        }
        return out.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
