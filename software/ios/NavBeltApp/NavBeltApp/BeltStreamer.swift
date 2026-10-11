import ARKit
import AVFoundation
import Vision
import CoreImage
import ImageIO
import UIKit
import Combine
import simd

/// Camera profile chosen in the UI or by the Orin. "auto" = legacy behavior
/// (1x, switching to 0.5x on reach phrases / {"camera_mode","mode":"search"}).
enum CameraProfile: String, CaseIterable, Identifiable {
    case auto = "auto"
    case wide = "1x"
    case mid = "0.75x"
    case ultraWide = "0.5x"

    var id: String { rawValue }
    var label: String { self == .auto ? "Auto" : rawValue }

    static let defaultsKey = "cameraProfile"
    static func load() -> CameraProfile {
        CameraProfile(rawValue: UserDefaults.standard.string(forKey: defaultsKey) ?? "") ?? .auto
    }
}

/// Native capture pixels -> sent-image pixels: optional centered 2/3 crop, scale to
/// Config.frameWidth x frameHeight, then rotation k (contract: 1 = 90 CW, 3 = 90 CCW, 2 = 180).
/// Native pixels are top-left origin (u right, v down), same as ARKit intrinsics.
private struct FrameGeom {
    let resW, resH, x0, y0, cw, ch, sx, sy, outW, outH: Double
    let k: Int

    init(resW: Double, resH: Double, crop: Bool, outW: Double, outH: Double, k: Int) {
        let cropW = crop ? resW * 2 / 3 : resW
        let cropH = crop ? resH * 2 / 3 : resH
        self.resW = resW
        self.resH = resH
        self.cw = cropW
        self.ch = cropH
        self.x0 = (resW - cropW) / 2
        self.y0 = (resH - cropH) / 2
        self.sx = outW / cropW
        self.sy = outH / cropH
        self.outW = outW
        self.outH = outH
        self.k = k
    }

    var finalW: Double { k % 2 == 1 ? outH : outW }
    var finalH: Double { k % 2 == 1 ? outW : outH }

    /// Pre-rotation (outW x outH) -> sent image.
    func rotate(_ x: Double, _ y: Double) -> (Double, Double) {
        switch k {
        case 1: return (outH - y, x)
        case 3: return (y, outW - x)
        case 2: return (outW - x, outH - y)
        default: return (x, y)
        }
    }

    /// Sent image -> pre-rotation (outW x outH).
    func unrotate(_ X: Double, _ Y: Double) -> (Double, Double) {
        switch k {
        case 1: return (Y, outH - X)
        case 3: return (outW - Y, X)
        case 2: return (outW - X, outH - Y)
        default: return (X, Y)
        }
    }

    /// Native pixel -> sent pixel, nil if outside the sent image.
    func map(_ u: Double, _ v: Double) -> (Double, Double)? {
        let x = (u - x0) * sx, y = (v - y0) * sy
        guard x >= 0, x < outW, y >= 0, y < outH else { return nil }
        return rotate(x, y)
    }

    /// Pixel in the upright (oriented, uncropped, native-resolution) image Vision sees -> native pixel.
    func nativeFromUpright(_ X: Double, _ Y: Double) -> (Double, Double) {
        switch k {
        case 1: return (Y, resH - X)
        case 3: return (resW - Y, X)
        case 2: return (resW - X, resH - Y)
        default: return (X, Y)
        }
    }

    func intrinsics(fx: Double, fy: Double, cx: Double, cy: Double) -> [Double] {
        let f = [fx * sx, fy * sy, (cx - x0) * sx, (cy - y0) * sy]
        switch k {
        case 1: return [f[1], f[0], outH - f[3], f[2]]
        case 3: return [f[1], f[0], f[3], outW - f[2]]
        case 2: return [f[0], f[1], outW - f[2], outH - f[3]]
        default: return f
        }
    }
}

/// Streams camera frames, ARKit pose, hand points, feature points and planes to the Orin
/// over a WebSocket, and hands any {"type":"say"} reply to VoiceIO.
final class BeltStreamer: NSObject, ObservableObject, ARSessionDelegate {
    private enum CameraMode: String {
        case navigation = "1x"
        case localSearch = "0.5x"
    }

    private struct Lens {
        var device: AVCaptureDevice.DeviceType
        var crop: Bool
        var label: String
    }

    @Published var running = false
    @Published var status = "Idle"
    @Published var profile: CameraProfile = CameraProfile.load()
    @Published var formatsInfo = ""

    private let session = ARSession()
    private let urlSession = URLSession(configuration: .default)
    private var socket: URLSessionWebSocketTask?
    private var host = Config.defaultHost
    private let ciContext = CIContext()
    private let work = DispatchQueue(label: "navbelt.encode")
    private var lastSent: TimeInterval = 0
    private var lastPlanes: TimeInterval = 0
    private var busy = false
    private var seq = 0
    private var sentSince = Date()
    private var sentCount = 0
    private var cameraMode = CameraMode.navigation          // legacy, only used when profile == .auto
    private var rotK = 0                                    // only touched on the work queue
    private var loggedFormats = false
    private weak var voice: VoiceIO?

    private let lensLock = NSLock()
    private var lensState = Lens(device: .builtInWideAngleCamera, crop: false, label: "1x")
    private var lens: Lens {
        get { lensLock.lock(); defer { lensLock.unlock() }; return lensState }
        set { lensLock.lock(); lensState = newValue; lensLock.unlock() }
    }

    func start(host: String, voice: VoiceIO) {
        self.host = host
        self.voice = voice
        logFormatsOnce()
        connect()
        session.delegate = self
        session.delegateQueue = DispatchQueue(label: "navbelt.ar")
        cameraMode = .navigation
        runSession(reset: true)
        running = true
    }

    func stop() {
        running = false
        session.pause()
        socket?.cancel(with: .goingAway, reason: nil)
        cameraMode = .navigation
        status = "Stopped"
    }

    // MARK: - Camera profile / lens
    private var ultraWideAvailable: Bool {
        ARWorldTrackingConfiguration.supportedVideoFormats.contains { $0.captureDeviceType == .builtInUltraWideCamera }
    }

    private func logFormatsOnce() {
        guard !loggedFormats else { return }
        loggedFormats = true
        let lines = ARWorldTrackingConfiguration.supportedVideoFormats.map { f -> String in
            let device = f.captureDeviceType.rawValue.replacingOccurrences(of: "AVCaptureDeviceType", with: "")
            return "\(device) \(Int(f.imageResolution.width))x\(Int(f.imageResolution.height)) @\(f.framesPerSecond)fps"
        }
        let text = lines.joined(separator: "\n")
        print("ARKit supportedVideoFormats:\n\(text)")
        formatsInfo = text
    }

    private func desiredLens() -> Lens {
        let wide = Lens(device: .builtInWideAngleCamera, crop: false, label: "1x")
        let ultraOK = ultraWideAvailable
        switch profile {
        case .wide:
            return wide
        case .mid:
            return ultraOK ? Lens(device: .builtInUltraWideCamera, crop: true, label: "0.75x") : wide
        case .ultraWide:
            return ultraOK ? Lens(device: .builtInUltraWideCamera, crop: false, label: "0.5x") : wide
        case .auto:
            return (cameraMode == .localSearch && ultraOK)
                ? Lens(device: .builtInUltraWideCamera, crop: false, label: "0.5x") : wide
        }
    }

    private func runSession(reset: Bool) {
        let wanted = desiredLens()
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.captureDeviceType == wanted.device
        }) {
            config.videoFormat = format
            lens = wanted
        } else {
            lens = Lens(device: .builtInWideAngleCamera, crop: false, label: "1x")
        }
        if reset {
            session.run(config, options: [.resetTracking, .removeExistingAnchors])
        } else {
            session.run(config)
        }
    }

    /// Fixed profile from the UI picker or the Orin. Persists, and restarts tracking (lens change resets world coords).
    func setProfile(_ newProfile: CameraProfile) {
        DispatchQueue.main.async {
            UserDefaults.standard.set(newProfile.rawValue, forKey: CameraProfile.defaultsKey)
            guard newProfile != self.profile else { return }
            self.profile = newProfile
            self.cameraMode = .navigation
            if self.running {
                self.runSession(reset: true)
                self.status = "Camera \(newProfile.label) · tracking reset"
            }
        }
    }

    /// Legacy search/navigation lens switching, only active in the "auto" profile.
    private func setCameraMode(_ requestedMode: CameraMode) {
        guard running, profile == .auto, requestedMode != cameraMode else { return }
        if requestedMode == .localSearch, !ultraWideAvailable {
            DispatchQueue.main.async {
                self.status = "Local search · 0.5x unavailable, using 1x"
            }
            return
        }

        cameraMode = requestedMode
        runSession(reset: false)
        DispatchQueue.main.async {
            self.status = requestedMode == .localSearch ? "Local search · 0.5x" : "Navigation · 1x"
        }
    }

    private func updateCameraMode(forSpokenText text: String) {
        guard profile == .auto else { return }
        let normalized = text.lowercased()
        if normalized.contains("right in front of you") || normalized.contains("reach out") {
            setCameraMode(.localSearch)
        } else if normalized.hasPrefix("got it") || normalized.hasPrefix("stopped") {
            setCameraMode(.navigation)
        }
    }

    private func actionPrompt(for text: String) -> String {
        let normalized = text.lowercased()
        if normalized.hasPrefix("walk") || normalized.hasPrefix("reach") {
            return text
        }

        let reachPhrases = ["reach out", "right in front of you", "within reach", "move your hand", "grab", "pick up"]
        if reachPhrases.contains(where: normalized.contains) {
            return "Reach. \(text)"
        }

        let walkPhrases = ["turn left", "turn right", "go left", "go right", "forward", "backward", "straight ahead", "keep walking", "continue walking", "step left", "step right"]
        if walkPhrases.contains(where: normalized.contains) {
            return "Walk. \(text)"
        }

        return text
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
                   let type = obj["type"] as? String {
                    if type == "say", let say = obj["text"] as? String {
                        self.updateCameraMode(forSpokenText: say)
                        let promptedText = self.actionPrompt(for: say)
                        DispatchQueue.main.async { self.voice?.say(promptedText) }
                    } else if type == "camera_mode" {
                        if let zoom = obj["zoom"] as? String, let p = CameraProfile(rawValue: zoom) {
                            self.setProfile(p)
                        } else if let mode = obj["mode"] as? String {
                            self.setCameraMode(mode == "search" ? .localSearch : .navigation)
                        }
                    }
                }
                self.receive()
            }
        }
    }

    func sendCommand(_ text: String) {
        let normalized = text.lowercased()
        if normalized.contains("find") || normalized.contains("locate") || normalized.contains("where") {
            setCameraMode(.navigation)
        }
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
        if now - lastPlanes >= Config.planesInterval {
            lastPlanes = now
            sendPlanes(frame.anchors)
        }
        guard !busy, now - lastSent >= 1.0 / Config.fps else { return }
        lastSent = now
        busy = true
        let pixels = frame.capturedImage
        let camera = frame.camera
        let depth = frame.sceneDepth?.depthMap
        let featurePoints: [simd_float3] = frame.rawFeaturePoints?.points ?? []
        let activeLens = lens
        work.async { [weak self] in
            self?.process(pixels: pixels, camera: camera, depth: depth, featurePoints: featurePoints,
                          lens: activeLens, time: now)
        }
    }

    // MARK: - Planes
    private func sendPlanes(_ anchors: [ARAnchor]) {
        guard socket != nil else { return }
        var planes: [[String: Any]] = []
        for case let p as ARPlaneAnchor in anchors where p.alignment == .horizontal {
            let verts = p.geometry.boundaryVertices
            guard !verts.isEmpty else { continue }
            let step = max(1, (verts.count + Config.maxPlaneVertices - 1) / Config.maxPlaneVertices)
            var poly: [[Double]] = []
            var i = 0
            while i < verts.count {
                poly.append([Self.rnd(Double(verts[i].x), 1000), Self.rnd(Double(verts[i].z), 1000)])
                i += step
            }
            planes.append(["id": p.identifier.uuidString,
                           "cls": Self.planeClass(p),
                           "T": Self.flatten(p.transform),
                           "poly": poly])
        }
        sendJSON(["type": "planes", "planes": planes])
    }

    private static func planeClass(_ p: ARPlaneAnchor) -> String {
        switch p.classification {
        case .floor: return "floor"
        case .table: return "table"
        case .seat: return "seat"
        case .ceiling: return "ceiling"
        default: return "none"
        }
    }

    private static func flatten(_ m: simd_float4x4) -> [Double] {
        let cols = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
        return cols.flatMap { [Double($0.x), Double($0.y), Double($0.z), Double($0.w)] }
    }

    private static func rnd(_ x: Double, _ scale: Double) -> Double {
        (x * scale).rounded() / scale
    }

    // MARK: - Orientation
    /// k from the camera-to-world rotation: phi = atan2(R[1,0], R[1,1]); 0 -> 0, -90 -> 1, +90 -> 3, 180 -> 2.
    private static func orientation(ax: Double, ay: Double, current: Int) -> Int {
        guard hypot(ax, ay) >= 0.5 else { return current }        // looking straight up/down: keep
        let phi = atan2(ax, ay) * 180 / Double.pi
        func center(_ k: Int) -> Double {
            switch k {
            case 1: return -90
            case 3: return 90
            case 2: return 180
            default: return 0
            }
        }
        func dist(_ a: Double, _ b: Double) -> Double {
            var d = abs(a - b).truncatingRemainder(dividingBy: 360)
            if d > 180 { d = 360 - d }
            return d
        }
        if dist(phi, center(current)) <= 55 { return current }   // hysteresis
        let candidates = [0, 1, 2, 3]
        return candidates.min(by: { dist(phi, center($0)) < dist(phi, center($1)) }) ?? current
    }

    /// EXIF orientation that rotates the native buffer upright for k (.right = 90 CW, .left = 90 CCW).
    private static func exif(_ k: Int) -> CGImagePropertyOrientation {
        switch k {
        case 1: return .right
        case 3: return .left
        case 2: return .down
        default: return .up
        }
    }

    // MARK: - Frame processing
    private func process(pixels: CVPixelBuffer, camera: ARCamera, depth: CVPixelBuffer?,
                         featurePoints: [simd_float3], lens: Lens, time: TimeInterval) {
        let t = camera.transform
        if Config.autoRotate {
            rotK = Self.orientation(ax: Double(t.columns.0.y), ay: Double(t.columns.1.y), current: rotK)
        } else {
            rotK = 0
        }
        let k = rotK
        let res = camera.imageResolution                          // e.g. 1920x1440, landscape
        let g = FrameGeom(resW: Double(res.width), resH: Double(res.height), crop: lens.crop,
                          outW: Double(Config.frameWidth), outH: Double(Config.frameHeight), k: k)

        // 1. JPEG: (crop) -> scale -> rotate upright. CIImage origin is bottom-left; the crop is centered so y0 is symmetric.
        var image = CIImage(cvPixelBuffer: pixels)
        if lens.crop {
            image = image.cropped(to: CGRect(x: g.x0, y: g.y0, width: g.cw, height: g.ch))
                .transformed(by: CGAffineTransform(translationX: CGFloat(-g.x0), y: CGFloat(-g.y0)))
        }
        image = image.transformed(by: CGAffineTransform(scaleX: CGFloat(g.sx), y: CGFloat(g.sy)))
        if k != 0 {
            image = image.oriented(Self.exif(k))
            let origin = image.extent.origin
            image = image.transformed(by: CGAffineTransform(translationX: -origin.x, y: -origin.y))
        }
        guard let cg = ciContext.createCGImage(image, from: CGRect(x: 0, y: 0, width: g.finalW, height: g.finalH)),
              let jpeg = UIImage(cgImage: cg).jpegData(compressionQuality: Config.jpegQuality) else {
            busy = false
            return
        }

        // 2. Hands (Apple Vision) on the full-res buffer, oriented upright; mapped into sent-image pixels (top-left origin).
        var hands: [[String: Any]] = []
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2
        try? VNImageRequestHandler(cvPixelBuffer: pixels, orientation: Self.exif(k)).perform([request])
        let uprightW = k % 2 == 1 ? g.resH : g.resW
        let uprightH = k % 2 == 1 ? g.resW : g.resH
        for hand in request.results ?? [] {
            guard let joints = try? hand.recognizedPoints(.all) else { continue }
            func sent(_ p: VNRecognizedPoint) -> [Double]? {
                guard p.confidence > 0.3 else { return nil }
                let native = g.nativeFromUpright(Double(p.location.x) * uprightW, (1 - Double(p.location.y)) * uprightH)
                guard let q = g.map(native.0, native.1) else { return nil }
                return [Self.rnd(q.0, 10), Self.rnd(q.1, 10)]
            }
            func px(_ joint: VNHumanHandPoseObservation.JointName) -> [Double]? {
                guard let p = joints[joint] else { return nil }
                return sent(p)
            }
            let visible = joints.values.compactMap { sent($0) }
            guard !visible.isEmpty else { continue }            // hand entirely outside the (cropped) image
            var h: [String: Any] = ["conf": Double(hand.confidence)]
            if let minX = visible.map({ $0[0] }).min(), let maxX = visible.map({ $0[0] }).max(),
               let minY = visible.map({ $0[1] }).min(), let maxY = visible.map({ $0[1] }).max() {
                h["bbox"] = [minX, minY, maxX - minX, maxY - minY]
            }
            switch hand.chirality {
            case .left: h["side"] = "left"
            case .right: h["side"] = "right"
            default: h["side"] = "unknown"
            }
            if let w = px(.wrist) { h["wrist"] = w }
            if let p = px(.thumbTip) { h["thumb"] = p }
            if let p = px(.indexTip) { h["index"] = p }
            if let p = px(.middleTip) { h["middle"] = p }
            if let p = px(.ringTip) { h["ring"] = p }
            if let p = px(.littleTip) { h["little"] = p }
            hands.append(h)
        }

        // 3. Pose (camera axes rotated to match the upright image) + intrinsics
        var cols = [t.columns.0, t.columns.1, t.columns.2, t.columns.3]
        switch k {
        case 1: cols = [t.columns.1, -t.columns.0, t.columns.2, t.columns.3]
        case 3: cols = [-t.columns.1, t.columns.0, t.columns.2, t.columns.3]
        case 2: cols = [-t.columns.0, -t.columns.1, t.columns.2, t.columns.3]
        default: break
        }
        let T: [Double] = cols.flatMap { [Double($0.x), Double($0.y), Double($0.z), Double($0.w)] }
        let K = camera.intrinsics
        let fx = Double(K[0][0]), fy = Double(K[1][1]), cx = Double(K[2][0]), cy = Double(K[2][1])
        let intr = g.intrinsics(fx: fx, fy: fy, cx: cx, cy: cy)

        // 4. ARKit feature points projected into the sent image: [u, v, z_cam]
        let inv = t.inverse
        var fp: [[Double]] = []
        for p in featurePoints {
            let c = inv * simd_float4(p.x, p.y, p.z, 1)
            let z = Double(-c.z)
            guard z > 0.25, z < 6 else { continue }
            let u = fx * Double(c.x) / z + cx
            let v = cy - fy * Double(c.y) / z
            guard let q = g.map(u, v) else { continue }
            fp.append([Self.rnd(q.0, 10), Self.rnd(q.1, 10), Self.rnd(z, 100)])
        }
        if fp.count > Config.maxFeaturePoints {
            let step = Double(fp.count) / Double(Config.maxFeaturePoints)
            fp = (0..<Config.maxFeaturePoints).map { fp[min(fp.count - 1, Int(Double($0) * step))] }
        }

        // 5. Optional LiDAR depth, resampled onto the sent image grid (float16)
        let dw = k % 2 == 1 ? Config.depthH : Config.depthW
        let dh = k % 2 == 1 ? Config.depthW : Config.depthH
        var depthData: Data? = nil
        if let d = depth { depthData = Self.downsample(depth: d, geom: g, toW: dw, toH: dh) }

        var tracking = false
        if case .normal = camera.trackingState { tracking = true }   // map + route planning only when ARKit is confident

        seq += 1
        let W = Int(g.finalW.rounded()), H = Int(g.finalH.rounded())
        var meta: [String: Any] = ["type": "frame", "seq": seq, "t": time, "tracking": tracking,
                                   "yaw": Double(camera.eulerAngles.y), "pitch": Double(camera.eulerAngles.x),
                                   "T": T, "intr": intr, "hands": hands,
                                   "W": W, "H": H, "width": W, "height": H,
                                   "camera": lens.label,
                                   "depth": depthData != nil, "dw": dw, "dh": dh,
                                   "fp": fp]
        if Config.autoRotate { meta["rot"] = k }
        sendFrame(metadata: meta, jpeg: jpeg, depth: depthData)

        sentCount += 1
        if Date().timeIntervalSince(sentSince) > 1 {
            let rate = Double(sentCount) / Date().timeIntervalSince(sentSince)
            let hc = hands.count, fc = fp.count
            let activeCamera = lens.label
            DispatchQueue.main.async {
                self.status = String(format: "Streaming %.0f fps · %@ · rot %d · hands %d · fp %d",
                                     rate, activeCamera, k, hc, fc)
            }
            sentCount = 0
            sentSince = Date()
        }
    }

    private func sendFrame(metadata: [String: Any], jpeg: Data, depth: Data?) {
        guard let metadataData = try? JSONSerialization.data(withJSONObject: metadata),
              let metadataText = String(data: metadataData, encoding: .utf8),
              let socket else {
            finishSendingFrame()
            return
        }

        socket.send(.string(metadataText)) { [weak self, weak socket] error in
            guard let self, error == nil, let socket else {
                self?.finishSendingFrame()
                return
            }
            socket.send(.data(jpeg)) { [weak self, weak socket] error in
                guard let self, error == nil, let socket else {
                    self?.finishSendingFrame()
                    return
                }
                guard let depth else {
                    self.finishSendingFrame()
                    return
                }
                socket.send(.data(depth)) { [weak self] _ in
                    self?.finishSendingFrame()
                }
            }
        }
    }

    private func finishSendingFrame() {
        work.async { [weak self] in
            self?.busy = false
        }
    }

    /// Samples the native depth map at each cell of a toW x toH grid laid over the sent image (crop + rotation applied).
    private static func downsample(depth: CVPixelBuffer, geom g: FrameGeom, toW: Int, toH: Int) -> Data {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        let rowBytes = CVPixelBufferGetBytesPerRow(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return Data() }
        var out = [Float16](repeating: 0, count: toW * toH)
        for y in 0..<toH {
            for x in 0..<toW {
                let X = (Double(x) + 0.5) * g.finalW / Double(toW)
                let Y = (Double(y) + 0.5) * g.finalH / Double(toH)
                let pre = g.unrotate(X, Y)
                let u = pre.0 / g.sx + g.x0, v = pre.1 / g.sy + g.y0
                let dx = min(w - 1, max(0, Int(u * Double(w) / g.resW)))
                let dy = min(h - 1, max(0, Int(v * Double(h) / g.resH)))
                let row = base.advanced(by: dy * rowBytes).assumingMemoryBound(to: Float32.self)
                out[y * toW + x] = Float16(row[dx])
            }
        }
        return out.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
