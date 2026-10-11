import SwiftUI

struct ContentView: View {
    @StateObject private var streamer = BeltStreamer()
    @StateObject private var voice = VoiceIO()
    @State private var host = Config.defaultHost

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                Text("NavBelt").font(.largeTitle).bold()
                TextField("Orin host or IP", text: $host)
                    .textFieldStyle(.roundedBorder)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .disabled(streamer.running)
                Button(streamer.running ? "Stop" : "Start") {
                    if streamer.running {
                        streamer.stop(); voice.stop()
                    } else {
                        streamer.start(host: host, voice: voice)
                        voice.start(streamer: streamer, host: host)
                    }
                }
                .font(.title).buttonStyle(.borderedProminent)
                VStack(alignment: .leading, spacing: 6) {
                    Text("Camera").font(.subheadline).foregroundStyle(.secondary)
                    Picker("Camera", selection: Binding(
                        get: { streamer.profile },
                        set: { streamer.setProfile($0) }
                    )) {
                        ForEach(CameraProfile.allCases) { p in
                            Text(p.label).tag(p)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                Text(streamer.status).font(.headline).multilineTextAlignment(.center)
                if !streamer.formatsInfo.isEmpty {
                    DisclosureGroup("ARKit video formats") {
                        Text(streamer.formatsInfo)
                            .font(.caption2.monospaced())
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.caption)
                }
                if streamer.running {
                    OrinCameraView(host: host)
                }
                Text("Heard: \(voice.lastHeard)").font(.body)
                Text("Said: \(voice.lastSaid)").font(.body)
                HStack {
                    Button("Test voice") { voice.say("Ready to go.") }
                    Button("Send 'find my keys'") { streamer.sendCommand("find my keys") }
                }
                .buttonStyle(.bordered)
            }
            .padding()
        }
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
    }
}
