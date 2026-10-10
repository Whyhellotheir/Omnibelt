import SwiftUI

struct ContentView: View {
    @StateObject private var streamer = BeltStreamer()
    @StateObject private var voice = VoiceIO()
    @State private var host = Config.defaultHost

    var body: some View {
        VStack(spacing: 18) {
            Text("NavBelt").font(.largeTitle).bold()
            TextField("Orin host or IP", text: $host)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button(streamer.running ? "Stop" : "Start") {
                if streamer.running {
                    streamer.stop(); voice.stop()
                } else {
                    streamer.start(host: host, voice: voice)
                    voice.start(streamer: streamer)
                }
            }
            .font(.title).buttonStyle(.borderedProminent)
            Text(streamer.status).font(.headline).multilineTextAlignment(.center)
            Text("Heard: \(voice.lastHeard)").font(.body)
            Text("Said: \(voice.lastSaid)").font(.body)
            HStack {
                Button("Test voice") { voice.say("Ready to go.") }
                Button("Send 'find my keys'") { streamer.sendCommand("find my keys") }
            }
            .buttonStyle(.bordered)
            Spacer()
        }
        .padding()
        .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
    }
}
