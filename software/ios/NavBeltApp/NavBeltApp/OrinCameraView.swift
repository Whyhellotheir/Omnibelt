import SwiftUI
import WebKit

struct OrinCameraView: View {
    let host: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Orin camera", systemImage: "video.fill")
                .font(.headline)

            if let streamURL = URL(string: "http://\(host):\(Config.port)/video/camera") {
                MJPEGWebView(url: streamURL)
                    .aspectRatio(4.0 / 3.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay {
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(.secondary.opacity(0.35))
                    }
                    .accessibilityLabel("Live camera view from the Orin")
            } else {
                ContentUnavailableView(
                    "Camera Unavailable",
                    systemImage: "video.slash",
                    description: Text("Enter a valid Orin host or IP address.")
                )
            }
        }
    }
}

private struct MJPEGWebView: UIViewRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.isOpaque = false
        webView.backgroundColor = .black
        webView.scrollView.backgroundColor = .black
        webView.scrollView.isScrollEnabled = false
        load(url, in: webView, coordinator: context.coordinator)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.loadedURL != url else { return }
        load(url, in: webView, coordinator: context.coordinator)
    }

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.stopLoading()
        coordinator.loadedURL = nil
    }

    private func load(_ url: URL, in webView: WKWebView, coordinator: Coordinator) {
        coordinator.loadedURL = url
        webView.load(URLRequest(url: url))
    }

    final class Coordinator {
        var loadedURL: URL?
    }
}
