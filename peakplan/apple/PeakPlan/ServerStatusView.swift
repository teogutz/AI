import SwiftUI
import PeakPlanKit

/// Health of the PeakPlan server: reachable, uptime, data, last sync from Apple Health, backups, disk.
struct ServerStatusView: View {
    @Environment(TrainingStore.self) private var store
    @State private var status: ServerStatusDTO?
    @State private var error: String?
    @State private var latencyMs: Int?
    @State private var loading = false

    var body: some View {
        List {
            SwiftUI.Section {
                HStack(spacing: 12) {
                    Circle().fill(status != nil ? Color.green : (error != nil ? Color.red : Color.gray))
                        .frame(width: 14, height: 14)
                    VStack(alignment: .leading) {
                        Text(status != nil ? "Online" : (error != nil ? "Unreachable" : "Checking…")).font(.headline)
                        Text(store.config?.baseURL.absoluteString ?? "").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let latencyMs { Text("\(latencyMs) ms").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                }
                if let error { Text(error).font(.caption).foregroundStyle(.red) }
            }

            if let s = status {
                SwiftUI.Section("Server") {
                    row("Running for", Self.duration(s.uptimeSec))
                    row("Version", "\(s.version) · Node \(s.node)")
                    row("CPU load", String(format: "%.2f", s.host.loadAvg1m))
                    row("Memory free", Self.bytes(s.host.memFreeBytes) + " of " + Self.bytes(s.host.memTotalBytes))
                    if let d = s.host.disk {
                        row("Disk free", Self.bytes(d.freeBytes) + " of " + Self.bytes(d.totalBytes),
                            warn: d.freeBytes / max(d.totalBytes, 1) < 0.1)
                    }
                }
                SwiftUI.Section("Data") {
                    row("Planned workouts", "\(s.database.workouts) (\(s.database.upcomingWorkouts) upcoming)")
                    row("Recorded activities", "\(s.database.activities)")
                    row("Last workout recorded", Self.ago(s.database.lastActivity))
                    row("Last import from Health", Self.ago(s.database.lastImport))
                    row("Last plan change", Self.ago(s.database.lastWorkoutChange))
                    row("Database size", Self.bytes(s.database.sizeBytes))
                }
                SwiftUI.Section {
                    if let b = s.backups.latest {
                        let age = Self.date(b.modified).map { Date().timeIntervalSince($0) } ?? .infinity
                        row("Latest backup", Self.ago(b.modified), warn: age > 2 * 86_400)
                        row("Backups kept", "\(s.backups.count)")
                    } else {
                        row("Backups", "None yet (the first runs within a day)", warn: true)
                    }
                } header: {
                    Text("Backups")
                } footer: {
                    Text("Daily copies are kept on the server for 14 days. Run deploy/gcp/backup.sh on your Mac to download one.")
                }
            }
        }
        .navigationTitle("Server status")
        .refreshable { await load() }
        .task { await load() }
        .toolbar {
            Button { Task { await load() } } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading)
        }
    }

    private func load() async {
        guard let api = store.client else { return }
        loading = true
        defer { loading = false }
        let start = Date()
        do {
            status = try await api.status()
            latencyMs = Int(Date().timeIntervalSince(start) * 1000)
            error = nil
        } catch {
            status = nil
            latencyMs = nil
            self.error = error.localizedDescription
        }
    }

    private func row(_ label: String, _ value: String, warn: Bool = false) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(warn ? Color.orange : Color.secondary).multilineTextAlignment(.trailing)
        }
    }

    static func bytes(_ b: Double) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file)
    }

    static func duration(_ sec: Double) -> String {
        let s = Int(sec)
        if s >= 86_400 { return "\(s / 86_400) d \(s % 86_400 / 3600) h" }
        if s >= 3600 { return "\(s / 3600) h \(s % 3600 / 60) min" }
        return "\(s / 60) min"
    }

    static func date(_ iso: String?) -> Date? {
        guard let iso else { return nil }
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: iso) { return d }
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: iso)
    }

    static func ago(_ iso: String?) -> String {
        guard let d = date(iso) else { return "–" }
        return d.formatted(.relative(presentation: .named))
    }
}
