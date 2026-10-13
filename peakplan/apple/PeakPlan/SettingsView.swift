import SwiftUI
import PeakPlanKit

struct SettingsView: View {
    @Environment(TrainingStore.self) private var store
    @State private var url = ""
    @State private var token = ""
    @State private var testing = false
    @State private var testResult: String?

    var body: some View {
        @Bindable var store = store
        Form {
            SwiftUI.Section {
                TextField("Server URL (https://…)", text: $url)
                    #if os(iOS)
                    .textContentType(.URL)
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    #endif
                    .autocorrectionDisabled()
                SecureField("Access token", text: $token)
                Button(testing ? "Connecting…" : "Connect") { Task { await connect() } }
                    .disabled(testing || URL(string: url) == nil || token.count < 16)
                if let testResult { Text(testResult).font(.caption) }
            } header: {
                Text("PeakPlan server")
            } footer: {
                Text("Your calendar lives on your PeakPlan server, which is also what Claude connects to. See README for setup.")
            }

            if let config = store.config {
                SwiftUI.Section("Connect Claude") {
                    let mcp = config.baseURL.appending(path: "mcp").appending(path: config.token).absoluteString
                    Text("In Claude → Settings → Connectors → Add custom connector, paste:").font(.caption)
                    Text(mcp).font(.caption.monospaced()).textSelection(.enabled)
                    #if os(iOS)
                    Button("Copy connector URL") { UIPasteboard.general.string = mcp }
                    #else
                    Button("Copy connector URL") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(mcp, forType: .string)
                    }
                    #endif
                }

                SwiftUI.Section("Thresholds") {
                    NumberRow("FTP (W)", value: $store.profile.ftpWatts)
                    PaceRow(label: "Run threshold pace /km", seconds: $store.profile.thresholdPaceSecPerKm)
                    PaceRow(label: "Swim CSS /100m", seconds: $store.profile.swimCssSecPer100m)
                    NumberRow("Threshold HR (LTHR)", value: $store.profile.lthrBpm)
                    NumberRow("Max HR", value: $store.profile.maxHrBpm)
                    NumberRow("Resting HR", value: $store.profile.restingHrBpm)
                    NumberRow("Weight (kg)", value: $store.profile.weightKg)
                    NumberRow("Hours / week available", value: $store.profile.weeklyHoursAvailable)
                    TextField("Goals & target events (Claude reads this)", text: $store.profile.goals, axis: .vertical)
                    Button("Save profile") { Task { await store.saveProfile() } }
                }

                #if os(iOS)
                SwiftUI.Section {
                    Button("Sync now (Health → server → Watch)") { Task { await SyncCoordinator.fullSync(store) } }
                    if let last = store.lastSync {
                        Text("Last sync \(last.formatted(.relative(presentation: .named)))").font(.caption).foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Apple Watch & Health")
                } footer: {
                    Text("The next 14 days of planned workouts are scheduled in the Watch's Workout app (max 15). Workouts without a start time remind you at \(WatchScheduleSync.defaultHour):00.")
                }
                #endif

                SwiftUI.Section {
                    Button("Disconnect", role: .destructive) { store.config = nil }
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .onAppear {
            url = store.config?.baseURL.absoluteString ?? url
            token = store.config?.token ?? token
        }
    }

    private func connect() async {
        guard let base = URL(string: url.trimmingCharacters(in: .whitespaces)) else { return }
        testing = true
        defer { testing = false }
        let config = ServerConfig(baseURL: base, token: token.trimmingCharacters(in: .whitespaces))
        do {
            _ = try await APIClient(config: config).profile()
            store.config = config
            testResult = "Connected ✓"
            await SyncCoordinator.fullSync(store)
        } catch {
            testResult = "Failed: \(error.localizedDescription)"
        }
    }
}

struct NumberRow: View {
    let label: String
    @Binding var value: Double?
    init(_ label: String, value: Binding<Double?>) {
        self.label = label
        _value = value
    }

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            TextField("–", value: $value, format: .number)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 100)
                #if os(iOS)
                .keyboardType(.decimalPad)
                #endif
        }
    }
}

/// Edits a pace stored as seconds, shown as m:ss.
struct PaceRow: View {
    let label: String
    @Binding var seconds: Double?

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            TextField("m:ss", text: Binding(
                get: { seconds.map(Fmt.pace) ?? "" },
                set: { text in
                    let p = text.split(separator: ":").compactMap { Double($0) }
                    if p.count == 2 { seconds = p[0] * 60 + p[1] } else if text.isEmpty { seconds = nil }
                }
            ))
            .multilineTextAlignment(.trailing)
            .frame(maxWidth: 100)
        }
    }
}
