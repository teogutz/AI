import SwiftUI
import PeakPlanKit

struct SettingsView: View {
    @Environment(TrainingStore.self) private var store
    @State private var url = ""
    @State private var token = ""
    @State private var testing = false
    @State private var testResult: String?
    #if os(iOS)
    @State private var healthSuggestions: [HealthImporter.ProfileSuggestion]?
    @State private var loadingHealth = false
    #endif

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
                SwiftUI.Section {
                    NavigationLink { ServerStatusView() } label: {
                        Label("Server status", systemImage: "server.rack")
                    }
                }

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

                if !store.thresholdWarnings.isEmpty {
                    SwiftUI.Section {
                        ForEach(store.thresholdWarnings) { ThresholdWarningRow(t: $0) }
                    } header: {
                        Text("Thresholds look out of date")
                    } footer: {
                        Text("Stale thresholds distort every % target and TSS. Estimated from your fastest sustained efforts of the last 6 weeks (super-shoe runs discounted ~3 %).")
                    }
                }

                SwiftUI.Section {
                    ForEach($store.profile.shoes) { $shoe in
                        HStack {
                            TextField("Name", text: $shoe.name)
                            Toggle("Super", isOn: $shoe.superShoe).fixedSize()
                        }
                        .swipeActions {
                            Button(shoe.retired ? "Unretire" : "Retire") { shoe.retired.toggle() }.tint(.gray)
                        }
                        .opacity(shoe.retired ? 0.5 : 1)
                    }
                    .onDelete { store.profile.shoes.remove(atOffsets: $0) }
                    Button("Add shoe") { store.profile.shoes.append(Shoe(name: "New shoe")) }
                    if !store.profile.shoes.isEmpty {
                        Picker("Default for runs", selection: $store.profile.defaultShoeId) {
                            Text("None").tag(String?.none)
                            ForEach(store.profile.shoes.filter { !$0.retired }) { Text($0.name).tag(String?.some($0.id)) }
                        }
                    }
                    Button("Save shoes") { Task { await store.saveProfile() } }
                } header: {
                    Text("Shoes")
                } footer: {
                    Text("Mark carbon-plated / super-foam shoes as Super: they're worth 2–4 % on pace, so comparisons and threshold estimates account for them. Set the shoe on each run in Review.")
                }

                SwiftUI.Section("Thresholds") {
                    #if os(iOS)
                    Button {
                        Task { await loadHealthSuggestions() }
                    } label: {
                        Label(loadingHealth ? "Reading Apple Health…" : "Fill from Apple Health", systemImage: "heart.text.square")
                    }
                    .disabled(loadingHealth)
                    #endif
                    NumberRow("FTP (W)", value: $store.profile.ftpWatts)
                    PaceRow(label: "Run threshold pace /km", seconds: $store.profile.thresholdPaceSecPerKm)
                    PaceRow(label: "Swim CSS /100m", seconds: $store.profile.swimCssSecPer100m)
                    NumberRow("Threshold HR (LTHR)", value: $store.profile.lthrBpm)
                    NumberRow("Max HR", value: $store.profile.maxHrBpm)
                    NumberRow("Resting HR", value: $store.profile.restingHrBpm)
                    NumberRow("Weight (kg)", value: $store.profile.weightKg)
                    NumberRow("VO2max (Cardio Fitness)", value: $store.profile.vo2Max)
                    NumberRow("Hours / week available", value: $store.profile.weeklyHoursAvailable)
                    Picker("Pace alerts on work intervals", selection: $store.profile.workPaceAlertMetric) {
                        Text("Average for the rep").tag(AlertMetric.average)
                        Text("Current pace").tag(AlertMetric.current)
                    }
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
        #if os(iOS)
        .sheet(isPresented: Binding(get: { healthSuggestions != nil }, set: { if !$0 { healthSuggestions = nil } })) {
            NavigationStack { HealthProfileSheet(suggestions: healthSuggestions ?? []) }
        }
        #endif
    }

    #if os(iOS)
    private func loadHealthSuggestions() async {
        loadingHealth = true
        defer { loadingHealth = false }
        healthSuggestions = await HealthImporter().profileSuggestions()
    }
    #endif

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
            #if os(iOS)
            // First setup: offer what Apple Health knows instead of typing it in.
            let p = store.profile
            if p.ftpWatts == nil && p.lthrBpm == nil && p.thresholdPaceSecPerKm == nil {
                await loadHealthSuggestions()
            }
            #endif
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

#if os(iOS)
/// Shows what was found in Apple Health; the athlete picks which values to copy into the profile.
struct HealthProfileSheet: View {
    @Environment(TrainingStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let suggestions: [HealthImporter.ProfileSuggestion]
    @State private var selected: Set<String> = []

    var body: some View {
        List {
            if suggestions.isEmpty {
                ContentUnavailableView("Nothing found", systemImage: "heart.slash",
                    description: Text("Apple Health has no usable data yet, or access was denied (Settings → Health → Data Access & Devices → PeakPlan)."))
            } else {
                SwiftUI.Section {
                    ForEach(suggestions) { s in
                        Toggle(isOn: Binding(
                            get: { selected.contains(s.id) },
                            set: { if $0 { selected.insert(s.id) } else { selected.remove(s.id) } }
                        )) {
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(Self.label(s.field)).font(.headline)
                                    if s.isEstimate {
                                        Text("Estimate").font(.caption2.bold())
                                            .padding(.horizontal, 6).padding(.vertical, 2)
                                            .background(Color.orange.opacity(0.2), in: Capsule())
                                    }
                                }
                                Text(Self.format(s.field, s.value)).font(.title3.monospacedDigit())
                                if let current = current(s.field) {
                                    Text("Currently \(Self.format(s.field, current))").font(.caption).foregroundStyle(.secondary)
                                }
                                Text(s.source).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                } footer: {
                    Text("Estimates are a starting point. Claude can refine them from your workouts, or do a 30-minute threshold test.")
                }
            }
        }
        .navigationTitle("From Apple Health")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            ToolbarItem(placement: .confirmationAction) {
                Button("Apply") {
                    apply()
                    Task { await store.saveProfile() }
                    dismiss()
                }
                .disabled(selected.isEmpty)
            }
        }
        .onAppear {
            // Pre-select values for fields that are still empty.
            selected = Set(suggestions.filter { current($0.field) == nil }.map(\.id))
        }
    }

    private func current(_ f: HealthImporter.ProfileSuggestion.Field) -> Double? {
        let p = store.profile
        switch f {
        case .ftp: return p.ftpWatts
        case .thresholdPace: return p.thresholdPaceSecPerKm
        case .lthr: return p.lthrBpm
        case .maxHr: return p.maxHrBpm
        case .restingHr: return p.restingHrBpm
        case .weight: return p.weightKg
        case .vo2Max: return p.vo2Max
        }
    }

    private func apply() {
        for s in suggestions where selected.contains(s.id) {
            switch s.field {
            case .ftp: store.profile.ftpWatts = s.value
            case .thresholdPace: store.profile.thresholdPaceSecPerKm = s.value
            case .lthr: store.profile.lthrBpm = s.value
            case .maxHr: store.profile.maxHrBpm = s.value
            case .restingHr: store.profile.restingHrBpm = s.value
            case .weight: store.profile.weightKg = s.value
            case .vo2Max: store.profile.vo2Max = s.value
            }
        }
    }

    static func label(_ f: HealthImporter.ProfileSuggestion.Field) -> String {
        switch f {
        case .ftp: "FTP"
        case .thresholdPace: "Run threshold pace"
        case .lthr: "Threshold heart rate"
        case .maxHr: "Max heart rate"
        case .restingHr: "Resting heart rate"
        case .weight: "Weight"
        case .vo2Max: "VO2max"
        }
    }

    static func format(_ f: HealthImporter.ProfileSuggestion.Field, _ v: Double) -> String {
        switch f {
        case .ftp: "\(Int(v)) W"
        case .thresholdPace: "\(Fmt.pace(v)) /km"
        case .lthr, .maxHr, .restingHr: "\(Int(v)) bpm"
        case .weight: String(format: "%.1f kg", v)
        case .vo2Max: String(format: "%.1f", v)
        }
    }
}
#endif

struct ThresholdWarningRow: View {
    @Environment(TrainingStore.self) private var store
    let t: ThresholdSuggestionDTO
    @State private var confirming = false

    var body: some View {
        if let suggested = t.suggested {
            VStack(alignment: .leading, spacing: 6) {
                Text(title(suggested)).font(.headline)
                Text(detail).font(.caption).foregroundStyle(.secondary)
                if t.largeChange == true {
                    Label("Large change: double-check before applying.", systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
                Button("Update to \(value(suggested))") {
                    if t.largeChange == true { confirming = true } else { Task { await store.apply(t) } }
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.vertical, 4)
            .confirmationDialog("Change threshold by \(Int(t.diffPct ?? 0)) %?", isPresented: $confirming, titleVisibility: .visible) {
                Button("Update to \(value(suggested))") { Task { await store.apply(t) } }
            } message: {
                Text("A change this large usually comes from a race or a test. If you're not sure, do a 30-min threshold test first.")
            }
        }
    }

    private var isPace: Bool { t.field == "thresholdPaceSecPerKm" }
    private func value(_ v: Double) -> String { isPace ? Fmt.pace(v) + " min/km" : "\(Int(v)) W" }
    private func title(_ suggested: Double) -> String {
        (isPace ? "Run threshold pace: " : "FTP: ") + (t.current.map(value) ?? "not set") + " → " + value(suggested)
    }
    private var detail: String {
        guard let b = t.basis else { return "" }
        let effort = isPace ? (b.paceSecPerKm.map { Fmt.pace($0) + " min/km" } ?? "") : (b.watts.map { "\(Int($0)) W" } ?? "")
        let shoe = b.superShoe == true ? " in super shoes" : ""
        return "Based on \(Int(b.durationMin)) min at \(effort)\(shoe) — \(b.title), \(TodayView.pretty(b.date))."
    }
}
