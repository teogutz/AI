import SwiftUI
import PeakPlanKit

/// Full workout builder, used on both iPhone and Mac.
struct WorkoutEditorView: View {
    @Environment(TrainingStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    @State private var w: WorkoutDTO
    @State private var editingStep: StepRef?

    init(original: WorkoutDTO) { _w = State(initialValue: original) }

    /// Identifies a step either at the top level or inside a repeat block.
    struct StepRef: Identifiable {
        let element: Int
        let inner: Int?
        var id: String { "\(element)-\(inner ?? -1)" }
    }

    var body: some View {
        Form {
            SwiftUI.Section {
                TextField("Title", text: $w.title)
                Picker("Sport", selection: $w.sport) {
                    ForEach(Sport.allCases) { Label($0.displayName, systemImage: $0.symbol).tag($0) }
                }
                Picker("Location", selection: $w.location) {
                    Text("Outdoor").tag(WorkoutLocation.outdoor)
                    Text("Indoor").tag(WorkoutLocation.indoor)
                }
                Toggle("Scheduled", isOn: scheduled)
                if w.date != nil {
                    DatePicker("Date", selection: dateBinding, displayedComponents: .date)
                    Toggle("Start time / Watch reminder", isOn: hasTime)
                    if w.startTime != nil {
                        DatePicker("Start", selection: timeBinding, displayedComponents: .hourAndMinute)
                    }
                }
                Picker("Status", selection: $w.status) {
                    ForEach(WorkoutStatus.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
                }
            }

            SwiftUI.Section("Description") {
                TextField("What is this workout for?", text: $w.description, axis: .vertical).lineLimit(2...6)
            }
            SwiftUI.Section("Coach notes") {
                TextField("Cues, fueling, terrain…", text: $w.coachNotes, axis: .vertical).lineLimit(2...6)
            }

            SwiftUI.Section {
                if !w.steps.isEmpty {
                    let est = WorkoutMetrics.estimate(w, profile: store.profile)
                    if w.hasIntensityProfile(store.profile) {
                        IntensityChart(workout: w, profile: store.profile).frame(height: 140)
                    }
                    Text("≈ \(Fmt.duration(est.seconds)) · \(Int(est.tss)) TSS")
                        .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(Array(w.steps.enumerated()), id: \.element.id) { index, element in
                    switch element {
                    case .step(let s):
                        Button { editingStep = StepRef(element: index, inner: nil) } label: { StepLine(step: s) }
                            .buttonStyle(.plain)
                    case .repeatBlock(let r):
                        VStack(alignment: .leading, spacing: 6) {
                            Stepper("Repeat \(r.count)×", value: repeatCount(index), in: 1...50)
                                .font(.subheadline.bold())
                            ForEach(Array(r.steps.enumerated()), id: \.element.id) { j, s in
                                HStack {
                                    Button { editingStep = StepRef(element: index, inner: j) } label: { StepLine(step: s) }
                                        .buttonStyle(.plain)
                                    Spacer()
                                    if r.steps.count > 1 {
                                        Button(role: .destructive) { removeInner(index, j) } label: { Image(systemName: "minus.circle") }
                                            .buttonStyle(.borderless)
                                    }
                                }
                                .padding(.leading, 12)
                            }
                            Button("Add step to repeat") { addInner(index) }
                                .buttonStyle(.borderless).font(.caption)
                        }
                    }
                }
                .onMove { w.steps.move(fromOffsets: $0, toOffset: $1) }
                .onDelete { w.steps.remove(atOffsets: $0) }

                Menu {
                    Button("Warm up") { add(.warmup, .time(seconds: 600), .heartRateZone(1)) }
                    Button("Work") { add(.work, .time(seconds: 300), nil) }
                    Button("Recovery") { add(.recovery, .time(seconds: 120), .heartRateZone(1)) }
                    Button("Cool down") { add(.cooldown, .time(seconds: 600), .heartRateZone(1)) }
                    Button("Repeat block (work / recovery)") {
                        w.steps.append(.repeatBlock(RepeatBlockDTO(count: 4, steps: [
                            WorkoutStepDTO(kind: .work, duration: .time(seconds: 240)),
                            WorkoutStepDTO(kind: .recovery, duration: .time(seconds: 120)),
                        ])))
                    }
                } label: {
                    Label("Add", systemImage: "plus.circle.fill")
                }
            } header: {
                Text("Steps")
            } footer: {
                Text("Tap a step to edit. Warm up first and cool down last appear as their own phases on Apple Watch.")
            }
        }
        .formStyle(.grouped)
        .navigationTitle(w.title)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") {
                    Task {
                        await store.save(w)
                        dismiss()
                    }
                }
                .disabled(w.title.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            #if os(iOS)
            ToolbarItem(placement: .topBarLeading) { EditButton() }
            #endif
        }
        .sheet(item: $editingStep) { ref in
            NavigationStack {
                StepEditorView(step: stepBinding(ref), sport: w.sport) {
                    delete(ref)
                    editingStep = nil
                }
            }
            #if os(macOS)
            .frame(minWidth: 420, minHeight: 460)
            #endif
        }
    }

    // MARK: Bindings & mutations

    private var scheduled: Binding<Bool> {
        Binding(get: { w.date != nil }, set: { w.date = $0 ? (w.date ?? Day.today) : nil })
    }

    private var dateBinding: Binding<Date> {
        Binding(get: { Day.date(w.date ?? Day.today) ?? Date() }, set: { w.date = Day.string($0) })
    }

    private var hasTime: Binding<Bool> {
        Binding(get: { w.startTime != nil }, set: { w.startTime = $0 ? (w.startTime ?? "06:00") : nil })
    }

    private var timeBinding: Binding<Date> {
        Binding(
            get: {
                let p = (w.startTime ?? "06:00").split(separator: ":").compactMap { Int($0) }
                return Calendar.current.date(bySettingHour: p.first ?? 6, minute: p.last ?? 0, second: 0, of: Date()) ?? Date()
            },
            set: {
                let c = Calendar.current.dateComponents([.hour, .minute], from: $0)
                w.startTime = String(format: "%02d:%02d", c.hour ?? 0, c.minute ?? 0)
            }
        )
    }

    private func repeatCount(_ index: Int) -> Binding<Int> {
        Binding(
            get: {
                if case .repeatBlock(let r) = w.steps[index] { return r.count }
                return 1
            },
            set: { n in
                if case .repeatBlock(var r) = w.steps[index] { r.count = n; w.steps[index] = .repeatBlock(r) }
            }
        )
    }

    private func stepBinding(_ ref: StepRef) -> Binding<WorkoutStepDTO> {
        Binding(
            get: {
                guard ref.element < w.steps.count else { return WorkoutStepDTO(kind: .work, duration: .open) }
                switch w.steps[ref.element] {
                case .step(let s): return s
                case .repeatBlock(let r): return r.steps[min(ref.inner ?? 0, r.steps.count - 1)]
                }
            },
            set: { new in
                guard ref.element < w.steps.count else { return }
                switch w.steps[ref.element] {
                case .step: w.steps[ref.element] = .step(new)
                case .repeatBlock(var r):
                    guard let j = ref.inner, j < r.steps.count else { return }
                    r.steps[j] = new
                    w.steps[ref.element] = .repeatBlock(r)
                }
            }
        )
    }

    private func add(_ kind: StepKind, _ duration: StepDuration, _ target: StepTarget?) {
        w.steps.append(.step(WorkoutStepDTO(kind: kind, duration: duration, target: target)))
    }

    private func addInner(_ index: Int) {
        if case .repeatBlock(var r) = w.steps[index] {
            r.steps.append(WorkoutStepDTO(kind: .recovery, duration: .time(seconds: 60)))
            w.steps[index] = .repeatBlock(r)
        }
    }

    private func removeInner(_ index: Int, _ j: Int) {
        if case .repeatBlock(var r) = w.steps[index] {
            r.steps.remove(at: j)
            w.steps[index] = .repeatBlock(r)
        }
    }

    private func delete(_ ref: StepRef) {
        if let j = ref.inner { removeInner(ref.element, j) } else { w.steps.remove(at: ref.element) }
    }
}

struct StepEditorView: View {
    @Binding var step: WorkoutStepDTO
    let sport: Sport
    let onDelete: () -> Void
    @Environment(\.dismiss) private var dismiss

    enum DurationType: String, CaseIterable { case time = "Time", distance = "Distance", open = "Open (lap button)" }
    enum TargetType: String, CaseIterable {
        case none = "None", power = "Power", hrZone = "HR zone", hr = "Heart rate", pace = "Pace", cadence = "Cadence"
    }

    var body: some View {
        Form {
            Picker("Type", selection: $step.kind) {
                ForEach(StepKind.allCases) { Text($0.displayName).tag($0) }
            }
            TextField("Name (optional)", text: Binding(get: { step.name ?? "" }, set: { step.name = $0.isEmpty ? nil : $0 }))

            SwiftUI.Section("Duration") {
                Picker("Ends after", selection: durationType) {
                    ForEach(DurationType.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                switch step.duration {
                case .time(let s):
                    HStack {
                        Stepper("Minutes: \(Int(s) / 60)", value: minutesBinding(s), in: 0...600)
                        Stepper("Sec: \(Int(s) % 60)", value: secondsBinding(s), in: 0...59, step: 5)
                    }
                case .distance(let m):
                    HStack {
                        Text("Meters")
                        TextField("Meters", value: Binding(get: { m }, set: { step.duration = .distance(meters: max(1, $0)) }), format: .number)
                            .multilineTextAlignment(.trailing)
                    }
                case .open:
                    Text("Press the lap / Action button on the Watch to move on.").font(.caption).foregroundStyle(.secondary)
                }
            }

            SwiftUI.Section("Target (Watch alerts when you drift out of range)") {
                Picker("Target", selection: targetType) {
                    ForEach(TargetType.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                targetFields
                switch step.target {
                case .pace, .power:
                    Picker("Watch alert uses", selection: $step.alertMetric) {
                        Text("Default").tag(AlertMetric?.none)
                        Text("Average for this step").tag(AlertMetric?.some(.average))
                        Text("Current value").tag(AlertMetric?.some(.current))
                    }
                default:
                    EmptyView()
                }
            }

            SwiftUI.Section("Notes") {
                TextField("Cue for this step", text: Binding(get: { step.notes ?? "" }, set: { step.notes = $0.isEmpty ? nil : $0 }), axis: .vertical)
            }

            Button("Delete step", role: .destructive) { onDelete(); dismiss() }
        }
        .formStyle(.grouped)
        .navigationTitle(step.kind.displayName)
        .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
    }

    // MARK: Duration

    private var durationType: Binding<DurationType> {
        Binding(
            get: {
                switch step.duration {
                case .time: .time
                case .distance: .distance
                case .open: .open
                }
            },
            set: {
                switch $0 {
                case .time: step.duration = .time(seconds: 300)
                case .distance: step.duration = .distance(meters: sport == .swim ? 100 : 1000)
                case .open: step.duration = .open
                }
            }
        )
    }

    private func minutesBinding(_ s: Double) -> Binding<Int> {
        Binding(get: { Int(s) / 60 }, set: { step.duration = .time(seconds: max(5, Double($0 * 60 + Int(s) % 60))) })
    }

    private func secondsBinding(_ s: Double) -> Binding<Int> {
        Binding(get: { Int(s) % 60 }, set: { step.duration = .time(seconds: max(5, Double(Int(s) / 60 * 60 + $0))) })
    }

    // MARK: Target

    private var targetType: Binding<TargetType> {
        Binding(
            get: {
                switch step.target {
                case nil: .none
                case .power: .power
                case .heartRateZone: .hrZone
                case .heartRate: .hr
                case .pace: .pace
                case .cadence: .cadence
                }
            },
            set: {
                switch $0 {
                case .none: step.target = nil
                case .power: step.target = .power(unit: .percentFtp, min: 88, max: 94)
                case .hrZone: step.target = .heartRateZone(2)
                case .hr: step.target = .heartRate(unit: .bpm, min: 140, max: 150)
                case .pace:
                    step.target = sport == .swim
                        ? .pace(unit: .secPer100m, min: 100, max: 105)
                        : .pace(unit: .secPerKm, min: 300, max: 315)
                case .cadence: step.target = .cadence(min: 85, max: 95)
                }
            }
        )
    }

    @ViewBuilder
    private var targetFields: some View {
        switch step.target {
        case .power(let unit, let lo, let hi):
            Picker("Unit", selection: Binding(get: { unit }, set: { step.target = .power(unit: $0, min: lo, max: hi) })) {
                Text("% FTP").tag(PowerUnit.percentFtp)
                Text("Watts").tag(PowerUnit.watts)
            }
            RangeFields(lo: lo, hi: hi) { step.target = .power(unit: unit, min: $0, max: $1) }
        case .heartRateZone(let z):
            Stepper("Zone \(z)", value: Binding(get: { z }, set: { step.target = .heartRateZone($0) }), in: 1...5)
        case .heartRate(let unit, let lo, let hi):
            Picker("Unit", selection: Binding(get: { unit }, set: { step.target = .heartRate(unit: $0, min: lo, max: hi) })) {
                Text("bpm").tag(HeartRateUnit.bpm)
                Text("% LTHR").tag(HeartRateUnit.percentLthr)
            }
            RangeFields(lo: lo, hi: hi) { step.target = .heartRate(unit: unit, min: $0, max: $1) }
        case .pace(let unit, let lo, let hi):
            Picker("Unit", selection: Binding(get: { unit }, set: { step.target = .pace(unit: $0, min: lo, max: hi) })) {
                Text("min/km").tag(PaceUnit.secPerKm)
                Text("min/100m").tag(PaceUnit.secPer100m)
                Text("% threshold").tag(PaceUnit.percentThreshold)
            }
            if unit == .percentThreshold {
                RangeFields(lo: lo, hi: hi) { step.target = .pace(unit: unit, min: $0, max: $1) }
            } else {
                PaceField(label: "Fastest", seconds: lo) { step.target = .pace(unit: unit, min: $0, max: hi) }
                PaceField(label: "Slowest", seconds: hi) { step.target = .pace(unit: unit, min: lo, max: $0) }
            }
        case .cadence(let lo, let hi):
            RangeFields(lo: lo, hi: hi) { step.target = .cadence(min: $0, max: $1) }
        case nil:
            EmptyView()
        }
    }
}

struct RangeFields: View {
    let lo: Double
    let hi: Double
    let set: (Double, Double) -> Void

    var body: some View {
        HStack {
            Text("From")
            TextField("Min", value: Binding(get: { lo }, set: { set($0, hi) }), format: .number)
            Text("to")
            TextField("Max", value: Binding(get: { hi }, set: { set(lo, $0) }), format: .number)
        }
        .multilineTextAlignment(.trailing)
    }
}

struct PaceField: View {
    let label: String
    let seconds: Double
    let set: (Double) -> Void

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Stepper(Fmt.pace(seconds), value: Binding(get: { seconds }, set: { set(max(30, $0)) }), in: 30...1200, step: 5)
                .monospacedDigit()
        }
    }
}
