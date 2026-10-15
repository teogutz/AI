import SwiftUI
import Charts
import PeakPlanKit
#if canImport(WorkoutKit) && os(iOS)
import WorkoutKit
#endif

extension StepKind {
    var color: Color {
        switch self {
        case .warmup: .teal
        case .work: .orange
        case .recovery: .green
        case .rest: .gray
        case .cooldown: .blue
        }
    }
}

extension WorkoutStatus {
    var color: Color {
        switch self {
        case .planned: .secondary
        case .completed: .green
        case .skipped: .red
        }
    }
}

extension WorkoutDTO {
    /// True when the workout has intervals worth drawing (several steps at different intensities).
    /// Strength sessions or a single steady step would just draw a flat bar.
    func hasIntensityProfile(_ profile: AthleteProfile) -> Bool {
        guard sport != .strength else { return false }
        let segs = WorkoutMetrics.segments(self, profile: profile)
        return segs.count >= 2 && Set(segs.map { Int(($0.intensity * 20).rounded()) }).count >= 2
    }
}

/// Compact TrainingPeaks-style workout shape for lists: bar width = time, height = intensity,
/// colour = step type, dashed line = threshold. No axes, so it reads at a glance.
struct IntensityStrip: View {
    let workout: WorkoutDTO
    let profile: AthleteProfile
    private let ceiling = 1.2 // 120 % of threshold fills the height

    var body: some View {
        let segs = WorkoutMetrics.segments(workout, profile: profile)
        let total = max(segs.reduce(0) { $0 + $1.duration }, 1)
        GeometryReader { geo in
            let gap: CGFloat = segs.count > 40 ? 0 : 1
            let usable = geo.size.width - gap * CGFloat(max(segs.count - 1, 0))
            ZStack(alignment: .bottomLeading) {
                HStack(alignment: .bottom, spacing: gap) {
                    ForEach(segs) { s in
                        RoundedRectangle(cornerRadius: 2)
                            .fill(s.kind.color.opacity(s.isOpen ? 0.45 : 0.9))
                            .frame(width: max(usable * s.duration / total, 1),
                                   height: geo.size.height * min(max(s.intensity, 0.15), ceiling) / ceiling)
                    }
                }
                Path { p in
                    let y = geo.size.height * (1 - 1 / ceiling)
                    p.move(to: CGPoint(x: 0, y: y))
                    p.addLine(to: CGPoint(x: geo.size.width, y: y))
                }
                .stroke(Color.secondary.opacity(0.6), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
        }
        .accessibilityLabel("Workout intensity profile")
    }
}

/// Full intensity chart for the workout screen: x = minutes, y = % of threshold.
struct IntensityChart: View {
    let workout: WorkoutDTO
    let profile: AthleteProfile

    var body: some View {
        let segs = WorkoutMetrics.segments(workout, profile: profile)
        let top = max(120, ((segs.map(\.intensity).max() ?? 1) * 100 / 20).rounded(.up) * 20)
        Chart {
            ForEach(segs) { s in
                RectangleMark(
                    xStart: .value("Start", s.start / 60),
                    xEnd: .value("End", (s.start + s.duration) / 60),
                    yStart: .value("Base", 0),
                    yEnd: .value("Intensity", s.intensity * 100)
                )
                .foregroundStyle(s.kind.color.opacity(s.isOpen ? 0.4 : 0.9))
            }
            RuleMark(y: .value("Threshold", 100.0))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                .foregroundStyle(.secondary)
                .annotation(position: .top, alignment: .trailing) {
                    Text("threshold").font(.caption2).foregroundStyle(.secondary)
                }
        }
        .chartYScale(domain: 0...top)
        .chartYAxis {
            AxisMarks(position: .leading, values: [0.0, 50.0, 100.0]) { value in
                AxisGridLine()
                AxisValueLabel { if let v = value.as(Double.self) { Text("\(Int(v))%") } }
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { value in
                AxisGridLine()
                AxisValueLabel { if let v = value.as(Double.self) { Text("\(Int(v))′") } }
            }
        }
    }
}

struct WorkoutRow: View {
    @Environment(TrainingStore.self) private var store
    let workout: WorkoutDTO

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: workout.sport.symbol)
                .font(.title3)
                .frame(width: 32)
                .foregroundStyle(workout.status.color)
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(workout.title).font(.headline).lineLimit(1)
                    if workout.status == .completed { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                    if workout.status == .skipped { Image(systemName: "xmark.circle.fill").foregroundStyle(.red) }
                }
                HStack(spacing: 8) {
                    if let t = workout.startTime { Text(t) }
                    Text(Fmt.duration(workout.plannedDurationSec))
                    if let d = workout.plannedDistanceM { Text(Fmt.distance(d)) }
                    if let tss = workout.plannedTss { Text("\(Int(tss)) TSS") }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

struct WorkoutDetailView: View {
    @Environment(TrainingStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let workoutID: String
    @State private var editing = false
    @State private var showWatchPreview = false

    private var workout: WorkoutDTO? { store.workouts[workoutID] }

    var body: some View {
        if let w = workout {
            List {
                SwiftUI.Section {
                    HStack(spacing: 24) {
                        Metric(label: "Duration", value: Fmt.duration(w.plannedDurationSec))
                        if w.plannedDistanceM != nil { Metric(label: "Distance", value: Fmt.distance(w.plannedDistanceM)) }
                        Metric(label: "TSS", value: w.plannedTss.map { "\(Int($0))" } ?? "–")
                        Metric(label: "Status", value: w.status.rawValue.capitalized)
                    }
                    if w.hasIntensityProfile(store.profile) {
                        IntensityChart(workout: w, profile: store.profile).frame(height: 160)
                    }
                }

                if !w.description.isEmpty {
                    SwiftUI.Section("Description") { Text(w.description) }
                }
                if !w.coachNotes.isEmpty {
                    SwiftUI.Section("Coach notes") { Text(w.coachNotes) }
                }

                SwiftUI.Section("Structure") {
                    ForEach(w.steps) { element in
                        switch element {
                        case .step(let s):
                            StepLine(step: s)
                        case .repeatBlock(let r):
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Repeat \(r.count)×").font(.subheadline.bold())
                                ForEach(r.steps) { StepLine(step: $0).padding(.leading, 12) }
                            }
                        }
                    }
                }

                if let a = store.activity(id: w.completedActivityId) {
                    SwiftUI.Section("Completed") {
                        ActivitySummary(activity: a, planned: w)
                        NavigationLink("Compare with plan, interval by interval") {
                            ComparisonView(item: ReviewItemDTO(date: w.date ?? Day.today, state: .completed, workout: w, activity: a))
                        }
                    }
                } else if let date = w.date, date < Day.today, w.status == .planned {
                    SwiftUI.Section {
                        Label("Missed: no completed workout synced for this day", systemImage: "exclamationmark.circle")
                            .foregroundStyle(.red)
                    }
                }

                SwiftUI.Section {
                    #if os(iOS)
                    Button {
                        showWatchPreview = true
                    } label: {
                        Label("Preview & send to Apple Watch", systemImage: "applewatch")
                    }
                    #endif
                    if w.status != .skipped && w.status != .completed {
                        Button("Mark skipped", role: .destructive) {
                            var s = w; s.status = .skipped
                            Task { await store.save(s) }
                        }
                    }
                    Button("Delete workout", role: .destructive) {
                        Task { await store.delete(w); dismiss() }
                    }
                }
            }
            .navigationTitle(w.title)
            .task { await store.reload(workoutID) }
            .toolbar {
                Button("Edit") {
                    Task {
                        await store.reload(workoutID) // edit the latest version, not a stale copy
                        editing = true
                    }
                }
            }
            .sheet(isPresented: $editing) {
                NavigationStack { WorkoutEditorView(original: store.workouts[workoutID] ?? w) }
                    #if os(macOS)
                    .frame(minWidth: 640, minHeight: 640)
                    #endif
            }
            #if canImport(WorkoutKit) && os(iOS)
            .modifier(WatchPreviewModifier(workout: w, profile: store.profile, isPresented: $showWatchPreview))
            #endif
        } else {
            ContentUnavailableView("Workout deleted", systemImage: "trash")
        }
    }
}

#if canImport(WorkoutKit) && os(iOS)
/// Apple's built-in preview sheet: shows the intervals and lets the athlete send them to the Watch.
struct WatchPreviewModifier: ViewModifier {
    let workout: WorkoutDTO
    let profile: AthleteProfile
    @Binding var isPresented: Bool

    func body(content: Content) -> some View {
        if let plan = try? WorkoutKitBridge(profile: profile).plan(for: workout) {
            content.workoutPreview(plan, isPresented: $isPresented)
        } else {
            content.alert("This workout can't be sent to Apple Watch", isPresented: $isPresented) {
                Button("OK") {}
            } message: {
                Text("Structured Watch workouts support running, cycling, walking, hiking, swimming and rowing, with at least one step.")
            }
        }
    }
}
#endif

struct StepLine: View {
    let step: WorkoutStepDTO
    var body: some View {
        HStack {
            RoundedRectangle(cornerRadius: 2).fill(step.kind.color).frame(width: 4)
            VStack(alignment: .leading) {
                Text(Fmt.step(step))
                if let n = step.notes, !n.isEmpty { Text(n).font(.caption).foregroundStyle(.secondary) }
            }
        }
    }
}

struct Metric: View {
    let label: String
    let value: String
    var body: some View {
        VStack(alignment: .leading) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.headline.monospacedDigit())
        }
    }
}

struct ActivitySummary: View {
    let activity: ActivityDTO
    var planned: WorkoutDTO?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(activity.title).font(.headline)
            HStack(spacing: 20) {
                Metric(label: "Time", value: Fmt.duration(activity.durationSec))
                if let d = activity.distanceM { Metric(label: "Distance", value: Fmt.distance(d)) }
                if let hr = activity.avgHr { Metric(label: "Avg HR", value: "\(Int(hr))") }
                if let p = activity.avgPower { Metric(label: "Avg W", value: "\(Int(p))") }
                if let t = activity.tss { Metric(label: "TSS", value: "\(Int(t))") }
            }
            if let d = activity.decoupling { DecouplingRow(d: d) }
            if let planned, let pd = planned.plannedDurationSec, pd > 0 {
                let pct = Int((activity.durationSec / pd * 100).rounded())
                Text("Compliance: \(pct)% of planned duration").font(.caption).foregroundStyle(pct >= 80 && pct <= 120 ? Color.green : Color.orange)
            }
        }
    }
}
