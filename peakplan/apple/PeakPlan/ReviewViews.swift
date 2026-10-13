import SwiftUI
import Charts
import PeakPlanKit

extension ComplianceColor {
    var color: Color {
        switch self {
        case .green: .green
        case .yellow: .yellow
        case .orange: .orange
        case .red: .red
        }
    }
}

/// Completed workouts against the plan, TrainingPeaks style: every day's planned workout is
/// coloured by how closely it was followed, with missed and unplanned sessions called out.
struct ReviewView: View {
    @Environment(TrainingStore.self) private var store
    @State private var items: [ReviewItemDTO] = []
    @State private var weeks = 4
    @State private var loading = false

    private var grouped: [(week: String, items: [ReviewItemDTO])] {
        let byWeek = Dictionary(grouping: items) { Self.weekStart($0.date) }
        return byWeek.keys.sorted(by: >).map { ($0, byWeek[$0]!.sorted { $0.date > $1.date }) }
    }

    var body: some View {
        List {
            Picker("Period", selection: $weeks) {
                Text("1 week").tag(1)
                Text("4 weeks").tag(4)
                Text("12 weeks").tag(12)
            }
            .pickerStyle(.segmented)

            if items.isEmpty && !loading {
                ContentUnavailableView("Nothing to review yet", systemImage: "checkmark.seal",
                                       description: Text("Completed workouts appear here after they sync from Apple Health."))
            }
            ForEach(grouped, id: \.week) { group in
                SwiftUI.Section {
                    ForEach(group.items) { item in
                        NavigationLink(value: item) { ReviewRow(item: item) }
                    }
                } header: {
                    WeekComplianceHeader(week: group.week, items: group.items)
                }
            }
        }
        .navigationTitle("Review")
        .navigationDestination(for: ReviewItemDTO.self) { ComparisonView(item: $0) }
        .navigationDestination(for: String.self) { WorkoutDetailView(workoutID: $0) }
        .overlay { if loading && items.isEmpty { ProgressView() } }
        .task(id: weeks) { await load() }
        .refreshable {
            await SyncCoordinator.fullSync(store)
            await load()
        }
    }

    private func load() async {
        guard let api = store.client else { return }
        loading = true
        defer { loading = false }
        do {
            items = try await api.review(from: Day.adding(-7 * weeks + 1, to: Day.today), to: Day.today)
        } catch {
            store.lastError = error.localizedDescription
        }
    }

    static func weekStart(_ day: String) -> String {
        let cal = Calendar(identifier: .iso8601)
        guard let d = Day.date(day), let start = cal.dateInterval(of: .weekOfYear, for: d)?.start else { return day }
        return Day.string(start)
    }
}

struct WeekComplianceHeader: View {
    let week: String
    let items: [ReviewItemDTO]

    var body: some View {
        let planned = items.filter { $0.workout != nil }
        let done = planned.filter { $0.state == .completed }
        let plannedH = planned.compactMap { $0.workout?.plannedDurationSec }.reduce(0, +) / 3600
        let doneH = items.compactMap { $0.activity?.durationSec }.reduce(0, +) / 3600
        HStack {
            Text("Week of \(TodayView.pretty(week))")
            Spacer()
            Text(String(format: "%ld/%ld done · %.1f of %.1f h", done.count, planned.count, doneH, plannedH))
                .monospacedDigit()
        }
    }
}

struct ReviewRow: View {
    let item: ReviewItemDTO

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(item.color?.color ?? Color.gray)
                .frame(width: 12, height: 12)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.workout?.title ?? item.activity?.title ?? "Workout").font(.headline).lineLimit(1)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: (item.workout?.sport ?? item.activity?.sport ?? .other).symbol).foregroundStyle(.secondary)
        }
    }

    private var subtitle: String {
        let day = Day.date(item.date)?.formatted(.dateTime.weekday(.abbreviated).day().month()) ?? item.date
        switch item.state {
        case .missed: return "\(day) · Missed"
        case .skipped: return "\(day) · Skipped"
        case .unplanned: return "\(day) · Unplanned · \(Fmt.duration(item.activity?.durationSec))"
        case .completed:
            let planned = Fmt.duration(item.workout?.plannedDurationSec)
            let actual = Fmt.duration(item.activity?.durationSec)
            let pct = item.durationPct.map { " (\(Int($0))%)" } ?? ""
            return "\(day) · \(actual) of \(planned)\(pct)"
        }
    }
}

/// Planned vs. completed for one session: totals, every interval against its target, and feedback.
struct ComparisonView: View {
    @Environment(TrainingStore.self) private var store
    let item: ReviewItemDTO
    @State private var comparison: ComparisonDTO?
    @State private var rpe: Double = 5
    @State private var hasRpe = false
    @State private var notes = ""
    @State private var linkTo: String?

    var body: some View {
        List {
            header

            if let c = comparison {
                SwiftUI.Section("Planned vs. completed") { totals(c) }
                if c.intervalsMatched {
                    SwiftUI.Section {
                        TargetChart(steps: c.steps)
                        ForEach(c.steps) { StepComparisonRow(step: $0) }
                    } header: {
                        Text("Intervals")
                    } footer: {
                        if let p = c.intervalsInRangePct { Text("\(Int(p))% of intervals on target") }
                    }
                } else if let a = item.activity, item.workout != nil {
                    SwiftUI.Section("Intervals") {
                        Text("The Watch recorded \(a.laps?.count ?? 0) segments for \(c.steps.count) planned steps, so intervals can't be matched one-to-one. Start the workout from PeakPlan or the Workout app's scheduled list to get interval data.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            } else if let a = item.activity {
                SwiftUI.Section("Completed") { ActivitySummary(activity: a) }
            }

            if let w = item.workout, item.state == .missed || item.state == .skipped {
                SwiftUI.Section {
                    NavigationLink("Open planned workout", value: w.id)
                    if item.state == .missed {
                        Button("Mark as skipped") { var s = w; s.status = .skipped; Task { await store.save(s) } }
                    }
                    Button("Reschedule to today") { Task { await store.move(w, to: Day.today) } }
                }
            }

            if let a = item.activity {
                if item.state == .unplanned {
                    let candidates = store.workouts(on: item.date).filter { $0.completedActivityId == nil }
                    if !candidates.isEmpty {
                        SwiftUI.Section("Link to planned workout") {
                            Picker("Planned workout", selection: $linkTo) {
                                Text("None").tag(String?.none)
                                ForEach(candidates) { Text($0.title).tag(String?.some($0.id)) }
                            }
                            Button("Link") {
                                Task { await store.updateActivity(a, rpe: hasRpe ? rpe : nil, notes: notes, workoutId: linkTo) }
                            }
                            .disabled(linkTo == nil)
                        }
                    }
                }
                SwiftUI.Section("How did it feel?") {
                    Toggle("Rate effort (RPE)", isOn: $hasRpe)
                    if hasRpe {
                        Slider(value: $rpe, in: 1...10, step: 1) { Text("RPE") } minimumValueLabel: { Text("1") } maximumValueLabel: { Text("10") }
                        Text("RPE \(Int(rpe)) – \(Self.rpeLabel(Int(rpe)))").font(.caption)
                    }
                    TextField("Notes for your coach (Claude reads these)", text: $notes, axis: .vertical).lineLimit(2...6)
                    Button("Save feedback") {
                        Task { await store.updateActivity(a, rpe: hasRpe ? rpe : nil, notes: notes, workoutId: a.workoutId) }
                    }
                }
            }
        }
        .navigationTitle(item.workout?.title ?? item.activity?.title ?? "Review")
        .task {
            if let r = item.activity?.rpe { rpe = r; hasRpe = true }
            notes = item.activity?.notes ?? ""
            guard let w = item.workout, let api = store.client else { return }
            comparison = try? await api.comparison(workoutId: w.id)
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Circle().fill((comparison?.color ?? item.color)?.color ?? Color.gray).frame(width: 16, height: 16)
            VStack(alignment: .leading) {
                Text(item.state.rawValue.capitalized).font(.headline)
                Text(TodayView.pretty(item.date)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let w = item.workout, !w.steps.isEmpty {
                IntensityChart(workout: w, profile: store.profile).frame(width: 140, height: 50)
                    .chartXAxis(.hidden).chartYAxis(.hidden)
            }
        }
    }

    @ViewBuilder
    private func totals(_ c: ComparisonDTO) -> some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
            GridRow {
                Text("").gridColumnAlignment(.leading)
                Text("Planned").font(.caption.bold())
                Text("Completed").font(.caption.bold())
                Text("%").font(.caption.bold())
            }
            Divider()
            row("Duration", Fmt.duration(c.totals.planned.durationSec), Fmt.duration(c.totals.actual?.durationSec), c.totals.durationPct)
            if c.totals.planned.distanceM != nil || c.totals.actual?.distanceM != nil {
                row("Distance", Fmt.distance(c.totals.planned.distanceM), Fmt.distance(c.totals.actual?.distanceM), c.totals.distancePct)
            }
            row("TSS", c.totals.planned.tss.map { "\(Int($0))" } ?? "–", c.totals.actual?.tss.map { "\(Int($0))" } ?? "–", c.totals.tssPct)
            if let hr = c.totals.actual?.avgHr { row("Avg HR", "", "\(Int(hr)) bpm", nil) }
            if let p = c.totals.actual?.avgPower { row("Avg power", "", "\(Int(p)) W", nil) }
        }
        .monospacedDigit()
    }

    private func row(_ label: String, _ planned: String, _ actual: String, _ pct: Double?) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(planned)
            Text(actual).bold()
            Text(pct.map { "\(Int($0))%" } ?? "")
                .foregroundStyle(pct.map { ($0 >= 80 && $0 <= 120) ? Color.green : Color.orange } ?? Color.secondary)
        }
    }

    static func rpeLabel(_ v: Int) -> String {
        switch v {
        case ...2: "Very easy"
        case 3...4: "Easy"
        case 5...6: "Moderate"
        case 7...8: "Hard"
        default: "Max effort"
        }
    }
}

extension ComparisonDTO.Target {
    func format(_ v: Double) -> String {
        switch metric {
        case "pace": "\(Fmt.pace(v))\(unit == "s/100m" ? "/100m" : "/km")"
        case "heart_rate": "\(Int(v.rounded())) bpm"
        case "power": "\(Int(v.rounded())) W"
        default: "\(Int(v.rounded())) \(unit)"
        }
    }

    var rangeText: String {
        metric == "pace" ? "\(Fmt.pace(low))–\(format(high))" : "\(Int(low.rounded()))–\(format(high))"
    }
}

extension ComparisonDTO.Step.Result {
    var symbol: String {
        switch self {
        case .inRange: "checkmark.circle.fill"
        case .below: "arrow.down.circle.fill"
        case .above: "arrow.up.circle.fill"
        }
    }
    var color: Color { self == .inRange ? .green : .orange }
}

struct StepComparisonRow: View {
    let step: ComparisonDTO.Step

    var body: some View {
        HStack(alignment: .top) {
            RoundedRectangle(cornerRadius: 2).fill(step.kind.color).frame(width: 4)
            VStack(alignment: .leading, spacing: 2) {
                Text("\(step.index + 1). \(step.name ?? step.kind.displayName)").font(.subheadline.bold())
                Text("Plan: \(Fmt.duration(step.plannedDurationSec))\(step.target.map { " @ \($0.rangeText)" } ?? "")")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Done: \(Fmt.duration(step.actualDurationSec))\(actualText)")
                    .font(.caption)
            }
            Spacer()
            if let r = step.result {
                Image(systemName: r.symbol).foregroundStyle(r.color)
            }
        }
        .monospacedDigit()
    }

    private var actualText: String {
        var parts: [String] = []
        if let t = step.target, let v = step.actualValue { parts.append(t.format(v)) }
        if step.target?.metric != "heart_rate", let hr = step.actualAvgHr { parts.append("\(Int(hr)) bpm") }
        if step.target?.metric != "power", let p = step.actualAvgPower { parts.append("\(Int(p)) W") }
        return parts.isEmpty ? "" : " · " + parts.joined(separator: " · ")
    }
}

/// Target band per interval with the actual average on top.
struct TargetChart: View {
    let steps: [ComparisonDTO.Step]

    var body: some View {
        // Chart the workout's main metric (the one most steps target).
        let counts = Dictionary(grouping: steps.compactMap(\.target?.metric), by: { $0 }).mapValues(\.count)
        if let metric = counts.max(by: { $0.value < $1.value })?.key {
            let shown = steps.filter { $0.target?.metric == metric }
            Chart(shown) { s in
                if let t = s.target {
                    RectangleMark(
                        x: .value("Step", "\(s.index + 1)"),
                        yStart: .value("Low", t.low),
                        yEnd: .value("High", t.high),
                        width: .ratio(0.7)
                    )
                    .foregroundStyle(s.kind.color.opacity(0.3))
                }
                if let v = s.actualValue {
                    PointMark(x: .value("Step", "\(s.index + 1)"), y: .value("Actual", v))
                        .foregroundStyle(s.result?.color ?? .primary)
                        .symbolSize(60)
                }
            }
            .chartYScale(domain: .automatic(includesZero: false, reversed: metric == "pace"))
            .chartYAxisLabel(shown.first?.target?.unit ?? "")
            .chartXAxisLabel("Step")
            .frame(height: 180)
        }
    }
}
