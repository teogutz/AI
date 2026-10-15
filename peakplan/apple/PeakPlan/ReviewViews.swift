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
        case .unplanned: return "\(day) · Unplanned · \(Fmt.duration(item.activity?.durationSec)) · tap to pair"
        case .manual: return "\(day) · Marked done, no recorded workout"
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
    @State private var item: ReviewItemDTO
    @State private var comparison: ComparisonDTO?
    @State private var rpe: Double = 5
    @State private var hasRpe = false
    @State private var notes = ""
    @State private var workoutCandidates: [WorkoutCandidateDTO] = []
    @State private var activityCandidates: [ActivityCandidateDTO] = []
    @State private var moveToActivityDay = true
    @State private var confirmUnpair = false
    @State private var pairing = false
    @State private var shoeId: String?

    init(item: ReviewItemDTO) { _item = State(initialValue: item) }

    var body: some View {
        List {
            header

            if let c = comparison {
                SwiftUI.Section("Planned vs. completed") {
                    totals(c)
                    if let d = c.decoupling { DecouplingRow(d: d) }
                    if let shoe = c.shoe {
                        Label(shoe.name + (shoe.superShoe ? " · super shoe (~2–4 % faster)" : ""), systemImage: "shoe")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if c.intervalsMatched {
                    SwiftUI.Section {
                        TargetChart(steps: c.steps)
                        ForEach(c.steps) { StepComparisonRow(step: $0) }
                        if let extra = c.extraLaps, !extra.isEmpty {
                            ForEach(Array(extra.enumerated()), id: \.offset) { i, lap in
                                ExtraLapRow(index: c.steps.count + i + 1, lap: lap)
                            }
                        }
                    } header: {
                        Text("Intervals")
                    } footer: {
                        VStack(alignment: .leading, spacing: 4) {
                            if let e = c.execution, e.judged > 0 { Text(Self.executionText(e)) }
                            if let r = c.repConsistency { Text(Self.repText(r)) }
                            if let n = c.stepsNotDone, n > 0 { Text("\(n) planned step\(n == 1 ? "" : "s") not done (workout ended early).") }
                            if let extra = c.extraLaps, !extra.isEmpty { Text("Extra time recorded after the plan is shown in grey.") }
                        }
                    }
                } else if item.activity != nil, item.workout != nil {
                    SwiftUI.Section("Intervals") {
                        Text((c.recordedSegments ?? 0) == 0
                             ? "No interval data was recorded for this workout. Interval data is available when the workout is started from PeakPlan or from the Workout app's scheduled list."
                             : "The \(c.recordedSegments ?? 0) recorded segments don't line up with the \(c.steps.count) planned steps (they look like automatic km splits), so intervals can't be compared. Start the workout from PeakPlan or the Workout app's scheduled list to record its steps.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            } else if let a = item.activity {
                SwiftUI.Section("Completed") { ActivitySummary(activity: a) }
            }

            pairingSections

            if let w = item.workout, item.state == .missed || item.state == .skipped || item.state == .manual {
                SwiftUI.Section {
                    NavigationLink("Open planned workout", value: w.id)
                    if item.state == .missed {
                        Button("Mark as skipped") { var s = w; s.status = .skipped; Task { await store.save(s) } }
                    }
                    Button("Reschedule to today") { Task { await store.move(w, to: Day.today) } }
                }
            }

            if let a = item.activity {
                SwiftUI.Section("How did it feel?") {
                    Toggle("Rate effort (RPE)", isOn: $hasRpe)
                    if hasRpe {
                        Slider(value: $rpe, in: 1...10, step: 1) { Text("RPE") } minimumValueLabel: { Text("1") } maximumValueLabel: { Text("10") }
                        Text("RPE \(Int(rpe)) – \(Self.rpeLabel(Int(rpe)))").font(.caption)
                    }
                    TextField("Notes for your coach (Claude reads these)", text: $notes, axis: .vertical).lineLimit(2...6)
                    if !store.profile.shoes.isEmpty, a.sport == .run || a.sport == .walk || a.sport == .hike {
                        Picker("Shoes", selection: Binding(get: { shoeId ?? a.shoeId }, set: { new in
                            shoeId = new
                            Task {
                                guard let id = a.id else { return }
                                _ = try? await store.client?.setShoe(activityId: id, shoeId: new)
                                await load()
                            }
                        })) {
                            Text("Not set").tag(String?.none)
                            ForEach(store.profile.shoes.filter { !$0.retired || $0.id == a.shoeId }) { s in
                                Text(s.name + (s.superShoe ? " ⚡︎" : "")).tag(String?.some(s.id))
                            }
                        }
                    }
                    Button("Save feedback") {
                        Task { await store.updateActivity(a, rpe: hasRpe ? rpe : nil, notes: notes, workoutId: a.workoutId) }
                    }
                }
            }
        }
        .navigationTitle(item.workout?.title ?? item.activity?.title ?? "Review")
        .task { await load() }
        .confirmationDialog("Unpair this workout from its plan?", isPresented: $confirmUnpair, titleVisibility: .visible) {
            Button("Unpair", role: .destructive) { Task { await pair(workoutId: nil) } }
        } message: {
            Text("The plan goes back to planned and this workout becomes unplanned. Automatic matching won't pair them again.")
        }
    }

    private func load() async {
        if let r = item.activity?.rpe { rpe = r; hasRpe = true }
        notes = item.activity?.notes ?? ""
        guard let api = store.client else { return }
        comparison = nil
        workoutCandidates = []
        activityCandidates = []
        if let w = item.workout, item.activity != nil {
            comparison = try? await api.comparison(workoutId: w.id)
        }
        if item.state == .unplanned, let id = item.activity?.id {
            workoutCandidates = (try? await api.workoutCandidates(activityId: id)) ?? []
        }
        if let w = item.workout, item.activity == nil {
            activityCandidates = (try? await api.activityCandidates(workoutId: w.id)) ?? []
        }
    }

    // MARK: Pairing

    /// Pair this item's activity with `workoutId` (nil = unpair) and show the result in place.
    private func pair(workoutId: String?, activity: ActivityDTO? = nil) async {
        guard let api = store.client, let a = activity ?? item.activity, let activityId = a.id else { return }
        pairing = true
        defer { pairing = false }
        do {
            let r = try await api.link(activityId: activityId, workoutId: workoutId, moveWorkout: moveToActivityDay)
            let day = String(r.activity.startDate.prefix(10))
            item = r.workout.map { ReviewItemDTO(date: day, state: .completed, workout: $0, activity: r.activity) }
                ?? ReviewItemDTO(date: day, state: .unplanned, activity: r.activity)
            await load()
            Task { await store.refresh() }
        } catch {
            store.lastError = error.localizedDescription
        }
    }

    @ViewBuilder
    private var pairingSections: some View {
        if item.state == .unplanned, let a = item.activity {
            SwiftUI.Section {
                if workoutCandidates.isEmpty {
                    Text("No planned workouts within 3 days of this one.").foregroundStyle(.secondary)
                }
                ForEach(workoutCandidates) { c in
                    Button { Task { await pair(workoutId: c.workout.id) } } label: {
                        CandidateRow(
                            symbol: c.workout.sport.symbol,
                            title: c.workout.title,
                            detail: Self.offsetText(c.dayOffset, planned: true) + " · " + Fmt.duration(c.workout.plannedDurationSec),
                            warnings: [c.sameSport ? nil : "Different sport (\(c.workout.sport.displayName))",
                                       c.pairedActivityId != nil ? "Already paired. Pairing replaces it." : nil].compactMap { $0 }
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(pairing)
                }
                if workoutCandidates.contains(where: { $0.dayOffset != 0 }) {
                    Toggle("Move the plan to \(TodayView.pretty(String(a.startDate.prefix(10))))", isOn: $moveToActivityDay)
                }
            } header: {
                Text("Pair with a planned workout")
            } footer: {
                Text("Pick the plan this workout was meant to be. It will be compared against it and count as done.")
            }
        }

        if item.activity == nil, let w = item.workout, item.state != .skipped || !activityCandidates.isEmpty {
            SwiftUI.Section {
                if activityCandidates.isEmpty {
                    Text("No recorded workouts within 3 days of \(TodayView.pretty(w.date ?? item.date)).").foregroundStyle(.secondary)
                }
                ForEach(activityCandidates) { c in
                    Button { Task { await pair(workoutId: w.id, activity: c.activity) } } label: {
                        CandidateRow(
                            symbol: c.activity.sport.symbol,
                            title: c.activity.title,
                            detail: Self.offsetText(c.dayOffset, planned: false) + " · " + Fmt.duration(c.activity.durationSec)
                                + (c.activity.distanceM.map { " · " + Fmt.distance($0) } ?? ""),
                            warnings: [c.sameSport ? nil : "Different sport (\(c.activity.sport.displayName))",
                                       c.pairedWorkoutId != nil ? "Paired with another plan. Pairing moves it here." : nil].compactMap { $0 }
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(pairing)
                }
                if activityCandidates.contains(where: { $0.dayOffset != 0 }) {
                    Toggle("Move the plan to the day it was done", isOn: $moveToActivityDay)
                }
            } header: {
                Text("Pair with a recorded workout")
            } footer: {
                Text("Did this plan on another day, or the Watch recorded it as a different sport? Pick the recorded workout.")
            }
        }

        if item.state == .completed, item.activity != nil {
            SwiftUI.Section {
                Button("Unpair from this plan", role: .destructive) { confirmUnpair = true }
                    .disabled(pairing)
            } footer: {
                Text("Use this if the workout was matched to the wrong plan. You can then pair it with the right one.")
            }
        }
    }

    static func offsetText(_ offset: Int, planned: Bool) -> String {
        switch offset {
        case 0: return "Same day"
        case let d where d < 0: return planned ? "Planned \(-d) day\(d == -1 ? "" : "s") earlier" : "Done \(-d) day\(d == -1 ? "" : "s") earlier"
        default: return planned ? "Planned \(offset) day\(offset == 1 ? "" : "s") later" : "Done \(offset) day\(offset == 1 ? "" : "s") later"
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
            if let w = item.workout, w.hasIntensityProfile(store.profile) {
                IntensityStrip(workout: w, profile: store.profile).frame(width: 140, height: 36)
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

    static func executionText(_ e: ComparisonDTO.Execution) -> String {
        var parts = ["\(e.inRange) of \(e.judged) on target"]
        if e.tooHard > 0 { parts.append("\(e.tooHard) harder than planned") }
        if e.tooEasy > 0 { parts.append("\(e.tooEasy) easier than planned") }
        let verdict = switch e.verdict ?? "" {
        case "over_executed": " → over-executed"
        case "under_executed": " → under-executed"
        case "mixed": " → mixed"
        default: ""
        }
        return parts.joined(separator: " · ") + verdict
    }

    static func repText(_ r: ComparisonDTO.RepConsistency) -> String {
        let fmt: (Double) -> String = r.metric == "pace" ? { Fmt.pace($0) } : { "\(Int($0.rounded()))" }
        let unit = r.metric == "pace" ? "s/km" : r.unit
        let spread = r.metric == "pace" ? "\(Int(r.spread.rounded())) s/km" : "\(Int(r.spread.rounded())) \(r.unit)"
        let fade = r.fade == 0 ? "" : (r.fade > 0 ? " · faded \(Int(r.fade.rounded())) \(unit)" : " · negative split")
        return "Rep spread \(spread) (best \(fmt(r.best)), worst \(fmt(r.worst)))\(fade)"
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
        case .tooHard: "arrow.up.circle.fill"
        case .tooEasy: "arrow.down.circle.fill"
        case .unknown: "questionmark.circle"
        }
    }
    /// Orange = harder than planned, blue = easier than planned.
    var color: Color {
        switch self {
        case .inRange: .green
        case .tooHard: .orange
        case .tooEasy: .blue
        case .unknown: .secondary
        }
    }
}

extension ComparisonDTO.Step {
    /// e.g. "6 % too fast", "4 % too low"
    var deviationText: String? {
        guard let d = deviation else { return nil }
        let words = d.replacingOccurrences(of: "_", with: " ")
        guard let p = deviationPct else { return words }
        return String(format: "%.0f%% ", abs(p)) + words
    }
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
        if let d = step.deviationText { parts.append(d) }
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

struct CandidateRow: View {
    let symbol: String
    let title: String
    let detail: String
    let warnings: [String]

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol).frame(width: 28).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(detail).font(.caption).foregroundStyle(.secondary)
                ForEach(warnings, id: \.self) { Text($0).font(.caption2).foregroundStyle(.orange) }
            }
            Spacer()
            Text("Pair").font(.subheadline.bold()).foregroundStyle(Color.accentColor)
        }
        .contentShape(Rectangle())
    }
}

/// A segment recorded after the planned steps (e.g. "Open" in Apple Fitness).
struct ExtraLapRow: View {
    let index: Int
    let lap: LapDTO

    var body: some View {
        HStack(alignment: .top) {
            RoundedRectangle(cornerRadius: 2).fill(Color.gray).frame(width: 4)
            VStack(alignment: .leading, spacing: 2) {
                Text("\(index). Extra (not in plan)").font(.subheadline.bold()).foregroundStyle(.secondary)
                Text(details).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .monospacedDigit()
    }

    private var details: String {
        var parts = [Fmt.duration(lap.durationSec)]
        if let m = lap.distanceM, m > 0 {
            parts.append(Fmt.distance(m))
            parts.append(Fmt.pace(lap.durationSec / (m / 1000)) + " /km")
        }
        if let hr = lap.avgHr { parts.append("\(Int(hr)) bpm") }
        return parts.joined(separator: " · ")
    }
}

struct DecouplingRow: View {
    let d: DecouplingDTO

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text("Cardiac drift").foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f%%", d.decouplingPct)).bold()
                Text(d.rating).font(.caption.bold())
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(color.opacity(0.2), in: Capsule())
                    .foregroundStyle(color)
            }
            Text(detail).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var color: Color { d.rating == "good" ? .green : d.rating == "moderate" ? .orange : .red }

    private var detail: String {
        let p = { (h: DecouplingDTO.Half) in h.paceSecPerKm.map { Fmt.pace($0) + "/km @ " } ?? "" }
        return "1st half \(p(d.firstHalf))\(Int(d.firstHalf.avgHr)) bpm → 2nd half \(p(d.secondHalf))\(Int(d.secondHalf.avgHr)) bpm. Under 5 % on a steady run = good aerobic durability."
    }
}
