import SwiftUI
import Charts
import PeakPlanKit

/// Performance management chart (fitness / fatigue / form) and weekly load, TrainingPeaks style.
struct FitnessView: View {
    @Environment(TrainingStore.self) private var store

    struct DayLoad: Identifiable {
        let date: Date
        let ctl: Double, atl: Double, tsb: Double
        var id: Date { date }
    }

    struct WeekLoad: Identifiable {
        let week: Date
        let kind: String
        let tss: Double
        var id: String { "\(week)-\(kind)" }
    }

    private var pmc: [DayLoad] {
        var daily: [String: Double] = [:]
        for a in store.activities { daily[String(a.startDate.prefix(10)), default: 0] += a.tss ?? 0 }
        guard let first = daily.keys.min() else { return [] }
        var ctl = 0.0, atl = 0.0, out: [DayLoad] = []
        var day = first
        let end = Day.adding(14, to: Day.today)
        // Future days use planned TSS so you can see where the plan takes your form.
        let planned = Dictionary(grouping: store.workouts.values.filter { $0.status == .planned }, by: { $0.date ?? "" })
        while day <= end {
            let tss = day <= Day.today
                ? daily[day] ?? 0
                : planned[day]?.compactMap(\.plannedTss).reduce(0, +) ?? 0
            let tsb = ctl - atl
            ctl += (tss - ctl) / 42
            atl += (tss - atl) / 7
            if let d = Day.date(day) { out.append(DayLoad(date: d, ctl: ctl, atl: atl, tsb: tsb)) }
            day = Day.adding(1, to: day)
        }
        return out
    }

    private var weekly: [WeekLoad] {
        let cal = Calendar(identifier: .iso8601)
        func week(_ s: String) -> Date? { Day.date(s).flatMap { cal.dateInterval(of: .weekOfYear, for: $0)?.start } }
        var planned: [Date: Double] = [:], done: [Date: Double] = [:]
        for w in store.workouts.values {
            if let d = w.date, let k = week(d) { planned[k, default: 0] += w.plannedTss ?? 0 }
        }
        for a in store.activities { if let k = week(String(a.startDate.prefix(10))) { done[k, default: 0] += a.tss ?? 0 } }
        return planned.map { WeekLoad(week: $0.key, kind: "Planned", tss: $0.value) }
            + done.map { WeekLoad(week: $0.key, kind: "Completed", tss: $0.value) }
    }

    var body: some View {
        List {
            let series = pmc
            if let today = series.last(where: { $0.date <= Date() }) {
                SwiftUI.Section {
                    HStack(spacing: 24) {
                        Metric(label: "Fitness (CTL)", value: String(format: "%.0f", today.ctl))
                        Metric(label: "Fatigue (ATL)", value: String(format: "%.0f", today.atl))
                        Metric(label: "Form (TSB)", value: String(format: "%+.0f", today.tsb))
                    }
                }
            }
            SwiftUI.Section("Performance management") {
                Chart(series) { d in
                    LineMark(x: .value("Day", d.date), y: .value("TSS/d", d.ctl)).foregroundStyle(by: .value("Metric", "Fitness"))
                    LineMark(x: .value("Day", d.date), y: .value("TSS/d", d.atl)).foregroundStyle(by: .value("Metric", "Fatigue"))
                    BarMark(x: .value("Day", d.date), y: .value("TSS/d", d.tsb)).foregroundStyle(by: .value("Metric", "Form"))
                        .opacity(0.4)
                }
                .chartForegroundStyleScale(["Fitness": Color.blue, "Fatigue": Color.pink, "Form": Color.yellow])
                .frame(height: 220)
                Text("Includes the next 14 days of planned workouts.").font(.caption).foregroundStyle(.secondary)
            }
            SwiftUI.Section("Weekly TSS") {
                Chart(weekly) { w in
                    BarMark(x: .value("Week", w.week, unit: .weekOfYear), y: .value("TSS", w.tss))
                        .foregroundStyle(by: .value("Kind", w.kind))
                        .position(by: .value("Kind", w.kind))
                }
                .frame(height: 200)
            }
            if store.activities.contains(where: { $0.tss == nil }) {
                Text("Set FTP / LTHR in Settings so completed workouts get a TSS.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Fitness")
    }
}
