import SwiftUI
import PeakPlanKit

struct TodayView: View {
    @Environment(TrainingStore.self) private var store

    var body: some View {
        List {
            let today = Day.today
            SwiftUI.Section("Today · \(Self.pretty(today))") {
                let ws = store.workouts(on: today)
                if ws.isEmpty { Text("Rest day 🛌").foregroundStyle(.secondary) }
                ForEach(ws) { w in
                    NavigationLink(value: w.id) { WorkoutRow(workout: w) }
                    if !w.steps.isEmpty { IntensityChart(workout: w, profile: store.profile).frame(height: 70) }
                }
                ForEach(store.activities(on: today)) { a in ActivitySummary(activity: a) }
            }
            ForEach(1..<7) { offset in
                let day = Day.adding(offset, to: today)
                let ws = store.workouts(on: day)
                if !ws.isEmpty {
                    SwiftUI.Section(Self.pretty(day)) {
                        ForEach(ws) { w in NavigationLink(value: w.id) { WorkoutRow(workout: w) } }
                    }
                }
            }
        }
        .navigationTitle("Today")
        .navigationDestination(for: String.self) { WorkoutDetailView(workoutID: $0) }
        .refreshable { await SyncCoordinator.fullSync(store) }
        .toolbar { NewWorkoutButton(day: Day.today) }
    }

    static func pretty(_ day: String) -> String {
        Day.date(day)?.formatted(.dateTime.weekday(.wide).day().month()) ?? day
    }
}

struct NewWorkoutButton: View {
    let day: String?
    @State private var draft: WorkoutDTO?

    var body: some View {
        Button { draft = WorkoutDTO(date: day) } label: { Label("New workout", systemImage: "plus") }
            .sheet(item: $draft) { w in
                NavigationStack { WorkoutEditorView(original: w) }
                    #if os(macOS)
                    .frame(minWidth: 640, minHeight: 640)
                    #endif
            }
    }
}

/// Week-by-week calendar. Drag a workout onto another day to move it (Option-drag on Mac /
/// "Copy" in the context menu to duplicate).
struct CalendarView: View {
    @Environment(TrainingStore.self) private var store
    @State private var weekOffset = 0

    private var weekStart: String {
        var cal = Calendar(identifier: .iso8601)
        cal.firstWeekday = 2
        let monday = cal.dateInterval(of: .weekOfYear, for: Date())?.start ?? Date()
        return Day.adding(weekOffset * 7, to: Day.string(monday))
    }

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 24) {
                ForEach(0..<4) { i in
                    WeekView(start: Day.adding(i * 7, to: weekStart))
                }
            }
            .padding()
        }
        .navigationTitle("Calendar")
        .navigationDestination(for: String.self) { WorkoutDetailView(workoutID: $0) }
        .refreshable { await SyncCoordinator.fullSync(store) }
        .toolbar {
            ToolbarItemGroup {
                Button { weekOffset -= 4 } label: { Image(systemName: "chevron.left") }
                Button("Today") { weekOffset = 0 }
                Button { weekOffset += 4 } label: { Image(systemName: "chevron.right") }
            }
        }
    }
}

struct WeekView: View {
    @Environment(TrainingStore.self) private var store
    let start: String

    private var days: [String] { (0..<7).map { Day.adding($0, to: start) } }

    var body: some View {
        let planned = days.flatMap { store.workouts(on: $0) }
        let tss = planned.compactMap(\.plannedTss).reduce(0, +)
        let hours = planned.compactMap(\.plannedDurationSec).reduce(0, +) / 3600
        let done = planned.filter { $0.status == .completed }.count

        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Week of \(TodayView.pretty(start))").font(.headline)
                Spacer()
                Text(String(format: "%.1f h · %ld TSS · %ld/%ld done", hours, Int(tss), done, planned.count))
                    .font(.caption).foregroundStyle(.secondary)
            }
            #if os(macOS)
            HStack(alignment: .top, spacing: 6) {
                ForEach(days, id: \.self) { DayCell(day: $0).frame(maxWidth: .infinity) }
            }
            #else
            VStack(spacing: 6) { ForEach(days, id: \.self) { DayCell(day: $0) } }
            #endif
        }
    }
}

struct DayCell: View {
    @Environment(TrainingStore.self) private var store
    let day: String
    @State private var targeted = false

    var body: some View {
        let isToday = day == Day.today
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(Day.date(day)?.formatted(.dateTime.weekday(.abbreviated).day()) ?? day)
                    .font(.caption.bold())
                    .foregroundStyle(isToday ? Color.accentColor : Color.secondary)
                Spacer()
                NewWorkoutButton(day: day).labelStyle(.iconOnly).buttonStyle(.borderless).font(.caption)
            }
            ForEach(store.workouts(on: day)) { w in
                NavigationLink(value: w.id) {
                    HStack(spacing: 6) {
                        Image(systemName: w.sport.symbol).foregroundStyle(w.status.color)
                        VStack(alignment: .leading) {
                            Text(w.title).font(.caption).lineLimit(2)
                            Text("\(Fmt.duration(w.plannedDurationSec)) · \(Int(w.plannedTss ?? 0)) TSS")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(6)
                    .background(.background.secondary, in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .draggable(w.id)
                .contextMenu {
                    Button("Duplicate to next day") { Task { await store.copy(w, to: Day.adding(1, to: day)) } }
                    Button("Move to library") { Task { await store.move(w, to: nil) } }
                    Button("Delete", role: .destructive) { Task { await store.delete(w) } }
                }
            }
            ForEach(store.activities(on: day)) { a in
                Label("\(a.title) · \(Fmt.duration(a.durationSec))", systemImage: "checkmark.seal")
                    .font(.caption2).foregroundStyle(.green)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 60, alignment: .topLeading)
        .padding(6)
        .background(targeted ? Color.accentColor.opacity(0.15) : Color.clear, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(isToday ? Color.accentColor : Color.secondary.opacity(0.2)))
        .dropDestination(for: String.self) { ids, _ in
            for id in ids {
                guard let w = store.workouts[id] else { continue }
                // Dropping a library template schedules a copy; dropping a dated workout moves it.
                Task { if w.date == nil { await store.copy(w, to: day) } else { await store.move(w, to: day) } }
            }
            return true
        } isTargeted: { targeted = $0 }
    }
}

struct LibraryView: View {
    @Environment(TrainingStore.self) private var store
    @State private var scheduling: WorkoutDTO?
    @State private var date = Date()

    var body: some View {
        List {
            if store.library.isEmpty {
                Text("Save workouts here as templates (turn off \"Scheduled\" in the editor), or ask Claude to build a library.")
                    .foregroundStyle(.secondary)
            }
            ForEach(store.library) { w in
                NavigationLink(value: w.id) { WorkoutRow(workout: w) }
                    .draggable(w.id)
                    .swipeActions { Button("Schedule") { scheduling = w }.tint(.blue) }
                    .contextMenu { Button("Schedule…") { scheduling = w } }
            }
        }
        .navigationTitle("Library")
        .navigationDestination(for: String.self) { WorkoutDetailView(workoutID: $0) }
        .toolbar { NewWorkoutButton(day: nil) }
        .sheet(item: $scheduling) { w in
            NavigationStack {
                Form { DatePicker("Date", selection: $date, displayedComponents: .date) }
                    .navigationTitle("Schedule \(w.title)")
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Schedule") {
                                Task { await store.copy(w, to: Day.string(date)) }
                                scheduling = nil
                            }
                        }
                        ToolbarItem(placement: .cancellationAction) { Button("Cancel") { scheduling = nil } }
                    }
            }
            .presentationDetents([.medium])
        }
    }
}
