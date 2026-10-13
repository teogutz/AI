import SwiftUI
import Observation
import WorkoutKit
import PeakPlanKit

@main
struct PeakPlanWatchApp: App {
    @State private var model = WatchModel()

    var body: some Scene {
        WindowGroup {
            NavigationStack { WatchTodayView() }
                .environment(model)
                .task { await model.loadScheduled() }
        }
    }
}

@MainActor
@Observable
final class WatchModel {
    /// Rich plan pushed from the iPhone (titles, notes, targets).
    var snapshot: WatchLink.Snapshot?
    /// Fallback: what's scheduled in the Workout app, available even if the phone never sent a snapshot.
    var scheduled: [ScheduledWorkoutPlan] = []

    init() {
        WatchLink.shared.onSnapshot = { [weak self] in self?.snapshot = $0 }
        WatchLink.shared.activate()
        snapshot = WatchLink.cachedSnapshot()
    }

    func loadScheduled() async {
        if await WorkoutScheduler.shared.authorizationState != .authorized {
            _ = await WorkoutScheduler.shared.requestAuthorization()
        }
        scheduled = await WorkoutScheduler.shared.scheduledWorkouts
    }

    func workouts(on day: String) -> [WorkoutDTO] {
        (snapshot?.workouts ?? []).filter { $0.date == day && $0.status == .planned }
    }

    var upcoming: [WorkoutDTO] {
        (snapshot?.workouts ?? []).filter { ($0.date ?? "") > Day.today && $0.status == .planned }
    }

    func plan(for w: WorkoutDTO) throws -> WorkoutPlan {
        try WorkoutKitBridge(profile: snapshot?.profile ?? AthleteProfile()).plan(for: w)
    }
}

struct WatchTodayView: View {
    @Environment(WatchModel.self) private var model

    var body: some View {
        List {
            let today = model.workouts(on: Day.today)
            Section("Today") {
                if model.snapshot == nil && model.scheduled.isEmpty {
                    Text("Open PeakPlan on your iPhone to sync your plan.").font(.footnote)
                } else if today.isEmpty && model.snapshot != nil {
                    Text("Rest day").foregroundStyle(.secondary)
                }
                ForEach(today) { w in
                    NavigationLink { WatchWorkoutView(workout: w) } label: { WatchRow(workout: w) }
                }
            }
            if model.snapshot == nil, !model.scheduled.isEmpty {
                Section("Scheduled") {
                    ForEach(model.scheduled, id: \.plan.id) { s in
                        ScheduledRow(scheduled: s)
                    }
                }
            }
            if !model.upcoming.isEmpty {
                Section("Coming up") {
                    ForEach(model.upcoming.prefix(6)) { w in
                        NavigationLink { WatchWorkoutView(workout: w) } label: { WatchRow(workout: w, showDate: true) }
                    }
                }
            }
        }
        .navigationTitle("PeakPlan")
    }
}

struct WatchRow: View {
    let workout: WorkoutDTO
    var showDate = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label(workout.title, systemImage: workout.sport.symbol).font(.headline).lineLimit(2)
            HStack {
                if showDate, let d = workout.date, let date = Day.date(d) {
                    Text(date.formatted(.dateTime.weekday(.abbreviated)))
                }
                Text(Fmt.duration(workout.plannedDurationSec))
                if let tss = workout.plannedTss { Text("\(Int(tss)) TSS") }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }
}

struct ScheduledRow: View {
    let scheduled: ScheduledWorkoutPlan
    @State private var error: String?

    var body: some View {
        Button {
            Task {
                do { try await scheduled.plan.openInWorkoutApp() } catch { self.error = error.localizedDescription }
            }
        } label: {
            VStack(alignment: .leading) {
                if case .custom(let c) = scheduled.plan.workout {
                    Text(c.displayName ?? "Workout").font(.headline)
                } else {
                    Text("Workout").font(.headline)
                }
                if let d = Calendar.current.date(from: scheduled.date) {
                    Text(d.formatted(.dateTime.weekday().hour().minute())).font(.caption2).foregroundStyle(.secondary)
                }
                if let error { Text(error).font(.caption2).foregroundStyle(.red) }
            }
        }
    }
}

struct WatchWorkoutView: View {
    @Environment(WatchModel.self) private var model
    let workout: WorkoutDTO
    @State private var error: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    Task { await start() }
                } label: {
                    Label("Start", systemImage: "play.fill").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)

                if let error { Text(error).font(.caption2).foregroundStyle(.red) }

                HStack {
                    Text(Fmt.duration(workout.plannedDurationSec))
                    if let tss = workout.plannedTss { Text("· \(Int(tss)) TSS") }
                }
                .font(.footnote.bold())

                if !workout.description.isEmpty { Text(workout.description).font(.footnote) }
                if !workout.coachNotes.isEmpty {
                    Text(workout.coachNotes).font(.footnote).foregroundStyle(.yellow)
                }

                ForEach(workout.steps) { element in
                    switch element {
                    case .step(let s):
                        StepText(step: s)
                    case .repeatBlock(let r):
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(r.count)×").font(.footnote.bold())
                            ForEach(r.steps) { StepText(step: $0).padding(.leading, 8) }
                        }
                    }
                }
            }
        }
        .navigationTitle(workout.title)
    }

    /// Hands the structured workout to Apple's Workout app, which runs the intervals with
    /// step-change haptics, target alerts, voice feedback and saves the result to Health.
    private func start() async {
        do {
            try await model.plan(for: workout).openInWorkoutApp()
        } catch {
            self.error = error.localizedDescription
        }
    }
}

struct StepText: View {
    let step: WorkoutStepDTO
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(step.name ?? step.kind.displayName).font(.caption.bold())
            Text([Fmt.stepDuration(step.duration), Fmt.target(step.target)].filter { !$0.isEmpty }.joined(separator: " @ "))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}
