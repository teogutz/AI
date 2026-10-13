#if canImport(WorkoutKit) && !os(macOS)
import Foundation
import HealthKit
import WorkoutKit

/// Turns a PeakPlan workout into an Apple `WorkoutPlan`, the same mechanism TrainingPeaks,
/// TrainerRoad etc. use: the plan is scheduled with `WorkoutScheduler` and then shows up in the
/// Apple Watch Workout app, which runs the intervals with its own alerts, haptics and voice.
public struct WorkoutKitBridge {
    public let resolver: TargetResolver

    public init(profile: AthleteProfile) { resolver = TargetResolver(profile: profile) }

    public enum BridgeError: LocalizedError {
        case unsupportedSport(Sport)
        case empty
        public var errorDescription: String? {
            switch self {
            case .unsupportedSport(let s): "\(s.displayName) workouts can't be sent to Apple Watch as structured intervals."
            case .empty: "Add at least one step before sending to Apple Watch."
            }
        }
    }

    public static func activityType(_ sport: Sport) -> HKWorkoutActivityType {
        switch sport {
        case .run: .running
        case .bike: .cycling
        case .swim: .swimming
        case .walk: .walking
        case .hike: .hiking
        case .row: .rowing
        case .strength: .traditionalStrengthTraining
        case .other: .mixedCardio
        }
    }

    public func plan(for w: WorkoutDTO) throws -> WorkoutPlan {
        let activity = Self.activityType(w.sport)
        let location: HKWorkoutSessionLocationType = w.location == .indoor ? .indoor : .outdoor
        guard CustomWorkout.supportsActivity(activity) else { throw BridgeError.unsupportedSport(w.sport) }
        guard !w.steps.isEmpty else { throw BridgeError.empty }

        var elements = w.steps
        var warmup: WorkoutStep?
        var cooldown: WorkoutStep?
        if case .step(let s) = elements.first, s.kind == .warmup {
            warmup = workoutStep(s, activity: activity, location: location)
            elements.removeFirst()
        }
        if case .step(let s) = elements.last, s.kind == .cooldown {
            cooldown = workoutStep(s, activity: activity, location: location)
            elements.removeLast()
        }

        // Repeats become interval blocks; runs of single steps are grouped into one-iteration blocks.
        var blocks: [IntervalBlock] = []
        var pending: [IntervalStep] = []
        func flush() {
            if !pending.isEmpty { blocks.append(IntervalBlock(steps: pending, iterations: 1)) }
            pending = []
        }
        for element in elements {
            switch element {
            case .step(let s):
                pending.append(intervalStep(s, activity: activity, location: location))
            case .repeatBlock(let r):
                flush()
                blocks.append(IntervalBlock(
                    steps: r.steps.map { intervalStep($0, activity: activity, location: location) },
                    iterations: r.count
                ))
            }
        }
        flush()

        let custom = CustomWorkout(
            activity: activity,
            location: location,
            displayName: w.title,
            warmup: warmup,
            blocks: blocks,
            cooldown: cooldown
        )
        return WorkoutPlan(.custom(custom), id: UUID(uuidString: w.id) ?? UUID())
    }

    private func intervalStep(_ s: WorkoutStepDTO, activity: HKWorkoutActivityType, location: HKWorkoutSessionLocationType) -> IntervalStep {
        let purpose: IntervalStep.Purpose = s.kind == .work ? .work : .recovery
        var step = IntervalStep(purpose)
        step.step = workoutStep(s, activity: activity, location: location)
        return step
    }

    private func workoutStep(_ s: WorkoutStepDTO, activity: HKWorkoutActivityType, location: HKWorkoutSessionLocationType) -> WorkoutStep {
        var step = WorkoutStep(goal: .open)
        let stepGoal = goal(s.duration)
        if CustomWorkout.supportsGoal(stepGoal, activity: activity, location: location) { step.goal = stepGoal }
        if let alert = alert(s.target), CustomWorkout.supportsAlert(alert, activity: activity, location: location) {
            step.alert = alert
        }
        return step
    }

    private func goal(_ d: StepDuration) -> WorkoutGoal {
        switch d {
        case .time(let s): .time(s, .seconds)
        case .distance(let m): .distance(m, .meters)
        case .open: .open
        }
    }

    /// Absolute alert ranges; relative targets are resolved against the athlete's thresholds.
    private func alert(_ t: StepTarget?) -> (any WorkoutAlert)? {
        guard let t else { return nil }
        switch t {
        case .power:
            guard let watts = resolver.watts(t) else { return nil }
            let alert: PowerRangeAlert = .power(watts, unit: .watts)
            return alert
        case .heartRateZone(let zone):
            let alert: HeartRateZoneAlert = .heartRate(zone: zone)
            return alert
        case .heartRate:
            guard let bpm = resolver.bpm(t) else { return nil }
            let alert: HeartRateRangeAlert = .heartRate(bpm)
            return alert
        case .pace:
            guard let speed = resolver.speed(t) else { return nil }
            let alert: SpeedRangeAlert = .speed(speed, unit: .metersPerSecond, metric: .current)
            return alert
        case .cadence(let lo, let hi):
            let alert: CadenceRangeAlert = .cadence(lo...hi)
            return alert
        }
    }
}

/// Keeps the Apple Watch's scheduled workouts in sync with the PeakPlan calendar.
public enum WatchScheduleSync {
    /// Default reminder time for workouts without a start time.
    public static var defaultHour = 6

    public static func authorize() async -> Bool {
        await WorkoutScheduler.shared.requestAuthorization() == .authorized
    }

    /// Replaces everything this app has scheduled with the upcoming planned workouts.
    /// Returns the workouts that could not be converted, with the reason.
    @discardableResult
    public static func sync(_ workouts: [WorkoutDTO], profile: AthleteProfile) async -> [(WorkoutDTO, Error)] {
        let scheduler = WorkoutScheduler.shared
        var state = await scheduler.authorizationState
        if state != .authorized { state = await scheduler.requestAuthorization() }
        guard state == .authorized else { return [] }

        let bridge = WorkoutKitBridge(profile: profile)
        let today = Day.today
        let upcoming = workouts
            .filter { $0.status == .planned && ($0.date ?? "") >= today }
            .sorted { ($0.date ?? "", $0.startTime ?? "") < ($1.date ?? "", $1.startTime ?? "") }
            .prefix(WorkoutScheduler.maxAllowedScheduledWorkoutCount)

        await scheduler.removeAllWorkouts()
        var failures: [(WorkoutDTO, Error)] = []
        for w in upcoming {
            guard let date = w.date, let comps = dateComponents(day: date, time: w.startTime) else { continue }
            do {
                let plan = try bridge.plan(for: w)
                await scheduler.schedule(plan, at: comps)
            } catch {
                failures.append((w, error))
            }
        }
        return failures
    }

    static func dateComponents(day: String, time: String?) -> DateComponents? {
        let d = day.split(separator: "-").compactMap { Int($0) }
        guard d.count == 3 else { return nil }
        let t = (time ?? "").split(separator: ":").compactMap { Int($0) }
        return DateComponents(
            year: d[0], month: d[1], day: d[2],
            hour: t.count == 2 ? t[0] : defaultHour,
            minute: t.count == 2 ? t[1] : 0
        )
    }
}
#endif
