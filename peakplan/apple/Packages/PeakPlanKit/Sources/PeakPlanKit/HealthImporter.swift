#if canImport(HealthKit) && os(iOS)
import Foundation
import HealthKit

/// Reads completed workouts from Apple Health (recorded by the Watch's Workout app — including
/// the PeakPlan workouts it ran) and converts them into activities for the server, so Claude can
/// analyse them.
public final class HealthImporter {
    private let store = HKHealthStore()

    public init() {}

    private static let bpm = HKUnit.count().unitDivided(by: .minute())

    private var readTypes: Set<HKObjectType> {
        var types: Set<HKObjectType> = [HKObjectType.workoutType()]
        let ids: [HKQuantityTypeIdentifier] = [
            .heartRate, .activeEnergyBurned, .distanceWalkingRunning, .distanceCycling, .distanceSwimming,
            .cyclingPower, .runningPower, .cyclingCadence,
        ]
        for id in ids { types.insert(HKQuantityType(id)) }
        // For pre-filling the athlete profile.
        let profileIds: [HKQuantityTypeIdentifier] = [
            .restingHeartRate, .bodyMass, .vo2Max, .cyclingFunctionalThresholdPower,
        ]
        for id in profileIds { types.insert(HKQuantityType(id)) }
        types.insert(HKCharacteristicType(.dateOfBirth))
        return types
    }

    public func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        try await store.requestAuthorization(toShare: [], read: readTypes)
    }

    /// Workouts from the last `days` days, skipping the HealthKit UUIDs in `excluding`
    /// (already uploaded) so per-interval statistics are only computed once.
    public func recentActivities(days: Int = 30, excluding: Set<String> = []) async throws -> [ActivityDTO] {
        let start = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: start, end: nil)
        let descriptor = HKSampleQueryDescriptor(
            predicates: [.workout(predicate)],
            sortDescriptors: [SortDescriptor(\.startDate, order: .reverse)],
            limit: 500
        )
        var result: [ActivityDTO] = []
        for w in try await descriptor.result(for: store) where !excluding.contains(w.uuid.uuidString) {
            var a = activity(from: w)
            a.laps = await laps(for: w, sport: a.sport)
            if w.duration >= 40 * 60 {
                let mid = w.startDate.addingTimeInterval(w.duration / 2)
                a.firstHalf = await half(DateInterval(start: w.startDate, end: mid), sport: a.sport)
                a.secondHalf = await half(DateInterval(start: mid, end: w.startDate.addingTimeInterval(w.duration)), sport: a.sport)
            }
            result.append(a)
        }
        return result
    }

    /// The Workout app records a segment event at every step of a structured workout; manual
    /// laps are used as a fallback. Each becomes a lap with its own averages, so steps can be
    /// compared with their targets.
    private func laps(for w: HKWorkout, sport: Sport) async -> [LapDTO] {
        let distanceType: HKQuantityTypeIdentifier = switch sport {
        case .bike: .distanceCycling
        case .swim: .distanceSwimming
        default: .distanceWalkingRunning
        }

        // Structured (custom/interval) workouts: the Workout app stores each step (Warm Up,
        // Work, Recovery, Cool Down, Open) as its own HKWorkoutActivity. This is what Apple
        // Fitness shows under "Intervals".
        let steps = w.workoutActivities.sorted { $0.startDate < $1.startDate }
        if steps.count > 1 {
            var laps: [LapDTO] = []
            for a in steps {
                let end = a.endDate ?? a.startDate.addingTimeInterval(a.duration)
                let interval = DateInterval(start: a.startDate, end: max(end, a.startDate))
                func stat(_ id: HKQuantityTypeIdentifier, _ unit: HKUnit, _ option: HKStatisticsOptions) async -> Double? {
                    if let s = a.statistics(for: HKQuantityType(id)) {
                        let q = option.contains(.cumulativeSum) ? s.sumQuantity() : s.averageQuantity()
                        if let q { return q.doubleValue(for: unit) }
                    }
                    return await statistic(id, unit: unit, interval, option)
                }
                var lap = LapDTO(startOffsetSec: a.startDate.timeIntervalSince(w.startDate), durationSec: interval.duration)
                lap.distanceM = await stat(distanceType, .meter(), .cumulativeSum)
                lap.avgHr = await stat(.heartRate, Self.bpm, .discreteAverage)
                if sport == .bike {
                    lap.avgPower = await stat(.cyclingPower, .watt(), .discreteAverage)
                    lap.avgCadence = await stat(.cyclingCadence, Self.bpm, .discreteAverage)
                } else if sport == .run {
                    lap.avgPower = await stat(.runningPower, .watt(), .discreteAverage)
                }
                laps.append(lap)
            }
            return laps
        }

        // Otherwise use segment / lap markers (free runs: automatic km and mile splits).
        let events = w.workoutEvents ?? []
        var intervals = events.filter { $0.type == .segment }.map(\.dateInterval)
        if intervals.isEmpty { intervals = events.filter { $0.type == .lap }.map(\.dateInterval) }
        let distanceId = distanceType
        let chains = Self.chains(intervals)
        guard !chains.isEmpty else { return [] }

        // Measure each series' distances, then keep the one that describes the workout's steps.
        var measured: [[LapDTO]] = []
        for chain in chains {
            var laps: [LapDTO] = []
            for interval in chain {
                var lap = LapDTO(startOffsetSec: interval.start.timeIntervalSince(w.startDate), durationSec: interval.duration)
                lap.distanceM = await statistic(distanceId, unit: .meter(), interval, .cumulativeSum)
                laps.append(lap)
            }
            measured.append(laps)
        }
        let best = Self.pickStepSeries(measured)

        var laps: [LapDTO] = []
        for var lap in best {
            let start = w.startDate.addingTimeInterval(lap.startOffsetSec)
            let interval = DateInterval(start: start, duration: lap.durationSec)
            lap.avgHr = await statistic(.heartRate, unit: Self.bpm, interval, .discreteAverage)
            if sport == .bike {
                lap.avgPower = await statistic(.cyclingPower, unit: .watt(), interval, .discreteAverage)
                lap.avgCadence = await statistic(.cyclingCadence, unit: Self.bpm, interval, .discreteAverage)
            } else if sport == .run {
                lap.avgPower = await statistic(.runningPower, unit: .watt(), interval, .discreteAverage)
            }
            laps.append(lap)
        }
        return laps
    }

    /// Averages for one half of a workout (cardiac drift / aerobic decoupling).
    private func half(_ interval: DateInterval, sport: Sport) async -> HalfDTO {
        let distanceId: HKQuantityTypeIdentifier = switch sport {
        case .bike: .distanceCycling
        case .swim: .distanceSwimming
        default: .distanceWalkingRunning
        }
        var h = HalfDTO(durationSec: interval.duration)
        h.distanceM = await statistic(distanceId, unit: .meter(), interval, .cumulativeSum)
        h.avgHr = await statistic(.heartRate, unit: Self.bpm, interval, .discreteAverage)
        if sport == .bike { h.avgPower = await statistic(.cyclingPower, unit: .watt(), interval, .discreteAverage) }
        return h
    }

    /// The Watch records several overlapping series at once (auto km splits, auto mile splits and,
    /// for structured workouts, one segment per step). Split them into non-overlapping series.
    static func chains(_ intervals: [DateInterval]) -> [[DateInterval]] {
        var chains: [[DateInterval]] = []
        for i in intervals.sorted(by: { ($0.start, $0.duration) < ($1.start, $1.duration) }) {
            if let idx = chains.firstIndex(where: { $0.last!.end <= i.start.addingTimeInterval(1) }) {
                chains[idx].append(i)
            } else {
                chains.append([i])
            }
        }
        return chains
    }

    /// Prefers a series that isn't automatic distance splits (i.e. the workout's own steps);
    /// otherwise the kilometre splits.
    static func pickStepSeries(_ series: [[LapDTO]]) -> [LapDTO] {
        func isAutoSplit(_ laps: [LapDTO], meters: Double) -> Bool {
            let full = laps.dropLast().compactMap(\.distanceM)
            guard full.count >= 2 else { return false }
            return Double(full.filter { abs($0 - meters) / meters < 0.03 }.count) / Double(full.count) >= 0.7
        }
        let auto = series.map { isAutoSplit($0, meters: 1000) || isAutoSplit($0, meters: 1609.34) }
        if let steps = series.indices.first(where: { !auto[$0] }) { return series[steps] }
        return series.first(where: { isAutoSplit($0, meters: 1000) }) ?? series[0]
    }

    private func statistic(_ id: HKQuantityTypeIdentifier, unit: HKUnit, _ interval: DateInterval,
                           _ option: HKStatisticsOptions) async -> Double? {
        let predicate = HKQuery.predicateForSamples(withStart: interval.start, end: interval.end, options: .strictStartDate)
        let descriptor = HKStatisticsQueryDescriptor(
            predicate: .quantitySample(type: HKQuantityType(id), predicate: predicate),
            options: option
        )
        guard let stats = try? await descriptor.result(for: store) else { return nil }
        let quantity = option.contains(.cumulativeSum) ? stats.sumQuantity() : stats.averageQuantity()
        return quantity?.doubleValue(for: unit)
    }

    private func activity(from w: HKWorkout) -> ActivityDTO {
        let iso = ISO8601DateFormatter()
        iso.timeZone = .current // keep the local offset so the server buckets by the athlete's day
        let sport = Self.sport(w.workoutActivityType)

        var a = ActivityDTO(
            externalId: w.uuid.uuidString,
            sport: sport,
            title: (w.metadata?[HKMetadataKeyWorkoutBrandName] as? String) ?? "\(sport.displayName) workout",
            startDate: iso.string(from: w.startDate),
            durationSec: w.duration
        )
        func stat(_ id: HKQuantityTypeIdentifier) -> HKStatistics? { w.statistics(for: HKQuantityType(id)) }

        let distanceId: HKQuantityTypeIdentifier = switch sport {
        case .bike: .distanceCycling
        case .swim: .distanceSwimming
        default: .distanceWalkingRunning
        }
        a.distanceM = stat(distanceId)?.sumQuantity()?.doubleValue(for: .meter())
        a.avgHr = stat(.heartRate)?.averageQuantity()?.doubleValue(for: Self.bpm)
        a.maxHr = stat(.heartRate)?.maximumQuantity()?.doubleValue(for: Self.bpm)
        a.energyKcal = stat(.activeEnergyBurned)?.sumQuantity()?.doubleValue(for: .kilocalorie())
        let powerId: HKQuantityTypeIdentifier = sport == .bike ? .cyclingPower : .runningPower
        a.avgPower = stat(powerId)?.averageQuantity()?.doubleValue(for: .watt())
        if sport == .bike { a.avgCadence = stat(.cyclingCadence)?.averageQuantity()?.doubleValue(for: Self.bpm) }
        if let elev = w.metadata?[HKMetadataKeyElevationAscended] as? HKQuantity {
            a.elevationGainM = elev.doubleValue(for: .meter())
        }
        return a
    }

    // MARK: - Profile suggestions

    public struct ProfileSuggestion: Identifiable, Sendable {
        public enum Field: String, Sendable, CaseIterable {
            case ftp, thresholdPace, lthr, maxHr, restingHr, weight, vo2Max
        }
        public let field: Field
        public let value: Double
        /// Where the number came from, shown to the athlete.
        public let source: String
        /// True when calculated by PeakPlan rather than measured by Apple.
        public let isEstimate: Bool
        public var id: String { field.rawValue }
    }

    /// Everything Apple Health can tell us for the athlete profile. Apple stores FTP (estimated by
    /// the Watch from rides with a power meter), resting HR, weight, VO2max and age; threshold pace,
    /// max HR and threshold HR are estimated from workout history.
    public func profileSuggestions() async -> [ProfileSuggestion] {
        try? await requestAuthorization()
        var out: [ProfileSuggestion] = []
        let df = Date.FormatStyle(date: .abbreviated, time: .omitted)

        if let ftp = await latest(.cyclingFunctionalThresholdPower, unit: .watt()) {
            out.append(.init(field: .ftp, value: ftp.value.rounded(), source: "Apple Watch FTP estimate (\(ftp.date.formatted(df)))", isEstimate: false))
        }

        let resting = await statistic(.restingHeartRate, unit: Self.bpm, days: 30, .discreteAverage)
        if let resting {
            out.append(.init(field: .restingHr, value: resting.rounded(), source: "Average resting HR, last 30 days", isEstimate: false))
        }

        if let kg = await latest(.bodyMass, unit: .gramUnit(with: .kilo)) {
            out.append(.init(field: .weight, value: (kg.value * 10).rounded() / 10, source: "Latest weight (\(kg.date.formatted(df)))", isEstimate: false))
        }

        if let vo2 = await latest(.vo2Max, unit: HKUnit(from: "ml/kg*min")) {
            out.append(.init(field: .vo2Max, value: (vo2.value * 10).rounded() / 10, source: "Cardio Fitness (\(vo2.date.formatted(df)))", isEstimate: false))
        }

        // Max HR: the highest recorded in the last year, unless it looks like a sensor spike,
        // otherwise the age formula (Tanaka: 208 − 0.7 × age).
        var age: Double?
        if let dob = try? store.dateOfBirthComponents(), let birth = Calendar.current.date(from: dob) {
            age = Double(Calendar.current.dateComponents([.year], from: birth, to: Date()).year ?? 0)
        }
        let formulaMax = age.map { 208 - 0.7 * $0 }
        let observedMax = await statistic(.heartRate, unit: Self.bpm, days: 365, .discreteMax)
        var maxHr: Double?
        if let observedMax, observedMax >= 140, observedMax <= (formulaMax.map { $0 + 15 } ?? 215) {
            maxHr = observedMax.rounded()
            out.append(.init(field: .maxHr, value: maxHr!, source: "Highest heart rate recorded in the last year", isEstimate: false))
        } else if let formulaMax {
            maxHr = formulaMax.rounded()
            out.append(.init(field: .maxHr, value: maxHr!, source: "Estimated from your age (\(Int(age!)))", isEstimate: true))
        }

        // Threshold HR ≈ 85 % of heart-rate reserve above resting (≈ 88–90 % of max).
        if let maxHr {
            let lthr = resting.map { $0 + 0.85 * (maxHr - $0) } ?? 0.89 * maxHr
            out.append(.init(field: .lthr, value: lthr.rounded(), source: "Estimated from max and resting HR. A 30-min test is more accurate.", isEstimate: true))
        }

        if let pace = await thresholdPaceEstimate() {
            out.append(pace)
        }
        return out
    }

    /// Fastest run of 20–70 minutes in the last 4 months. A hard run of about an hour is run at
    /// threshold pace; shorter efforts are a bit faster, so they're adjusted slightly slower.
    private func thresholdPaceEstimate() async -> ProfileSuggestion? {
        let start = Calendar.current.date(byAdding: .day, value: -120, to: Date())!
        let predicate = NSCompoundPredicate(andPredicateWithSubpredicates: [
            HKQuery.predicateForSamples(withStart: start, end: nil),
            HKQuery.predicateForWorkouts(with: .running),
        ])
        let descriptor = HKSampleQueryDescriptor(predicates: [.workout(predicate)], sortDescriptors: [], limit: 300)
        guard let runs = try? await descriptor.result(for: store) else { return nil }
        var best: (pace: Double, run: HKWorkout, km: Double)?
        for run in runs where run.duration >= 20 * 60 && run.duration <= 70 * 60 {
            guard let m = run.statistics(for: HKQuantityType(.distanceWalkingRunning))?.sumQuantity()?.doubleValue(for: .meter()),
                  m > 3000 else { continue }
            var pace = run.duration / (m / 1000)
            if run.duration < 40 * 60 { pace *= 1.04 }
            if best == nil || pace < best!.pace { best = (pace, run, m / 1000) }
        }
        guard let best else { return nil }
        let day = best.run.startDate.formatted(date: .abbreviated, time: .omitted)
        return .init(field: .thresholdPace, value: best.pace.rounded(),
                     source: String(format: "From your fastest run of 20–70 min (%.1f km on %@)", best.km, day),
                     isEstimate: true)
    }

    private func latest(_ id: HKQuantityTypeIdentifier, unit: HKUnit) async -> (value: Double, date: Date)? {
        let descriptor = HKSampleQueryDescriptor(
            predicates: [.quantitySample(type: HKQuantityType(id))],
            sortDescriptors: [SortDescriptor(\.endDate, order: .reverse)],
            limit: 1
        )
        guard let sample = try? await descriptor.result(for: store).first else { return nil }
        return (sample.quantity.doubleValue(for: unit), sample.endDate)
    }

    private func statistic(_ id: HKQuantityTypeIdentifier, unit: HKUnit, days: Int, _ option: HKStatisticsOptions) async -> Double? {
        let start = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: start, end: Date())
        let descriptor = HKStatisticsQueryDescriptor(
            predicate: .quantitySample(type: HKQuantityType(id), predicate: predicate),
            options: option
        )
        guard let stats = try? await descriptor.result(for: store) else { return nil }
        let q: HKQuantity? = switch option {
        case .discreteMax: stats.maximumQuantity()
        case .discreteAverage: stats.averageQuantity()
        default: stats.sumQuantity()
        }
        return q?.doubleValue(for: unit)
    }

    static func sport(_ t: HKWorkoutActivityType) -> Sport {
        switch t {
        case .running: .run
        case .cycling: .bike
        case .swimming: .swim
        case .walking: .walk
        case .hiking: .hike
        case .rowing: .row
        case .traditionalStrengthTraining, .functionalStrengthTraining, .coreTraining: .strength
        default: .other
        }
    }
}
#endif
