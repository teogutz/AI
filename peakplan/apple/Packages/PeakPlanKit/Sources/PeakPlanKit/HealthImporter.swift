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
        return types
    }

    public func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        try await store.requestAuthorization(toShare: [], read: readTypes)
    }

    public func recentActivities(days: Int = 30) async throws -> [ActivityDTO] {
        let start = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: start, end: nil)
        let descriptor = HKSampleQueryDescriptor(
            predicates: [.workout(predicate)],
            sortDescriptors: [SortDescriptor(\.startDate, order: .reverse)],
            limit: 500
        )
        return try await descriptor.result(for: store).map(activity(from:))
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
