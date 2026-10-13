import Foundation
import Observation

/// App-wide state shared by the iPhone and Mac apps. The server is the source of truth
/// (so edits from Claude, the Mac and the iPhone all meet in one place); a JSON cache keeps
/// the last known calendar available offline.
@MainActor
@Observable
public final class TrainingStore {
    public var config: ServerConfig? {
        didSet { config?.save() }
    }
    public private(set) var workouts: [String: WorkoutDTO] = [:]
    public private(set) var activities: [ActivityDTO] = []
    public var profile = AthleteProfile()
    public var isLoading = false
    public var lastError: String?
    public var lastSync: Date?

    public init() {
        config = ServerConfig.load()
        loadCache()
    }

    private var api: APIClient? { config.map(APIClient.init) }

    // MARK: Queries

    public func workouts(on day: String) -> [WorkoutDTO] {
        workouts.values.filter { $0.date == day }.sorted { ($0.startTime ?? "99", $0.title) < ($1.startTime ?? "99", $1.title) }
    }

    public var library: [WorkoutDTO] {
        workouts.values.filter { $0.date == nil }.sorted { $0.title < $1.title }
    }

    public func activities(on day: String) -> [ActivityDTO] {
        activities.filter { $0.startDate.hasPrefix(day) && $0.workoutId == nil }
    }

    public func activity(id: String?) -> ActivityDTO? {
        guard let id else { return nil }
        return activities.first { $0.id == id }
    }

    public func upcoming(days: Int = 14) -> [WorkoutDTO] {
        let today = Day.today, end = Day.adding(days, to: today)
        return workouts.values
            .filter { ($0.date ?? "") >= today && ($0.date ?? "") <= end }
            .sorted { ($0.date ?? "", $0.startTime ?? "") < ($1.date ?? "", $1.startTime ?? "") }
    }

    // MARK: Sync

    /// Loads a window around today: past weeks for review, future weeks for the plan.
    public func refresh(pastDays: Int = 120, futureDays: Int = 42) async {
        guard let api else { return }
        isLoading = true
        defer { isLoading = false }
        let from = Day.adding(-pastDays, to: Day.today), to = Day.adding(futureDays, to: Day.today)
        do {
            async let w = api.workouts(from: from, to: to)
            async let lib = api.library()
            async let a = api.activities(from: from, to: to)
            async let p = api.profile()
            let (ws, ls, acts, prof) = try await (w, lib, a, p)
            workouts = Dictionary((ws + ls).map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
            activities = acts
            profile = prof
            lastSync = Date()
            lastError = nil
            saveCache()
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func save(_ workout: WorkoutDTO) async {
        workouts[workout.id] = workout // optimistic
        guard let api else { return }
        do {
            workouts[workout.id] = try await api.save(workout)
            saveCache()
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func move(_ workout: WorkoutDTO, to day: String?) async {
        var w = workout
        w.date = day
        await save(w)
    }

    /// Duplicates a workout (e.g. from the library) onto a day.
    public func copy(_ workout: WorkoutDTO, to day: String) async {
        var w = workout
        w.id = UUID().uuidString.lowercased()
        w.date = day
        w.status = .planned
        w.completedActivityId = nil
        await save(w)
    }

    public func delete(_ workout: WorkoutDTO) async {
        workouts[workout.id] = nil
        do {
            try await api?.deleteWorkout(id: workout.id)
            saveCache()
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func saveProfile() async {
        guard let api else { return }
        do {
            profile = try await api.saveProfile(profile)
            await refresh() // planned TSS changes with thresholds
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func upload(_ newActivities: [ActivityDTO]) async {
        guard let api, !newActivities.isEmpty else { return }
        do {
            try await api.upload(newActivities)
        } catch {
            lastError = error.localizedDescription
        }
    }

    // MARK: Cache

    private struct Cache: Codable {
        var workouts: [WorkoutDTO]
        var activities: [ActivityDTO]
        var profile: AthleteProfile
    }

    private var cacheURL: URL {
        URL.applicationSupportDirectory.appending(path: "peakplan-cache.json")
    }

    private func saveCache() {
        let cache = Cache(workouts: Array(workouts.values), activities: activities, profile: profile)
        try? FileManager.default.createDirectory(at: .applicationSupportDirectory, withIntermediateDirectories: true)
        try? JSONEncoder().encode(cache).write(to: cacheURL, options: .atomic)
    }

    private func loadCache() {
        guard let data = try? Data(contentsOf: cacheURL),
              let cache = try? JSONDecoder().decode(Cache.self, from: data) else { return }
        workouts = Dictionary(cache.workouts.map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        activities = cache.activities
        profile = cache.profile
    }
}
