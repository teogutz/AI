import Foundation

// Mirrors server/src/schema.ts. JSON keys and enum raw values must match.

public enum Sport: String, Codable, CaseIterable, Identifiable, Sendable {
    case run, bike, swim, walk, hike, row, strength, other
    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .run: "Run"
        case .bike: "Bike"
        case .swim: "Swim"
        case .walk: "Walk"
        case .hike: "Hike"
        case .row: "Row"
        case .strength: "Strength"
        case .other: "Other"
        }
    }

    public var symbol: String {
        switch self {
        case .run: "figure.run"
        case .bike: "figure.outdoor.cycle"
        case .swim: "figure.pool.swim"
        case .walk: "figure.walk"
        case .hike: "figure.hiking"
        case .row: "figure.rower"
        case .strength: "dumbbell"
        case .other: "figure.mixed.cardio"
        }
    }
}

public enum WorkoutLocation: String, Codable, CaseIterable, Sendable { case outdoor, indoor }
public enum WorkoutStatus: String, Codable, CaseIterable, Sendable { case planned, completed, skipped }

public enum StepKind: String, Codable, CaseIterable, Identifiable, Sendable {
    case warmup, work, recovery, rest, cooldown
    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .warmup: "Warm Up"
        case .work: "Work"
        case .recovery: "Recovery"
        case .rest: "Rest"
        case .cooldown: "Cool Down"
        }
    }
}

public enum StepDuration: Codable, Hashable, Sendable {
    case time(seconds: Double)
    case distance(meters: Double)
    case open

    private enum CodingKeys: String, CodingKey { case type, seconds, meters }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(String.self, forKey: .type) {
        case "time": self = .time(seconds: try c.decode(Double.self, forKey: .seconds))
        case "distance": self = .distance(meters: try c.decode(Double.self, forKey: .meters))
        default: self = .open
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .time(let s):
            try c.encode("time", forKey: .type)
            try c.encode(s, forKey: .seconds)
        case .distance(let m):
            try c.encode("distance", forKey: .type)
            try c.encode(m, forKey: .meters)
        case .open:
            try c.encode("open", forKey: .type)
        }
    }
}

public enum PowerUnit: String, Codable, CaseIterable, Sendable { case percentFtp = "percent_ftp", watts }
public enum HeartRateUnit: String, Codable, CaseIterable, Sendable { case bpm, percentLthr = "percent_lthr" }
public enum PaceUnit: String, Codable, CaseIterable, Sendable {
    /// For sec/km and sec/100m, `min` is the faster pace (smaller number).
    case secPerKm = "sec_per_km", secPer100m = "sec_per_100m", percentThreshold = "percent_threshold"
}

public enum StepTarget: Codable, Hashable, Sendable {
    case power(unit: PowerUnit, min: Double, max: Double)
    case heartRateZone(Int)
    case heartRate(unit: HeartRateUnit, min: Double, max: Double)
    case pace(unit: PaceUnit, min: Double, max: Double)
    case cadence(min: Double, max: Double)

    private enum CodingKeys: String, CodingKey { case type, unit, min, max, zone }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .type)
        if type == "heart_rate_zone" {
            self = .heartRateZone(try c.decode(Int.self, forKey: .zone))
            return
        }
        let lo = try c.decode(Double.self, forKey: .min)
        let hi = try c.decode(Double.self, forKey: .max)
        switch type {
        case "power": self = .power(unit: try c.decode(PowerUnit.self, forKey: .unit), min: lo, max: hi)
        case "heart_rate": self = .heartRate(unit: try c.decode(HeartRateUnit.self, forKey: .unit), min: lo, max: hi)
        case "pace": self = .pace(unit: try c.decode(PaceUnit.self, forKey: .unit), min: lo, max: hi)
        case "cadence": self = .cadence(min: lo, max: hi)
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: c, debugDescription: "Unknown target \(type)")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .power(let unit, let lo, let hi):
            try c.encode("power", forKey: .type)
            try c.encode(unit, forKey: .unit)
            try c.encode(lo, forKey: .min)
            try c.encode(hi, forKey: .max)
        case .heartRateZone(let z):
            try c.encode("heart_rate_zone", forKey: .type)
            try c.encode(z, forKey: .zone)
        case .heartRate(let unit, let lo, let hi):
            try c.encode("heart_rate", forKey: .type)
            try c.encode(unit, forKey: .unit)
            try c.encode(lo, forKey: .min)
            try c.encode(hi, forKey: .max)
        case .pace(let unit, let lo, let hi):
            try c.encode("pace", forKey: .type)
            try c.encode(unit, forKey: .unit)
            try c.encode(lo, forKey: .min)
            try c.encode(hi, forKey: .max)
        case .cadence(let lo, let hi):
            try c.encode("cadence", forKey: .type)
            try c.encode(lo, forKey: .min)
            try c.encode(hi, forKey: .max)
        }
    }
}

public struct WorkoutStepDTO: Codable, Hashable, Identifiable, Sendable {
    /// Local-only identity for SwiftUI lists; not sent to the server.
    public var id = UUID()
    public var kind: StepKind
    public var name: String?
    public var duration: StepDuration
    public var target: StepTarget?
    public var notes: String?

    public init(kind: StepKind, name: String? = nil, duration: StepDuration, target: StepTarget? = nil, notes: String? = nil) {
        self.kind = kind
        self.name = name
        self.duration = duration
        self.target = target
        self.notes = notes
    }

    private enum CodingKeys: String, CodingKey { case type, kind, name, duration, target, notes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = try c.decode(StepKind.self, forKey: .kind)
        name = try c.decodeIfPresent(String.self, forKey: .name)
        duration = try c.decode(StepDuration.self, forKey: .duration)
        target = try c.decodeIfPresent(StepTarget.self, forKey: .target)
        notes = try c.decodeIfPresent(String.self, forKey: .notes)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode("step", forKey: .type)
        try c.encode(kind, forKey: .kind)
        try c.encodeIfPresent(name, forKey: .name)
        try c.encode(duration, forKey: .duration)
        try c.encodeIfPresent(target, forKey: .target)
        try c.encodeIfPresent(notes, forKey: .notes)
    }
}

public struct RepeatBlockDTO: Codable, Hashable, Identifiable, Sendable {
    public var id = UUID()
    public var count: Int
    public var steps: [WorkoutStepDTO]

    public init(count: Int, steps: [WorkoutStepDTO]) {
        self.count = count
        self.steps = steps
    }

    private enum CodingKeys: String, CodingKey { case type, count, steps }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        count = try c.decode(Int.self, forKey: .count)
        steps = try c.decode([WorkoutStepDTO].self, forKey: .steps)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode("repeat", forKey: .type)
        try c.encode(count, forKey: .count)
        try c.encode(steps, forKey: .steps)
    }
}

public enum WorkoutElement: Codable, Hashable, Identifiable, Sendable {
    case step(WorkoutStepDTO)
    case repeatBlock(RepeatBlockDTO)

    public var id: UUID {
        switch self {
        case .step(let s): s.id
        case .repeatBlock(let r): r.id
        }
    }

    private enum CodingKeys: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if try c.decode(String.self, forKey: .type) == "repeat" {
            self = .repeatBlock(try RepeatBlockDTO(from: decoder))
        } else {
            self = .step(try WorkoutStepDTO(from: decoder))
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .step(let s): try s.encode(to: encoder)
        case .repeatBlock(let r): try r.encode(to: encoder)
        }
    }
}

public struct WorkoutDTO: Codable, Hashable, Identifiable, Sendable {
    public var id: String
    public var title: String
    public var sport: Sport
    public var location: WorkoutLocation
    /// Local calendar day "YYYY-MM-DD"; nil means it lives in the library.
    public var date: String?
    /// Optional "HH:MM" local start time.
    public var startTime: String?
    public var description: String
    public var coachNotes: String
    public var status: WorkoutStatus
    public var steps: [WorkoutElement]
    public var plannedDurationSec: Double?
    public var plannedDistanceM: Double?
    public var plannedTss: Double?
    public var completedActivityId: String?
    public var updatedAt: String?

    public init(
        id: String = UUID().uuidString.lowercased(),
        title: String = "New Workout",
        sport: Sport = .run,
        location: WorkoutLocation = .outdoor,
        date: String? = nil,
        startTime: String? = nil,
        description: String = "",
        coachNotes: String = "",
        status: WorkoutStatus = .planned,
        steps: [WorkoutElement] = []
    ) {
        self.id = id
        self.title = title
        self.sport = sport
        self.location = location
        self.date = date
        self.startTime = startTime
        self.description = description
        self.coachNotes = coachNotes
        self.status = status
        self.steps = steps
    }

    /// Fields the server accepts on create / update.
    public struct Payload: Encodable {
        let id: String, title: String, sport: Sport, location: WorkoutLocation
        let date: String?, startTime: String?, description: String, coachNotes: String
        let status: WorkoutStatus, steps: [WorkoutElement]

        private enum CodingKeys: String, CodingKey {
            case id, title, sport, location, date, startTime, description, coachNotes, status, steps
        }

        // date / startTime are sent as explicit nulls so they can be cleared (server merges updates).
        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(id, forKey: .id)
            try c.encode(title, forKey: .title)
            try c.encode(sport, forKey: .sport)
            try c.encode(location, forKey: .location)
            try c.encode(date, forKey: .date)
            try c.encode(startTime, forKey: .startTime)
            try c.encode(description, forKey: .description)
            try c.encode(coachNotes, forKey: .coachNotes)
            try c.encode(status, forKey: .status)
            try c.encode(steps, forKey: .steps)
        }
    }

    public var payload: Payload {
        Payload(id: id, title: title, sport: sport, location: location, date: date, startTime: startTime,
                description: description, coachNotes: coachNotes, status: status, steps: steps)
    }
}

public struct ActivityDTO: Codable, Hashable, Identifiable, Sendable {
    public var id: String?
    public var externalId: String
    public var source: String
    public var sport: Sport
    public var title: String
    public var startDate: String
    public var durationSec: Double
    public var distanceM: Double?
    public var elevationGainM: Double?
    public var avgHr: Double?
    public var maxHr: Double?
    public var avgPower: Double?
    public var normalizedPower: Double?
    public var avgCadence: Double?
    public var energyKcal: Double?
    public var rpe: Double?
    public var notes: String
    public var workoutId: String?
    public var tss: Double?
    /// One per workout step / segment recorded by the Watch.
    public var laps: [LapDTO]?

    public init(externalId: String, sport: Sport, title: String, startDate: String, durationSec: Double) {
        self.externalId = externalId
        self.source = "healthkit"
        self.sport = sport
        self.title = title
        self.startDate = startDate
        self.durationSec = durationSec
        self.notes = ""
    }
}

public struct LapDTO: Codable, Hashable, Sendable {
    public var startOffsetSec: Double
    public var durationSec: Double
    public var distanceM: Double?
    public var avgHr: Double?
    public var avgPower: Double?
    public var avgCadence: Double?

    public init(startOffsetSec: Double, durationSec: Double) {
        self.startOffsetSec = startOffsetSec
        self.durationSec = durationSec
    }
}

// MARK: - Review (planned vs. completed), computed by the server

public enum ComplianceColor: String, Codable, Sendable { case green, yellow, orange, red }

public struct ReviewItemDTO: Codable, Hashable, Identifiable, Sendable {
    /// manual = marked completed by hand, with no recorded activity paired to it.
    public enum State: String, Codable, Sendable { case completed, unplanned, missed, skipped, manual }
    public var date: String
    public var state: State
    public var color: ComplianceColor?
    public var durationPct: Double?
    public var tssPct: Double?
    public var workout: WorkoutDTO?
    public var activity: ActivityDTO?

    public var id: String { activity?.id ?? workout?.id ?? date }

    public init(date: String, state: State, color: ComplianceColor? = nil, durationPct: Double? = nil,
                tssPct: Double? = nil, workout: WorkoutDTO? = nil, activity: ActivityDTO? = nil) {
        self.date = date
        self.state = state
        self.color = color
        self.durationPct = durationPct
        self.tssPct = tssPct
        self.workout = workout
        self.activity = activity
    }
}

// MARK: - Pairing completed activities with planned workouts

/// A planned workout an activity could be paired with. dayOffset = plan day − activity day.
public struct WorkoutCandidateDTO: Codable, Hashable, Identifiable, Sendable {
    public var workout: WorkoutDTO
    public var dayOffset: Int
    public var sameSport: Bool
    public var pairedActivityId: String?
    public var id: String { workout.id }
}

/// A completed activity that could fulfil a plan. dayOffset = activity day − plan day.
public struct ActivityCandidateDTO: Codable, Hashable, Identifiable, Sendable {
    public var activity: ActivityDTO
    public var dayOffset: Int
    public var sameSport: Bool
    public var pairedWorkoutId: String?
    public var id: String { activity.id ?? activity.externalId }
}

public struct LinkResultDTO: Codable, Sendable {
    public var activity: ActivityDTO
    public var workout: WorkoutDTO?
}

// MARK: - Server status

public struct ServerStatusDTO: Codable, Sendable {
    public struct Host: Codable, Sendable {
        public struct Disk: Codable, Sendable { public var totalBytes: Double; public var freeBytes: Double }
        public var loadAvg1m: Double
        public var memTotalBytes: Double
        public var memFreeBytes: Double
        public var disk: Disk?
    }
    public struct Database: Codable, Sendable {
        public var sizeBytes: Double
        public var workouts: Int
        public var upcomingWorkouts: Int
        public var activities: Int
        public var lastWorkoutChange: String?
        public var lastActivity: String?
        public var lastImport: String?
    }
    public struct Backups: Codable, Sendable {
        public struct Latest: Codable, Sendable { public var file: String; public var sizeBytes: Double; public var modified: String }
        public var count: Int
        public var latest: Latest?
    }
    public var ok: Bool
    public var version: String
    public var node: String
    public var startedAt: String
    public var uptimeSec: Double
    public var host: Host
    public var database: Database
    public var backups: Backups
}

public struct ComparisonDTO: Codable, Hashable, Sendable {
    public struct Totals: Codable, Hashable, Sendable {
        public struct Planned: Codable, Hashable, Sendable {
            public var durationSec: Double?
            public var distanceM: Double?
            public var tss: Double?
        }
        public struct Actual: Codable, Hashable, Sendable {
            public var durationSec: Double
            public var distanceM: Double?
            public var tss: Double?
            public var avgHr: Double?
            public var avgPower: Double?
            public var rpe: Double?
        }
        public var planned: Planned
        public var actual: Actual?
        public var durationPct: Double?
        public var distancePct: Double?
        public var tssPct: Double?
    }

    public struct Target: Codable, Hashable, Sendable {
        public var metric: String // power | heart_rate | pace | cadence
        public var low: Double
        public var high: Double
        public var unit: String
    }

    public struct Step: Codable, Hashable, Identifiable, Sendable {
        public enum Result: String, Codable, Sendable {
            case inRange = "in_range", below, above
        }
        public var index: Int
        public var kind: StepKind
        public var name: String?
        public var plannedDurationSec: Double?
        public var plannedDistanceM: Double?
        public var target: Target?
        public var actualDurationSec: Double?
        public var actualDistanceM: Double?
        public var actualAvgHr: Double?
        public var actualAvgPower: Double?
        public var actualValue: Double?
        public var result: Result?
        public var id: Int { index }
    }

    public var workoutId: String
    public var activityId: String?
    public var color: ComplianceColor
    public var totals: Totals
    public var intervalsMatched: Bool
    public var recordedSegments: Int?
    public var stepsNotDone: Int?
    /// Segments recorded after the plan finished (e.g. kept running after the cool-down).
    public var extraLaps: [LapDTO]?
    public var intervalsInRangePct: Double?
    public var steps: [Step]
}

public struct AthleteProfile: Codable, Hashable, Sendable {
    public var name: String = "Athlete"
    public var ftpWatts: Double?
    public var thresholdPaceSecPerKm: Double?
    public var swimCssSecPer100m: Double?
    public var lthrBpm: Double?
    public var maxHrBpm: Double?
    public var restingHrBpm: Double?
    public var weightKg: Double?
    public var goals: String = ""
    public var weeklyHoursAvailable: Double?
    public var vo2Max: Double?

    public init() {}

    // Tolerant decoding so older caches / partial JSON don't fail.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "Athlete"
        ftpWatts = try c.decodeIfPresent(Double.self, forKey: .ftpWatts)
        thresholdPaceSecPerKm = try c.decodeIfPresent(Double.self, forKey: .thresholdPaceSecPerKm)
        swimCssSecPer100m = try c.decodeIfPresent(Double.self, forKey: .swimCssSecPer100m)
        lthrBpm = try c.decodeIfPresent(Double.self, forKey: .lthrBpm)
        maxHrBpm = try c.decodeIfPresent(Double.self, forKey: .maxHrBpm)
        restingHrBpm = try c.decodeIfPresent(Double.self, forKey: .restingHrBpm)
        weightKg = try c.decodeIfPresent(Double.self, forKey: .weightKg)
        goals = try c.decodeIfPresent(String.self, forKey: .goals) ?? ""
        weeklyHoursAvailable = try c.decodeIfPresent(Double.self, forKey: .weeklyHoursAvailable)
        vo2Max = try c.decodeIfPresent(Double.self, forKey: .vo2Max)
    }
}

// MARK: - Calendar day helpers

public enum Day {
    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()

    public static func string(_ date: Date) -> String { formatter.string(from: date) }
    public static func date(_ string: String) -> Date? { formatter.date(from: string) }
    public static var today: String { string(Date()) }

    public static func adding(_ days: Int, to day: String) -> String {
        guard let d = date(day), let n = Calendar.current.date(byAdding: .day, value: days, to: d) else { return day }
        return string(n)
    }
}
