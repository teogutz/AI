import Foundation

/// Client-side estimates for drawing the workout profile chart (the server computes the
/// authoritative planned TSS / duration). Mirrors server/src/metrics.ts.
public enum WorkoutMetrics {
    public struct Segment: Identifiable, Hashable, Sendable {
        public let id: Int
        public let start: Double
        public let duration: Double
        /// Fraction of threshold (1.0 = FTP / threshold pace / LTHR).
        public let intensity: Double
        public let kind: StepKind
        public let label: String
        /// True when the step has no fixed length; drawn with a nominal width.
        public let isOpen: Bool
    }

    static let hrZoneIF: [Int: Double] = [1: 0.6, 2: 0.72, 3: 0.83, 4: 0.93, 5: 1.03]

    static func defaultIF(_ kind: StepKind) -> Double {
        switch kind {
        case .warmup: 0.6
        case .work: 0.8
        case .recovery, .cooldown: 0.55
        case .rest: 0.5
        }
    }

    public static func flatten(_ elements: [WorkoutElement]) -> [WorkoutStepDTO] {
        elements.flatMap { e -> [WorkoutStepDTO] in
            switch e {
            case .step(let s): [s]
            case .repeatBlock(let r): Array(repeating: r.steps, count: max(r.count, 0)).flatMap { $0 }
            }
        }
    }

    public static func intensity(_ s: WorkoutStepDTO, profile: AthleteProfile) -> Double {
        let mid = { (a: Double, b: Double) in (a + b) / 2 }
        guard let t = s.target else { return defaultIF(s.kind) }
        switch t {
        case .power(.percentFtp, let a, let b): return mid(a, b) / 100
        case .power(.watts, let a, let b): return profile.ftpWatts.map { mid(a, b) / $0 } ?? defaultIF(s.kind)
        case .heartRateZone(let z): return hrZoneIF[z] ?? defaultIF(s.kind)
        case .heartRate(.percentLthr, let a, let b): return mid(a, b) / 100
        case .heartRate(.bpm, let a, let b): return profile.lthrBpm.map { mid(a, b) / $0 } ?? defaultIF(s.kind)
        case .pace(.percentThreshold, let a, let b): return mid(a, b) / 100
        case .pace(.secPerKm, let a, let b): return profile.thresholdPaceSecPerKm.map { $0 / mid(a, b) } ?? defaultIF(s.kind)
        case .pace(.secPer100m, let a, let b): return profile.swimCssSecPer100m.map { $0 / mid(a, b) } ?? defaultIF(s.kind)
        case .cadence: return defaultIF(s.kind)
        }
    }

    static func thresholdSpeed(_ sport: Sport, profile: AthleteProfile) -> Double {
        switch sport {
        case .run: 1000 / (profile.thresholdPaceSecPerKm ?? 300)
        case .swim: 100 / (profile.swimCssSecPer100m ?? 120)
        case .bike: 10
        case .walk, .hike: 5.5 / 3.6
        case .row: 500 / 120
        default: 1000 / 360
        }
    }

    public static func seconds(_ s: WorkoutStepDTO, sport: Sport, profile: AthleteProfile) -> Double? {
        switch s.duration {
        case .time(let sec): return sec
        case .open: return nil
        case .distance(let m):
            if case .pace(.secPerKm, let a, let b) = s.target { return m / 1000 * (a + b) / 2 }
            if case .pace(.secPer100m, let a, let b) = s.target { return m / 100 * (a + b) / 2 }
            let ifac = min(1.1, max(0.6, intensity(s, profile: profile)))
            return m / (thresholdSpeed(sport, profile: profile) * ifac)
        }
    }

    public static func segments(_ w: WorkoutDTO, profile: AthleteProfile) -> [Segment] {
        var t = 0.0
        return flatten(w.steps).enumerated().map { i, s in
            let sec = seconds(s, sport: w.sport, profile: profile)
            let seg = Segment(id: i, start: t, duration: sec ?? 300, intensity: intensity(s, profile: profile),
                              kind: s.kind, label: Fmt.step(s), isOpen: sec == nil)
            t += seg.duration
            return seg
        }
    }

    public static func estimate(_ w: WorkoutDTO, profile: AthleteProfile) -> (seconds: Double, tss: Double) {
        let segs = segments(w, profile: profile).filter { !$0.isOpen }
        let sec = segs.reduce(0) { $0 + $1.duration }
        let load = segs.reduce(0) { $0 + $1.duration * $1.intensity * $1.intensity }
        return (sec, (load / 3600 * 100).rounded())
    }
}
