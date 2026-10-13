import Foundation

public enum Fmt {
    public static func duration(_ seconds: Double?) -> String {
        guard let seconds, seconds > 0 else { return "–" }
        let s = Int(seconds.rounded())
        let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
        if h > 0 { return String(format: "%d:%02d:%02d", h, m, sec) }
        return sec == 0 ? "\(m) min" : String(format: "%d:%02d", m, sec)
    }

    public static func distance(_ meters: Double?) -> String {
        guard let meters, meters > 0 else { return "–" }
        return meters >= 1000 ? String(format: "%.2f km", meters / 1000) : "\(Int(meters)) m"
    }

    public static func pace(_ secPerUnit: Double) -> String {
        let s = Int(secPerUnit.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    public static func stepDuration(_ d: StepDuration) -> String {
        switch d {
        case .time(let s): duration(s)
        case .distance(let m): distance(m)
        case .open: "Open"
        }
    }

    public static func target(_ t: StepTarget?) -> String {
        guard let t else { return "" }
        let r = { (a: Double, b: Double) in "\(Int(a))–\(Int(b))" }
        switch t {
        case .power(.percentFtp, let a, let b): return "\(r(a, b))% FTP"
        case .power(.watts, let a, let b): return "\(r(a, b)) W"
        case .heartRateZone(let z): return "HR Zone \(z)"
        case .heartRate(.bpm, let a, let b): return "\(r(a, b)) bpm"
        case .heartRate(.percentLthr, let a, let b): return "\(r(a, b))% LTHR"
        case .pace(.secPerKm, let a, let b): return "\(pace(a))–\(pace(b)) /km"
        case .pace(.secPer100m, let a, let b): return "\(pace(a))–\(pace(b)) /100m"
        case .pace(.percentThreshold, let a, let b): return "\(r(a, b))% thr. pace"
        case .cadence(let a, let b): return "\(r(a, b)) rpm"
        }
    }

    public static func step(_ s: WorkoutStepDTO) -> String {
        let t = target(s.target)
        return "\(s.name ?? s.kind.displayName) · \(stepDuration(s.duration))" + (t.isEmpty ? "" : " @ \(t)")
    }
}

/// Converts relative targets (% FTP, % LTHR, % threshold, HR zones) into absolute values
/// using the athlete profile. Needed because the Apple Watch alerts take absolute ranges.
public struct TargetResolver: Sendable {
    public let profile: AthleteProfile
    public init(profile: AthleteProfile) { self.profile = profile }

    // Friel zones as % of LTHR — keep in sync with server/src/metrics.ts.
    static let hrZonesPctLthr: [Int: ClosedRange<Double>] = [1: 50...84, 2: 85...89, 3: 90...94, 4: 95...99, 5: 100...106]

    public func watts(_ t: StepTarget) -> ClosedRange<Double>? {
        switch t {
        case .power(.watts, let a, let b): return ordered(a, b)
        case .power(.percentFtp, let a, let b):
            guard let ftp = profile.ftpWatts else { return nil }
            return ordered(a * ftp / 100, b * ftp / 100)
        default: return nil
        }
    }

    public func bpm(_ t: StepTarget) -> ClosedRange<Double>? {
        switch t {
        case .heartRate(.bpm, let a, let b): return ordered(a, b)
        case .heartRate(.percentLthr, let a, let b):
            guard let lthr = profile.lthrBpm else { return nil }
            return ordered(a * lthr / 100, b * lthr / 100)
        case .heartRateZone(let z):
            guard let lthr = profile.lthrBpm, let pct = Self.hrZonesPctLthr[z] else { return nil }
            return ordered(pct.lowerBound * lthr / 100, pct.upperBound * lthr / 100)
        default: return nil
        }
    }

    /// Speed range in meters/second.
    public func speed(_ t: StepTarget) -> ClosedRange<Double>? {
        guard case .pace(let unit, let a, let b) = t else { return nil }
        switch unit {
        case .secPerKm: return ordered(1000 / b, 1000 / a)
        case .secPer100m: return ordered(100 / b, 100 / a)
        case .percentThreshold:
            guard let thr = profile.thresholdPaceSecPerKm else { return nil }
            let v = 1000 / thr
            return ordered(v * a / 100, v * b / 100)
        }
    }

    private func ordered(_ a: Double, _ b: Double) -> ClosedRange<Double> { Swift.min(a, b)...Swift.max(a, b) }
}
