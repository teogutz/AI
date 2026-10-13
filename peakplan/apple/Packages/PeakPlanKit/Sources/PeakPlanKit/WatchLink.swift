#if canImport(WatchConnectivity) && !os(macOS)
import Foundation
import WatchConnectivity

/// Sends the upcoming plan from the iPhone to the PeakPlan Watch app (so it can show
/// descriptions, notes and targets offline), and receives it on the Watch.
public final class WatchLink: NSObject, WCSessionDelegate, @unchecked Sendable {
    public struct Snapshot: Codable, Sendable {
        public var workouts: [WorkoutDTO]
        public var profile: AthleteProfile
        public var sentAt: Date
    }

    public static let shared = WatchLink()

    /// Called on the main queue whenever a new snapshot arrives (Watch side).
    public var onSnapshot: (@MainActor (Snapshot) -> Void)?

    public func activate() {
        guard WCSession.isSupported() else { return }
        WCSession.default.delegate = self
        WCSession.default.activate()
    }

    public func send(workouts: [WorkoutDTO], profile: AthleteProfile) {
        guard WCSession.isSupported(), WCSession.default.activationState == .activated else { return }
        #if os(iOS)
        guard WCSession.default.isWatchAppInstalled else { return }
        #endif
        let snapshot = Snapshot(workouts: workouts, profile: profile, sentAt: Date())
        guard let data = try? JSONEncoder().encode(snapshot) else { return }
        try? WCSession.default.updateApplicationContext(["snapshot": data])
    }

    public static func cachedSnapshot() -> Snapshot? {
        guard WCSession.isSupported(),
              let data = WCSession.default.receivedApplicationContext["snapshot"] as? Data else { return nil }
        return try? JSONDecoder().decode(Snapshot.self, from: data)
    }

    // MARK: WCSessionDelegate

    public func session(_ session: WCSession, activationDidCompleteWith state: WCSessionActivationState, error: Error?) {
        // Deliver whatever the phone sent while this app wasn't running.
        #if os(watchOS)
        self.session(session, didReceiveApplicationContext: session.receivedApplicationContext)
        #endif
    }

    public func session(_ session: WCSession, didReceiveApplicationContext context: [String: Any]) {
        guard let data = context["snapshot"] as? Data,
              let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data) else { return }
        Task { @MainActor in self.onSnapshot?(snapshot) }
    }

    #if os(iOS)
    public func sessionDidBecomeInactive(_ session: WCSession) {}
    public func sessionDidDeactivate(_ session: WCSession) { WCSession.default.activate() }
    #endif
}
#endif
