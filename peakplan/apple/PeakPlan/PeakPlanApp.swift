import SwiftUI
import PeakPlanKit
#if os(iOS)
import BackgroundTasks
#endif

@main
struct PeakPlanApp: App {
    @State private var store = TrainingStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(store)
                .task {
                    #if os(iOS)
                    WatchLink.shared.activate()
                    #endif
                    await SyncCoordinator.fullSync(store)
                }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .active { Task { await SyncCoordinator.fullSync(store) } }
                    #if os(iOS)
                    if phase == .background { SyncCoordinator.scheduleBackgroundRefresh() }
                    #endif
                }
        }
        #if os(iOS)
        // Periodically pull changes made by Claude or on the Mac and push them to the Watch,
        // without having to open the app.
        .backgroundTask(.appRefresh(SyncCoordinator.refreshTaskID)) {
            SyncCoordinator.scheduleBackgroundRefresh()
            await SyncCoordinator.fullSync(store)
        }
        #endif
        #if os(macOS)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Refresh") { Task { await SyncCoordinator.fullSync(store) } }
                    .keyboardShortcut("r")
            }
        }
        #endif

        #if os(macOS)
        Settings {
            SettingsView().environment(store).frame(width: 520, height: 640)
        }
        #endif
    }
}

/// One place that pulls the plan, pushes completed workouts and updates the Watch.
@MainActor
enum SyncCoordinator {
    static func fullSync(_ store: TrainingStore) async {
        guard store.config != nil else { return }
        #if os(iOS)
        await importHealth(store)
        #endif
        await store.refresh()
        #if os(iOS)
        await updateWatch(store)
        #endif
    }

    #if os(iOS)
    nonisolated static let refreshTaskID = "com.teogutz.peakplan.refresh"

    nonisolated static func scheduleBackgroundRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: refreshTaskID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 60 * 60)
        try? BGTaskScheduler.shared.submit(request)
    }

    static func importHealth(_ store: TrainingStore) async {
        let importer = HealthImporter()
        let key = "peakplan.uploadedHealthWorkouts.v2" // v2: re-import once with cleaned-up intervals
        let uploaded = Set(UserDefaults.standard.stringArray(forKey: key) ?? [])
        do {
            try await importer.requestAuthorization()
            let new = try await importer.recentActivities(days: 14, excluding: uploaded)
            guard !new.isEmpty, try await store.client?.upload(new) != nil else { return }
            // Remember what was uploaded (bounded) so interval stats are not recomputed every sync.
            let all = Array(uploaded) + new.map(\.externalId)
            UserDefaults.standard.set(Array(all.suffix(500)), forKey: key)
        } catch {
            store.lastError = "Apple Health: \(error.localizedDescription)"
        }
    }

    /// Schedules the upcoming plan in the Apple Watch Workout app and refreshes the Watch app.
    static func updateWatch(_ store: TrainingStore) async {
        let upcoming = store.upcoming(days: 14)
        let failures = await WatchScheduleSync.sync(upcoming, profile: store.profile)
        if let failure = failures.first {
            store.lastError = "\(failure.0.title): \(failure.1.localizedDescription)"
        }
        WatchLink.shared.send(workouts: upcoming, profile: store.profile)
    }
    #endif
}
