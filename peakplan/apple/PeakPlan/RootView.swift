import SwiftUI
import PeakPlanKit

enum AppSection: String, CaseIterable, Identifiable {
    case today = "Today", calendar = "Calendar", library = "Library", fitness = "Fitness", settings = "Settings"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .today: "sun.max"
        case .calendar: "calendar"
        case .library: "books.vertical"
        case .fitness: "chart.line.uptrend.xyaxis"
        case .settings: "gearshape"
        }
    }
}

struct RootView: View {
    @Environment(TrainingStore.self) private var store
    @State private var section: AppSection? = .today

    var body: some View {
        Group {
            if store.config == nil {
                NavigationStack { SettingsView() }
            } else {
                #if os(macOS)
                NavigationSplitView {
                    List(AppSection.allCases.filter { $0 != .settings }, selection: $section) { s in
                        Label(s.rawValue, systemImage: s.symbol).tag(s)
                    }
                    .navigationSplitViewColumnWidth(min: 160, ideal: 190)
                } detail: {
                    NavigationStack { content(section ?? .today) }
                }
                #else
                TabView {
                    ForEach(AppSection.allCases) { s in
                        NavigationStack { content(s) }
                            .tabItem { Label(s.rawValue, systemImage: s.symbol) }
                    }
                }
                #endif
            }
        }
        .overlay(alignment: .bottom) { ErrorBanner() }
    }

    @ViewBuilder
    private func content(_ s: AppSection) -> some View {
        switch s {
        case .today: TodayView()
        case .calendar: CalendarView()
        case .library: LibraryView()
        case .fitness: FitnessView()
        case .settings: SettingsView()
        }
    }
}

struct ErrorBanner: View {
    @Environment(TrainingStore.self) private var store

    var body: some View {
        if let error = store.lastError {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                Text(error).font(.footnote).lineLimit(3)
                Spacer()
                Button { store.lastError = nil } label: { Image(systemName: "xmark") }.buttonStyle(.plain)
            }
            .padding(10)
            .background(.orange.opacity(0.9), in: RoundedRectangle(cornerRadius: 10))
            .foregroundStyle(.white)
            .padding()
            .padding(.bottom, 44)
        }
    }
}
