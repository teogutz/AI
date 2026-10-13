import Foundation

public struct ServerConfig: Codable, Hashable, Sendable {
    public var baseURL: URL
    public var token: String

    public init(baseURL: URL, token: String) {
        self.baseURL = baseURL
        self.token = token
    }

    private static let key = "peakplan.server"

    public static func load(from defaults: UserDefaults = .standard) -> ServerConfig? {
        guard let data = defaults.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(ServerConfig.self, from: data)
    }

    public func save(to defaults: UserDefaults = .standard) {
        defaults.set(try? JSONEncoder().encode(self), forKey: Self.key)
    }
}

public enum APIError: LocalizedError {
    case http(Int, String)
    public var errorDescription: String? {
        switch self {
        case .http(let code, let body): "Server error \(code): \(body)"
        }
    }
}

public struct APIClient: Sendable {
    public let config: ServerConfig
    public init(config: ServerConfig) { self.config = config }

    private func json(_ value: some Encodable) throws -> Data { try JSONEncoder().encode(value) }

    private func request(_ method: String, _ path: String, query: [String: String] = [:], body: Data? = nil) async throws -> Data {
        var comps = URLComponents(url: config.baseURL.appending(path: "api").appending(path: path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { comps.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) } }
        var req = URLRequest(url: comps.url!)
        req.httpMethod = method
        req.setValue("Bearer \(config.token)", forHTTPHeaderField: "Authorization")
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = body
        }
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else { throw APIError.http(code, String(decoding: data, as: UTF8.self)) }
        return data
    }

    private func decode<T: Decodable>(_ data: Data) throws -> T { try JSONDecoder().decode(T.self, from: data) }

    public func workouts(from: String, to: String) async throws -> [WorkoutDTO] {
        try decode(await request("GET", "workouts", query: ["from": from, "to": to]))
    }

    public func library() async throws -> [WorkoutDTO] {
        try decode(await request("GET", "workouts", query: ["library": "true"]))
    }

    /// Creates or updates (the server upserts by id).
    public func save(_ workout: WorkoutDTO) async throws -> WorkoutDTO {
        try decode(await request("POST", "workouts", body: json(workout.payload)))
    }

    public func deleteWorkout(id: String) async throws {
        _ = try await request("DELETE", "workouts/\(id)")
    }

    public func profile() async throws -> AthleteProfile {
        try decode(await request("GET", "profile"))
    }

    public func saveProfile(_ p: AthleteProfile) async throws -> AthleteProfile {
        try decode(await request("PUT", "profile", body: json(p)))
    }

    public func activities(from: String, to: String) async throws -> [ActivityDTO] {
        try decode(await request("GET", "activities", query: ["from": from, "to": to]))
    }

    @discardableResult
    public func upload(_ activities: [ActivityDTO]) async throws -> [ActivityDTO] {
        try decode(await request("POST", "activities", body: json(activities)))
    }
}
