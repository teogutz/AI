import XCTest
@testable import PeakPlanKit

final class ModelTests: XCTestCase {
    // Same JSON the server tests use; guards against the Swift and TS schemas drifting apart.
    let json = """
    {"id":"5c7f6a1e-2f1b-4c1a-9a51-0f4b2b1f0c11","title":"Sweet Spot 3x12","sport":"bike","location":"indoor",
     "date":"2026-10-14","description":"","coachNotes":"","status":"planned","plannedTss":70,
     "steps":[
      {"type":"step","kind":"warmup","duration":{"type":"time","seconds":600},"target":{"type":"power","unit":"percent_ftp","min":55,"max":65}},
      {"type":"repeat","count":3,"steps":[
        {"type":"step","kind":"work","duration":{"type":"time","seconds":720},"target":{"type":"power","unit":"percent_ftp","min":88,"max":94}},
        {"type":"step","kind":"recovery","duration":{"type":"time","seconds":300},"target":{"type":"heart_rate_zone","zone":2}}]},
      {"type":"step","kind":"cooldown","duration":{"type":"open"}}]}
    """

    func testDecodeAndRoundTrip() throws {
        let w = try JSONDecoder().decode(WorkoutDTO.self, from: Data(json.utf8))
        XCTAssertEqual(w.steps.count, 3)
        guard case .repeatBlock(let r) = w.steps[1] else { return XCTFail("expected repeat") }
        XCTAssertEqual(r.count, 3)
        XCTAssertEqual(r.steps[0].target, .power(unit: .percentFtp, min: 88, max: 94))
        XCTAssertEqual(r.steps[1].target, .heartRateZone(2))

        let encoded = try JSONEncoder().encode(w.payload)
        let obj = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
        let steps = obj["steps"] as! [[String: Any]]
        XCTAssertEqual(steps[1]["type"] as? String, "repeat")
        XCTAssertNil((steps[0])["id"], "local ids must not be sent")
        let again = try JSONDecoder().decode(WorkoutDTO.self, from: encoded)
        XCTAssertEqual(again.title, w.title)
    }

    func testSegments() throws {
        let w = try JSONDecoder().decode(WorkoutDTO.self, from: Data(json.utf8))
        let segs = WorkoutMetrics.segments(w, profile: AthleteProfile())
        XCTAssertEqual(segs.count, 1 + 3 * 2 + 1)
        XCTAssertEqual(segs[1].intensity, 0.91, accuracy: 0.001)
        XCTAssertEqual(segs[1].start, 600)
    }

    func testTargetResolver() {
        var p = AthleteProfile()
        p.ftpWatts = 250
        p.lthrBpm = 170
        p.thresholdPaceSecPerKm = 240
        let r = TargetResolver(profile: p)
        XCTAssertEqual(r.watts(.power(unit: .percentFtp, min: 90, max: 100)), 225...250)
        let z2 = r.bpm(.heartRateZone(2))!
        XCTAssertEqual(z2.lowerBound, 144.5, accuracy: 0.01)
        XCTAssertEqual(z2.upperBound, 151.3, accuracy: 0.01)
        let s = r.speed(.pace(unit: .secPerKm, min: 250, max: 200))!
        XCTAssertEqual(s.lowerBound, 4, accuracy: 0.001)
        XCTAssertEqual(s.upperBound, 5, accuracy: 0.001)
    }
}

final class PayloadTests: XCTestCase {
    func testNilDateIsSentAsNull() throws {
        let data = try JSONEncoder().encode(WorkoutDTO(title: "Template", date: nil).payload)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertTrue(obj["date"] is NSNull)
        XCTAssertTrue(obj["startTime"] is NSNull)
    }
}
