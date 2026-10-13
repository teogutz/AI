// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "PeakPlanKit",
    platforms: [.iOS(.v17), .macOS(.v14), .watchOS(.v10)],
    products: [.library(name: "PeakPlanKit", targets: ["PeakPlanKit"])],
    targets: [
        .target(name: "PeakPlanKit"),
        .testTarget(name: "PeakPlanKitTests", dependencies: ["PeakPlanKit"]),
    ]
)
