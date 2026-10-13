# PeakPlan: a TrainingPeaks-style training calendar that Claude can coach from

PeakPlan is a personal training planner with four parts:

* **iPhone and Mac apps** for planning and editing structured workouts: a calendar with drag and drop, a workout builder, a template library, and a fitness/fatigue chart.
* **Apple Watch:** each day's workout shows up in Apple's **Workout app** and in the PeakPlan Watch app. You press Start and the Watch runs the intervals, with step changes, target alerts (power, heart rate, pace, cadence), haptics and voice. This is the **WorkoutKit** framework that TrainingPeaks, TrainerRoad and Runna use.
* **Claude integration (MCP):** Claude can read your plan and completed workouts, analyse your load (CTL/ATL/TSB), and create, move or rewrite workouts. Changes appear on your phone, Mac and Watch.
* **Apple Health import:** completed workouts (time, distance, heart rate, power, cadence, calories) are uploaded automatically. They are matched to the planned workout, marked complete and given a TSS, so Claude can compare what you did with what was planned.

```
              Claude (claude.ai / Desktop / Code)
                         │  MCP: /mcp
                         ▼
   ┌───────────── PeakPlan server (Node + SQLite) ─────────────┐
   │ workouts · activities · athlete profile · TSS / CTL / ATL │
   └───────────────▲───────────────────────▲───────────────────┘
          REST /api│                       │REST /api
         ┌─────────┴──────────┐   ┌────────┴───────┐
         │ iPhone app         │   │ Mac app        │
         │ - edits plan       │   │ - edits plan   │
         │ - imports Health   │   └────────────────┘
         │ - WorkoutScheduler ├──► Apple Watch Workout app (intervals, alerts)
         │ - WatchConnectivity├──► PeakPlan Watch app ("Start")
         └────────────────────┘
```

The server is the single source of truth, so edits from Claude, the Mac and the iPhone all go to one place.

---

## 1. Run the server

The server needs Node 22.13 or later and uses no native dependencies.

```bash
cd server
npm install
export PEAKPLAN_TOKEN=$(openssl rand -hex 24)   # keep this secret
npm run build && npm start                       # listens on :8787
npm test                                         # REST + MCP end-to-end tests
```

Claude.ai and your phone need to reach it over **HTTPS**. Pick one:

* **Docker host / Fly.io / Railway / Render:** use `server/Dockerfile`. Mount a volume at `/data` (that's where the SQLite DB lives) and set `PEAKPLAN_TOKEN`.
  ```bash
  fly launch --no-deploy && fly volumes create data --size 1
  fly secrets set PEAKPLAN_TOKEN=...   # then add [mounts] source="data" destination="/data" to fly.toml
  fly deploy
  ```
* **Your Mac at home:** run `npm start`, then expose it with `cloudflared tunnel --url http://localhost:8787` or Tailscale Funnel.

## 2. Connect Claude

| Client | How |
|---|---|
| **Claude.ai / Claude app** (web, iPhone, Mac) | Settings → Connectors → *Add custom connector* → URL `https://YOUR-HOST/mcp/YOUR_TOKEN` |
| **Claude Code** | `claude mcp add --transport http peakplan https://YOUR-HOST/mcp --header "Authorization: Bearer YOUR_TOKEN"` |
| **Claude Desktop, server on the same Mac** | `"peakplan": {"command": "node", "args": ["/path/to/server/dist/stdio.js"], "env": {"PEAKPLAN_DB": "/path/to/server/data/peakplan.db"}}` |

The iPhone/Mac Settings screen shows the connector URL with a Copy button.

**Tools Claude gets:**

| Tool | What it does |
|---|---|
| `get_athlete_profile` / `update_athlete_profile` | FTP, threshold pace, swim CSS, LTHR + HR zones, weight, goals, weekly hours |
| `list_workouts` / `get_workout` | The plan (or the template library), with intervals as text and JSON |
| `create_workout` / `update_workout` / `copy_workout` / `delete_workout` | Build, move, rewrite, skip or remove workouts |
| `list_activities` / `add_activity_feedback` | Completed workouts from Apple Health (with per-step laps); RPE and notes |
| `review_completed_workouts` | Completed / missed / skipped / unplanned sessions with a compliance colour |
| `compare_workout` | Planned vs. actual for one session, interval by interval against each target |
| `get_training_summary` | Weekly planned vs. done (hours, TSS, compliance, by sport) and CTL/ATL/TSB |
| prompt `weekly_review` | "Review my last 4 weeks and build next week" |

**Things to ask Claude:**

* *"Look at my last 3 weeks and tell me whether I'm absorbing the load. Then plan next week: 8 hours, 2 quality bike sessions, a long run Sunday."*
* *"Tomorrow's interval session: I'm tired. Swap it for a Z2 hour and move the intervals to Thursday."*
* *"My FTP test came out at 262 W, so update my profile."* (Workouts in % FTP rescale automatically.)
* *"Build a 12-week half-marathon block ending 15 March, starting at 35 km/week."*

## 3. Build the Apple apps

You need Xcode 15.3 or later and [XcodeGen](https://github.com/yonaskolb/XcodeGen) 2.38 or later (older versions don't embed the Watch app).

```bash
brew install xcodegen
cd apple
xcodegen            # creates PeakPlan.xcodeproj from project.yml
open PeakPlan.xcodeproj
```

1. In `project.yml`, set `DEVELOPMENT_TEAM` and replace `com.example.peakplan` (in the bundle IDs, `WKCompanionAppBundleIdentifier`, `BGTaskSchedulerPermittedIdentifiers` and `SyncCoordinator.refreshTaskID`) with your own prefix. Then run `xcodegen` again.
2. Run **PeakPlan-iOS** on your iPhone. The Watch app installs with it. Run **PeakPlan-macOS** on the Mac.
3. In Settings, enter the server URL and token. On iPhone, **Fill from Apple Health** pre-fills your profile:
   - **Read from Health:** FTP (estimated by the Watch from rides with a power meter), resting HR, weight, VO2max, and your max HR (the highest recorded in the last year, or estimated from your age).
   - **Estimated, marked "Estimate":** threshold HR, from max and resting HR, and threshold pace, from your fastest 20–70 min run in the last 4 months.
   - You choose which values to apply.
4. Accept the Health and Workout-scheduling permissions on the iPhone.

| Platform | What it does |
|---|---|
| **iPhone** | Today, Calendar (drag to move), **Review**, Library, Fitness chart, editor. Every sync it **imports Health workouts → refreshes the plan → schedules the next 14 days on the Watch** (`WorkoutScheduler`; Apple allows 15 at a time) and sends the plan to the Watch app. A background refresh does this roughly hourly, so workouts Claude creates reach the Watch without opening the app. "Preview & send to Apple Watch" opens Apple's own preview sheet. |
| **Mac** | The same calendar and editor in a sidebar layout, with week grids and drag-and-drop between days and from the Library. ⌘R refreshes. Apple doesn't allow scheduling Watch workouts from a Mac, so Mac edits reach the Watch at the iPhone's next sync. |
| **Apple Watch** | Shows today's and upcoming workouts with notes and steps. **Start** hands the workout to the Workout app (`openInWorkoutApp()`), which runs the intervals and saves the result to Health. Scheduled workouts also appear directly in the Workout app, as with TrainingPeaks. |

## Reviewing completed workouts

The **Review** tab (iPhone and Mac) lists every session from the last 1, 4 or 12 weeks, grouped by week with "x/y done · hours done of planned". Each session gets a TrainingPeaks-style colour:

| Colour | Meaning |
|---|---|
| 🟢 Green | 80–120 % of planned duration (distance or TSS if duration is missing) |
| 🟡 Yellow | 50–79 % or 121–150 % |
| 🟠 Orange | Below 50 % or above 150 % |
| 🔴 Red | Missed or skipped |
| ⚪ Grey | Unplanned (no matching plan). You can link it to a planned workout that day. |

Opening a session shows:

* **Totals:** planned vs. completed duration, distance and TSS, with compliance %, plus average HR and power.
* **Intervals:** the Workout app records a segment for each step, and PeakPlan imports each one with its average HR, power, cadence and distance. Every interval is checked against its target (✓ on target, ↓ too easy or slow, ↑ too hard or fast), and a chart shows the target bands with your actual averages.
* **Feedback:** RPE (1–10) and notes. Claude reads these with `review_completed_workouts`, `compare_workout` and `list_activities`.

Missed workouts can be marked skipped or moved to today from the same screen.

Interval matching requires the number of segments the Watch recorded to equal the number of planned steps. That holds when you start the workout from PeakPlan or from the Workout app's scheduled list. Otherwise only totals are compared.

## Workout format

Workouts use one JSON format across the server, apps and Claude. It is defined in `server/src/schema.ts` and mirrored in `PeakPlanKit/Models.swift`:

```json
{
  "title": "Sweet Spot 3x12", "sport": "bike", "location": "indoor",
  "date": "2026-10-14", "startTime": "06:30",
  "description": "Sustained sub-threshold work", "coachNotes": "Cadence 90+, fuel 60 g/h",
  "steps": [
    {"type": "step", "kind": "warmup", "duration": {"type": "time", "seconds": 600},
     "target": {"type": "power", "unit": "percent_ftp", "min": 55, "max": 65}},
    {"type": "repeat", "count": 3, "steps": [
      {"type": "step", "kind": "work", "duration": {"type": "time", "seconds": 720},
       "target": {"type": "power", "unit": "percent_ftp", "min": 88, "max": 94}},
      {"type": "step", "kind": "recovery", "duration": {"type": "time", "seconds": 300},
       "target": {"type": "heart_rate_zone", "zone": 1}}]},
    {"type": "step", "kind": "cooldown", "duration": {"type": "time", "seconds": 600}}
  ]
}
```

* **Durations:** time, distance or open (ends when you press lap).
* **Targets:** power (% FTP or W), heart-rate zone, heart rate (bpm or % LTHR), pace (sec/km, sec/100 m or % threshold) and cadence.
* **Thresholds:** relative targets are turned into absolute Watch alerts using your profile.
* **TSS:** planned TSS uses the standard formula (hours × IF² × 100). Completed TSS uses power for bike and row, and heart rate otherwise.

## Status and known limitations

* **The server is tested.** It has 9 tests covering REST, the MCP round trip, auth, matching activities to workouts, interval comparison, and CTL/ATL.
* **The Swift code has not been compiled yet.** It was written in a Linux environment without Xcode. Expect a few small compile fixes on the first build. The likeliest spots are the WorkoutKit alert factory signatures in `WorkoutKitBridge.swift`.
* **Watch interval support depends on the sport.** Apple's structured workouts cover running, cycling, walking, hiking, swimming and rowing. Strength and "other" workouts still sync to the calendar but can't be sent as intervals.
* **This is a single-user app.** Access is one shared token. Use HTTPS and keep the token private, because the connector URL contains it.
