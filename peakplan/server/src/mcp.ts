import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { Store } from "./store.js";
import { ProfilePatch, StepOrRepeat, WorkoutInput } from "./schema.js";
import { describeSteps, hrZonesBpm } from "./metrics.js";

const STEP_FORMAT_HELP = `
Structured steps (rendered as intervals on the Apple Watch):
- step:   {"type":"step","kind":"warmup|work|recovery|rest|cooldown","name"?:string,
           "duration":{"type":"time","seconds":N} | {"type":"distance","meters":N} | {"type":"open"},
           "target"?: <target>, "alertMetric"?: "average"|"current", "notes"?:string}
  alertMetric: whether the Watch alert uses the step's average or the current value. Default: average for
  pace on work steps (athlete preference), current otherwise. Use current for recovery jogs; HR is always current.
- repeat: {"type":"repeat","count":N,"steps":[step,...]}   (repeats cannot be nested)
Targets:
- {"type":"power","unit":"percent_ftp"|"watts","min":88,"max":94}
- {"type":"heart_rate_zone","zone":1-5}
- {"type":"heart_rate","unit":"bpm"|"percent_lthr","min":..,"max":..}
- {"type":"pace","unit":"sec_per_km"|"sec_per_100m"|"percent_threshold","min":..,"max":..}
  (for sec_per_* min is the FASTER pace, e.g. min 270 max 285 = 4:30-4:45/km)
- {"type":"cadence","min":85,"max":95}
Any target may also carry "alertMetric":"average"|"current" (same as the step-level field). Precedence:
step-level alertMetric > target-level alertMetric > default. On save the target value is moved onto the step.
The returned structure shows the alert used for pace/power steps: "(avg alert)" / "(current alert)" when set
explicitly, "(avg alert, default)" / "(current alert, default)" when it comes from the default.
Put the warm-up first and the cool-down last: the Apple Watch shows them as its own Warm Up / Cool Down phases.
Prefer % of FTP / threshold targets so workouts scale automatically when thresholds change.`;

const json = (data: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] });

function summarize(w: any, profile?: { workPaceAlertMetric: "average" | "current" }) {
  return {
    id: w.id,
    date: w.date,
    startTime: w.startTime ?? null,
    title: w.title,
    sport: w.sport,
    status: w.status,
    plannedDurationMin: w.plannedDurationSec ? Math.round(w.plannedDurationSec / 60) : null,
    plannedDistanceKm: w.plannedDistanceM ? +(w.plannedDistanceM / 1000).toFixed(2) : null,
    plannedTss: w.plannedTss,
    structure: describeSteps(w.steps ?? [], profile),
    completedActivityId: w.completedActivityId,
  };
}

export function buildMcpServer(store: Store): McpServer {
  const server = new McpServer(
    { name: "peakplan", version: "1.0.0" },
    {
      instructions:
        "PeakPlan is the athlete's training calendar. Planned workouts sync to their iPhone, Mac and Apple Watch " +
        "(the Watch runs the intervals with alerts). Completed workouts are imported from Apple Health. " +
        "Call get_athlete_profile before writing workouts so targets fit the athlete's thresholds. " +
        "Dates are the athlete's local calendar days (YYYY-MM-DD).",
    },
  );

  // ----- reading -----

  server.registerTool(
    "get_athlete_profile",
    {
      title: "Get athlete profile",
      description:
        "Thresholds (FTP, threshold pace, swim CSS, LTHR), HR zones, weight, goals, weekly availability, shoes, and " +
        "thresholdWarnings when recent work contradicts the profile (stale thresholds distort targets and TSS).",
      annotations: { readOnlyHint: true },
    },
    async () => {
      const p = store.getProfile();
      const warnings = store.thresholdCheck().filter((t) => t.stale);
      return json({ ...p, heartRateZonesBpm: hrZonesBpm(p.lthrBpm), thresholdWarnings: warnings });
    },
  );

  server.registerTool(
    "list_workouts",
    {
      title: "List planned workouts",
      description:
        "Planned workouts on the calendar between two dates (inclusive), with status, planned TSS and a text summary of the intervals. " +
        "Set library=true to list unscheduled template workouts instead.",
      inputSchema: {
        from: z.string().describe("YYYY-MM-DD").optional(),
        to: z.string().describe("YYYY-MM-DD").optional(),
        library: z.boolean().optional(),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ from, to, library }) => json(store.listWorkouts({ from, to, library }).map((w) => summarize(w, store.getProfile()))),
  );

  server.registerTool(
    "get_workout",
    {
      title: "Get workout",
      description: "Full workout including structured steps (JSON), description, coach notes and the linked completed activity.",
      inputSchema: { id: z.string() },
      annotations: { readOnlyHint: true },
    },
    async ({ id }) => {
      const w = store.getWorkout(id);
      if (!w) return { ...json({ error: "not found" }), isError: true };
      const activity = w.completedActivityId ? store.getActivity(w.completedActivityId) : null;
      return json({ ...w, structure: describeSteps(w.steps, store.getProfile()), completedActivity: activity });
    },
  );

  server.registerTool(
    "list_activities",
    {
      title: "List completed activities",
      description:
        "Completed workouts imported from Apple Health (duration, distance, HR, power, TSS, RPE, notes) between two dates, inclusive.",
      inputSchema: { from: z.string().describe("YYYY-MM-DD"), to: z.string().describe("YYYY-MM-DD") },
      annotations: { readOnlyHint: true },
    },
    async ({ from, to }) =>
      json(
        store.listActivities({ from, to }).map((a) => ({
          ...store.enrich(a),
          plannedWorkout: a.workoutId ? summarize(store.getWorkout(a.workoutId) ?? {}, store.getProfile()) : null,
        })),
      ),
  );

  server.registerTool(
    "get_training_summary",
    {
      title: "Training summary",
      description:
        "Weekly planned vs. completed hours/TSS/compliance per week, volume by sport, and current fitness (CTL), fatigue (ATL) and form (TSB).",
      inputSchema: { from: z.string().describe("YYYY-MM-DD"), to: z.string().describe("YYYY-MM-DD") },
      annotations: { readOnlyHint: true },
    },
    async ({ from, to }) => json(store.trainingSummary(from, to)),
  );

  server.registerTool(
    "review_completed_workouts",
    {
      title: "Review completed vs. planned",
      description:
        "Every completed, unplanned, missed or skipped workout between two dates with a TrainingPeaks-style compliance colour " +
        "(green 80-120% of planned duration, yellow 50-150%, orange beyond, red missed/skipped), duration and TSS compliance, RPE and notes.",
      inputSchema: { from: z.string().describe("YYYY-MM-DD"), to: z.string().describe("YYYY-MM-DD") },
      annotations: { readOnlyHint: true },
    },
    async ({ from, to }) =>
      json(
        store.review(from, to).map((i) => ({
          ...i,
          workout: i.workout ? summarize(i.workout, store.getProfile()) : null,
        })),
      ),
  );

  server.registerTool(
    "compare_workout",
    {
      title: "Compare a workout with its plan, interval by interval",
      description:
        "For a planned workout: planned vs. actual totals and each interval's actual avg power / HR / pace / cadence against " +
        "its target. result is about effort (in_range / too_hard / too_easy); deviation is the direction in the metric's own " +
        "terms (pace: too_fast / too_slow; power/HR: too_high / too_low) with deviationPct. Also: execution (over- vs " +
        "under-executed counts), repConsistency (spread fastest–slowest work rep, CV, fade), aerobic decoupling and shoe.",
      inputSchema: { id: z.string().describe("Planned workout id") },
      annotations: { readOnlyHint: true },
    },
    async ({ id }) => json(store.comparison(id)),
  );

  server.registerTool(
    "find_pairing_candidates",
    {
      title: "Find what a workout could be paired with",
      description:
        "Pass activityId to get planned workouts within ±3 days it could fulfil, or workoutId to get completed activities " +
        "within ±3 days that could fulfil that plan. Best matches first (same sport, unpaired, closest day and duration).",
      inputSchema: { activityId: z.string().optional(), workoutId: z.string().optional() },
      annotations: { readOnlyHint: true },
    },
    async ({ activityId, workoutId }) => {
      if (activityId)
        return json(store.workoutCandidates(activityId).map((c) => ({ ...c, workout: summarize(c.workout, store.getProfile()) })));
      if (workoutId) return json(store.activityCandidates(workoutId));
      return { ...json({ error: "pass activityId or workoutId" }), isError: true };
    },
  );

  server.registerTool(
    "check_thresholds",
    {
      title: "Check thresholds against recent work",
      description:
        "Estimates run threshold pace and bike FTP from the fastest sustained efforts of the last 6 weeks (adjusting for " +
        "super shoes) and compares them with the profile. stale=true means the profile is clearly out of date.",
      annotations: { readOnlyHint: true },
    },
    async () => json(store.thresholdCheck()),
  );

  server.registerTool(
    "get_quality_trends",
    {
      title: "Rep consistency & cardiac drift trends",
      description:
        "Per session over time: rep consistency for interval workouts (spread between fastest and slowest work rep, CV, fade, " +
        "over/under-execution) and aerobic decoupling (Pa:HR, first vs second half) for runs/rides of 40+ min, with shoes worn.",
      inputSchema: {
        from: z.string().describe("YYYY-MM-DD"),
        to: z.string().describe("YYYY-MM-DD"),
        sport: z.enum(["run", "bike", "swim", "walk", "hike", "row", "strength", "other"]).optional(),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ from, to, sport }) => json(store.qualityTrends(from, to, sport)),
  );

  // ----- writing -----

  server.registerTool(
    "pair_activity_with_workout",
    {
      title: "Pair a completed activity with a planned workout",
      description:
        "Use when a completed activity was not matched to its plan (done on another day, different sport recorded, or matched " +
        "to the wrong plan). Undoes previous pairings on both sides. workoutId=null unpairs. By default a plan done on a " +
        "different day is moved to the day it was actually done.",
      inputSchema: {
        activityId: z.string(),
        workoutId: z.string().nullable(),
        moveWorkoutToActivityDay: z.boolean().default(true),
      },
    },
    async ({ activityId, workoutId, moveWorkoutToActivityDay }) => {
      const r = store.linkActivity(activityId, workoutId, { moveWorkout: moveWorkoutToActivityDay });
      return json({ activityId: r.activity.id, workout: r.workout ? summarize(r.workout, store.getProfile()) : null });
    },
  );

  server.registerTool(
    "create_workout",
    {
      title: "Create workout",
      description: "Schedule a structured workout on a day (or date=null to save it to the library)." + STEP_FORMAT_HELP,
      inputSchema: WorkoutInput.shape,
    },
    async (input) => json(summarize(store.createWorkout(input), store.getProfile())),
  );

  server.registerTool(
    "update_workout",
    {
      title: "Update workout",
      description:
        "Change any fields of a workout: move it (date), rename it, rewrite its steps, mark it skipped, add coach notes. " +
        "Only the fields you pass change; steps are replaced as a whole." +
        STEP_FORMAT_HELP,
      inputSchema: {
        id: z.string(),
        title: z.string().optional(),
        sport: WorkoutInput.shape.sport.optional(),
        location: z.enum(["outdoor", "indoor"]).optional(),
        date: z.string().nullable().optional(),
        startTime: z.string().nullable().optional(),
        description: z.string().optional(),
        coachNotes: z.string().optional(),
        status: z.enum(["planned", "completed", "skipped"]).optional(),
        steps: z.array(StepOrRepeat).optional(),
      },
    },
    async ({ id, ...patch }) => {
      const clean = Object.fromEntries(Object.entries(patch).filter(([, v]) => v !== undefined));
      return json(summarize(store.updateWorkout(id, clean), store.getProfile()));
    },
  );

  server.registerTool(
    "copy_workout",
    {
      title: "Copy workout",
      description: "Copy an existing workout (e.g. from the library) onto a date.",
      inputSchema: { id: z.string(), date: z.string().describe("YYYY-MM-DD"), startTime: z.string().optional() },
    },
    async ({ id, date, startTime }) => {
      const w = store.getWorkout(id);
      if (!w) return { ...json({ error: "not found" }), isError: true };
      return json(summarize(store.createWorkout({ ...w, date, startTime: startTime ?? w.startTime, status: "planned" }), store.getProfile()));
    },
  );

  server.registerTool(
    "delete_workout",
    {
      title: "Delete workout",
      description: "Remove a planned workout from the calendar.",
      inputSchema: { id: z.string() },
      annotations: { destructiveHint: true },
    },
    async ({ id }) => json({ deleted: store.deleteWorkout(id) }),
  );

  server.registerTool(
    "update_athlete_profile",
    {
      title: "Update athlete profile",
      description:
        "Update thresholds or goals (e.g. after an FTP test). Future workouts' planned TSS/duration are recalculated.",
      inputSchema: ProfilePatch.shape,
    },
    async (patch) => {
      const clean = Object.fromEntries(Object.entries(patch).filter(([, v]) => v !== undefined));
      return json(store.updateProfile(clean));
    },
  );

  server.registerTool(
    "add_activity_feedback",
    {
      title: "Annotate completed activity",
      description: "Record RPE, notes or the shoe worn on a completed activity (use pair_activity_with_workout to change which plan it fulfils).",
      inputSchema: {
        id: z.string(),
        rpe: z.number().min(1).max(10).optional(),
        notes: z.string().optional(),
        workoutId: z.string().optional(),
        shoeId: z.string().nullable().optional().describe("Shoe id from the profile's shoes list"),
      },
    },
    async ({ id, ...patch }) => {
      const clean = Object.fromEntries(Object.entries(patch).filter(([, v]) => v !== undefined));
      return json(store.updateActivity(id, clean));
    },
  );

  // ----- prompts -----

  server.registerPrompt(
    "weekly_review",
    {
      title: "Weekly review & next week plan",
      description: "Review last week's training and build next week's workouts.",
    },
    () => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text:
              "Act as my endurance coach. Use get_athlete_profile, then get_training_summary and review_completed_workouts for the last 4 weeks " +
              "(use compare_workout on the key sessions to check intervals against targets), " +
              "and list_workouts for the next 7 days. Review compliance, load progression (CTL/ATL/TSB) and how I responded. " +
              "Then propose next week's plan, and after I confirm, create or update the workouts with structured intervals.",
          },
        },
      ],
    }),
  );

  return server;
}
