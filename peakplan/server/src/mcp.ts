import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { Store } from "./store.js";
import { ProfilePatch, StepOrRepeat, WorkoutInput } from "./schema.js";
import { describeSteps, hrZonesBpm } from "./metrics.js";

const STEP_FORMAT_HELP = `
Structured steps (rendered as intervals on the Apple Watch):
- step:   {"type":"step","kind":"warmup|work|recovery|rest|cooldown","name"?:string,
           "duration":{"type":"time","seconds":N} | {"type":"distance","meters":N} | {"type":"open"},
           "target"?: <target>, "notes"?:string}
- repeat: {"type":"repeat","count":N,"steps":[step,...]}   (repeats cannot be nested)
Targets:
- {"type":"power","unit":"percent_ftp"|"watts","min":88,"max":94}
- {"type":"heart_rate_zone","zone":1-5}
- {"type":"heart_rate","unit":"bpm"|"percent_lthr","min":..,"max":..}
- {"type":"pace","unit":"sec_per_km"|"sec_per_100m"|"percent_threshold","min":..,"max":..}
  (for sec_per_* min is the FASTER pace, e.g. min 270 max 285 = 4:30-4:45/km)
- {"type":"cadence","min":85,"max":95}
Put the warm-up first and the cool-down last: the Apple Watch shows them as its own Warm Up / Cool Down phases.
Prefer % of FTP / threshold targets so workouts scale automatically when thresholds change.`;

const json = (data: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] });

function summarize(w: any) {
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
    structure: describeSteps(w.steps),
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
      description: "Thresholds (FTP, threshold pace, swim CSS, LTHR), HR zones, weight, goals and weekly availability.",
      annotations: { readOnlyHint: true },
    },
    async () => {
      const p = store.getProfile();
      return json({ ...p, heartRateZonesBpm: hrZonesBpm(p.lthrBpm) });
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
    async ({ from, to, library }) => json(store.listWorkouts({ from, to, library }).map(summarize)),
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
      return json({ ...w, structure: describeSteps(w.steps), completedActivity: activity });
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
          ...a,
          plannedWorkout: a.workoutId ? summarize(store.getWorkout(a.workoutId) ?? {}) : null,
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

  // ----- writing -----

  server.registerTool(
    "create_workout",
    {
      title: "Create workout",
      description: "Schedule a structured workout on a day (or date=null to save it to the library)." + STEP_FORMAT_HELP,
      inputSchema: WorkoutInput.shape,
    },
    async (input) => json(summarize(store.createWorkout(input))),
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
      return json(summarize(store.updateWorkout(id, clean)));
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
      return json(summarize(store.createWorkout({ ...w, date, startTime: startTime ?? w.startTime, status: "planned" })));
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
      description: "Record RPE / notes on a completed activity, or link it to a different planned workout.",
      inputSchema: {
        id: z.string(),
        rpe: z.number().min(1).max(10).optional(),
        notes: z.string().optional(),
        workoutId: z.string().optional(),
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
              "Act as my endurance coach. Use get_athlete_profile, then get_training_summary and list_activities for the last 4 weeks, " +
              "and list_workouts for the next 7 days. Review compliance, load progression (CTL/ATL/TSB) and how I responded. " +
              "Then propose next week's plan, and after I confirm, create or update the workouts with structured intervals.",
          },
        },
      ],
    }),
  );

  return server;
}
