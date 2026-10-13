import { z } from "zod";

// Shared workout format. The Swift models in apple/Packages/PeakPlanKit mirror
// these shapes exactly, so keep the two in sync.

export const Sport = z.enum(["run", "bike", "swim", "walk", "hike", "row", "strength", "other"]);
export const Location = z.enum(["outdoor", "indoor"]);
export const StepKind = z.enum(["warmup", "work", "recovery", "rest", "cooldown"]);
export const WorkoutStatus = z.enum(["planned", "completed", "skipped"]);

export const Duration = z.discriminatedUnion("type", [
  z.object({ type: z.literal("time"), seconds: z.number().positive() }),
  z.object({ type: z.literal("distance"), meters: z.number().positive() }),
  z.object({ type: z.literal("open") }),
]);

export const Target = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("power"),
    unit: z.enum(["percent_ftp", "watts"]),
    min: z.number().positive(),
    max: z.number().positive(),
  }),
  z.object({ type: z.literal("heart_rate_zone"), zone: z.number().int().min(1).max(5) }),
  z.object({
    type: z.literal("heart_rate"),
    unit: z.enum(["bpm", "percent_lthr"]),
    min: z.number().positive(),
    max: z.number().positive(),
  }),
  z.object({
    type: z.literal("pace"),
    // sec_per_km / sec_per_100m: min is the FASTER pace (smaller number).
    // percent_threshold: % of threshold speed (e.g. 95-100 = slightly below threshold).
    unit: z.enum(["sec_per_km", "sec_per_100m", "percent_threshold"]),
    min: z.number().positive(),
    max: z.number().positive(),
  }),
  z.object({ type: z.literal("cadence"), min: z.number().positive(), max: z.number().positive() }),
]);

export const Step = z.object({
  type: z.literal("step"),
  kind: StepKind,
  name: z.string().optional(),
  duration: Duration,
  target: Target.optional(),
  notes: z.string().optional(),
});

export const Repeat = z.object({
  type: z.literal("repeat"),
  count: z.number().int().min(1).max(99),
  steps: z.array(Step).min(1),
});

export const StepOrRepeat = z.discriminatedUnion("type", [Step, Repeat]);

const isoDate = z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "Use YYYY-MM-DD");

export const WorkoutInput = z.object({
  title: z.string().min(1),
  sport: Sport,
  location: Location.default("outdoor"),
  date: isoDate.nullable().describe("Scheduled day (YYYY-MM-DD). null = library template."),
  startTime: z
    .string()
    .regex(/^\d{2}:\d{2}$/)
    .nullable()
    .optional()
    .describe("Optional local start time HH:MM, used for the Apple Watch reminder."),
  description: z.string().default(""),
  coachNotes: z.string().default(""),
  status: WorkoutStatus.default("planned"),
  steps: z.array(StepOrRepeat).default([]),
});

export type Duration = z.infer<typeof Duration>;
export type Target = z.infer<typeof Target>;
export type Step = z.infer<typeof Step>;
export type Repeat = z.infer<typeof Repeat>;
export type StepOrRepeat = z.infer<typeof StepOrRepeat>;
export type WorkoutInput = z.infer<typeof WorkoutInput>;
export type SportT = z.infer<typeof Sport>;

export interface Workout extends WorkoutInput {
  id: string;
  plannedDurationSec: number | null;
  plannedDistanceM: number | null;
  plannedTss: number | null;
  completedActivityId: string | null;
  createdAt: string;
  updatedAt: string;
}

export const Lap = z.object({
  startOffsetSec: z.number().nonnegative(),
  durationSec: z.number().nonnegative(),
  distanceM: z.number().nullable().optional(),
  avgHr: z.number().nullable().optional(),
  avgPower: z.number().nullable().optional(),
  avgCadence: z.number().nullable().optional(),
});
export type Lap = z.infer<typeof Lap>;

export const ActivityInput = z.object({
  externalId: z.string().describe("Unique id from the source (e.g. HealthKit UUID)."),
  source: z.string().default("healthkit"),
  sport: Sport,
  title: z.string().default(""),
  startDate: z.string().describe("ISO-8601 timestamp"),
  durationSec: z.number().nonnegative(),
  distanceM: z.number().nonnegative().nullable().optional(),
  elevationGainM: z.number().nullable().optional(),
  avgHr: z.number().nullable().optional(),
  maxHr: z.number().nullable().optional(),
  avgPower: z.number().nullable().optional(),
  normalizedPower: z.number().nullable().optional(),
  avgCadence: z.number().nullable().optional(),
  energyKcal: z.number().nullable().optional(),
  rpe: z.number().min(1).max(10).nullable().optional(),
  notes: z.string().default(""),
  workoutId: z.string().nullable().optional().describe("Planned workout this activity fulfils."),
  laps: z
    .array(Lap)
    .default([])
    .describe("One entry per workout step/segment as recorded by Apple Watch, in order."),
});
export type ActivityInput = z.infer<typeof ActivityInput>;

export interface Activity extends ActivityInput {
  id: string;
  tss: number | null;
  createdAt: string;
}

export const Profile = z.object({
  name: z.string().default("Athlete"),
  ftpWatts: z.number().positive().nullable().default(null),
  thresholdPaceSecPerKm: z.number().positive().nullable().default(null),
  swimCssSecPer100m: z.number().positive().nullable().default(null),
  lthrBpm: z.number().positive().nullable().default(null),
  maxHrBpm: z.number().positive().nullable().default(null),
  restingHrBpm: z.number().positive().nullable().default(null),
  weightKg: z.number().positive().nullable().default(null),
  goals: z.string().default("").describe("Season goals / target events, free text for the coach."),
  weeklyHoursAvailable: z.number().nonnegative().nullable().default(null),
  vo2Max: z.number().positive().nullable().default(null).describe("Apple Watch Cardio Fitness (ml/kg/min)."),
});
export type Profile = z.infer<typeof Profile>;

/** Same fields as Profile but every field optional and without defaults (for partial updates). */
export const ProfilePatch = z.object(
  Object.fromEntries(Object.entries(Profile.shape).map(([k, v]) => [k, v.unwrap().optional()])) as {
    [K in keyof typeof Profile.shape]: z.ZodOptional<ReturnType<(typeof Profile.shape)[K]["unwrap"]>>;
  },
);
