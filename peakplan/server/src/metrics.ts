import type { Activity, ActivityInput, Profile, SportT, Step, StepOrRepeat, Target, Workout } from "./schema.js";

// Planned-load estimates, TrainingPeaks style: TSS = hours * IF^2 * 100.
// The Swift app has a mirror of this in PeakPlanKit/WorkoutMetrics.swift.

const HR_ZONE_IF: Record<number, number> = { 1: 0.6, 2: 0.72, 3: 0.83, 4: 0.93, 5: 1.03 };
const KIND_DEFAULT_IF: Record<Step["kind"], number> = {
  warmup: 0.6,
  work: 0.8,
  recovery: 0.55,
  rest: 0.5,
  cooldown: 0.55,
};

// Friel-style heart rate zones as % of lactate threshold HR.
export const HR_ZONES_PCT_LTHR: Record<number, [number, number]> = {
  1: [50, 84],
  2: [85, 89],
  3: [90, 94],
  4: [95, 99],
  5: [100, 106],
};

export function hrZonesBpm(lthr: number | null) {
  if (!lthr) return null;
  return Object.fromEntries(
    Object.entries(HR_ZONES_PCT_LTHR).map(([z, [lo, hi]]) => [
      `z${z}`,
      [Math.round((lo / 100) * lthr), Math.round((hi / 100) * lthr)],
    ]),
  );
}

export function flatten(steps: StepOrRepeat[]): Step[] {
  return steps.flatMap((s) => (s.type === "repeat" ? Array.from({ length: s.count }, () => s.steps).flat() : [s]));
}

const mid = (t: { min: number; max: number }) => (t.min + t.max) / 2;

export function intensityFactor(step: Step, profile: Profile): number {
  const t: Target | undefined = step.target;
  if (!t) return KIND_DEFAULT_IF[step.kind];
  switch (t.type) {
    case "power":
      if (t.unit === "percent_ftp") return mid(t) / 100;
      return profile.ftpWatts ? mid(t) / profile.ftpWatts : KIND_DEFAULT_IF[step.kind];
    case "heart_rate_zone":
      return HR_ZONE_IF[t.zone];
    case "heart_rate":
      if (t.unit === "percent_lthr") return mid(t) / 100;
      return profile.lthrBpm ? mid(t) / profile.lthrBpm : KIND_DEFAULT_IF[step.kind];
    case "pace":
      if (t.unit === "percent_threshold") return mid(t) / 100;
      if (t.unit === "sec_per_km")
        return profile.thresholdPaceSecPerKm ? profile.thresholdPaceSecPerKm / mid(t) : KIND_DEFAULT_IF[step.kind];
      return profile.swimCssSecPer100m ? profile.swimCssSecPer100m / mid(t) : KIND_DEFAULT_IF[step.kind];
    case "cadence":
      return KIND_DEFAULT_IF[step.kind];
  }
}

/** Threshold speed in m/s for the sport, used to turn distance steps into time. */
function thresholdSpeed(sport: SportT, profile: Profile): number {
  switch (sport) {
    case "run":
      return 1000 / (profile.thresholdPaceSecPerKm ?? 300);
    case "swim":
      return 100 / (profile.swimCssSecPer100m ?? 120);
    case "bike":
      return 36 / 3.6;
    case "walk":
    case "hike":
      return 5.5 / 3.6;
    case "row":
      return 500 / 120;
    default:
      return 1000 / 360;
  }
}

export function stepSeconds(step: Step, sport: SportT, profile: Profile): number | null {
  const d = step.duration;
  if (d.type === "time") return d.seconds;
  if (d.type === "open") return null;
  const t = step.target;
  if (t?.type === "pace" && t.unit === "sec_per_km") return (d.meters / 1000) * mid(t);
  if (t?.type === "pace" && t.unit === "sec_per_100m") return (d.meters / 100) * mid(t);
  // Speed scales roughly with intensity for distance-based steps.
  const speed = thresholdSpeed(sport, profile) * Math.min(1.1, Math.max(0.6, intensityFactor(step, profile)));
  return d.meters / speed;
}

export function plannedMetrics(steps: StepOrRepeat[], sport: SportT, profile: Profile) {
  const flat = flatten(steps);
  if (flat.length === 0) return { plannedDurationSec: null, plannedDistanceM: null, plannedTss: null };
  let seconds = 0;
  let load = 0;
  let meters = 0;
  let hasDistance = false;
  for (const step of flat) {
    const sec = stepSeconds(step, sport, profile) ?? 0;
    const ifac = intensityFactor(step, profile);
    seconds += sec;
    load += sec * ifac * ifac;
    if (step.duration.type === "distance") {
      meters += step.duration.meters;
      hasDistance = true;
    }
  }
  return {
    plannedDurationSec: Math.round(seconds),
    plannedDistanceM: hasDistance ? Math.round(meters) : null,
    plannedTss: Math.round((load / 3600) * 100),
  };
}

/** Actual TSS for a completed activity: power-based if possible, else heart-rate based. */
export function activityTss(a: ActivityInput, profile: Profile): number | null {
  const hours = a.durationSec / 3600;
  const power = a.normalizedPower ?? a.avgPower;
  if (power && profile.ftpWatts && (a.sport === "bike" || a.sport === "row")) {
    const ifac = power / profile.ftpWatts;
    return Math.round(hours * ifac * ifac * 100);
  }
  if (a.avgHr && profile.lthrBpm) {
    const ifac = a.avgHr / profile.lthrBpm;
    return Math.round(hours * ifac * ifac * 100);
  }
  return null;
}

export function describeSteps(steps: StepOrRepeat[]): string {
  const fmtDur = (s: Step) => {
    const d = s.duration;
    if (d.type === "open") return "open";
    if (d.type === "distance") return d.meters >= 1000 ? `${+(d.meters / 1000).toFixed(2)} km` : `${d.meters} m`;
    const m = Math.floor(d.seconds / 60);
    const sec = Math.round(d.seconds % 60);
    return sec ? `${m}:${String(sec).padStart(2, "0")}` : `${m} min`;
  };
  const fmtTarget = (t?: Target) => {
    if (!t) return "";
    switch (t.type) {
      case "power":
        return ` @ ${t.min}-${t.max}${t.unit === "watts" ? " W" : "% FTP"}`;
      case "heart_rate_zone":
        return ` @ HR Z${t.zone}`;
      case "heart_rate":
        return ` @ ${t.min}-${t.max}${t.unit === "bpm" ? " bpm" : "% LTHR"}`;
      case "pace": {
        const p = (v: number) => `${Math.floor(v / 60)}:${String(Math.round(v % 60)).padStart(2, "0")}`;
        if (t.unit === "percent_threshold") return ` @ ${t.min}-${t.max}% threshold pace`;
        return ` @ ${p(t.min)}-${p(t.max)} ${t.unit === "sec_per_km" ? "/km" : "/100m"}`;
      }
      case "cadence":
        return ` @ ${t.min}-${t.max} rpm`;
    }
  };
  const line = (s: Step) => `${s.name ?? s.kind} ${fmtDur(s)}${fmtTarget(s.target)}`;
  return steps
    .map((s) => (s.type === "repeat" ? `${s.count}x [ ${s.steps.map(line).join(" / ")} ]` : line(s)))
    .join("\n");
}

// ---------- planned vs. completed ----------

export type ComplianceColor = "green" | "yellow" | "orange" | "red";

/** TrainingPeaks-style colour: green 80–120 %, yellow 50–79 / 121–150 %, orange beyond, red = missed. */
export function complianceColor(pct: number | null): ComplianceColor {
  if (pct === null) return "green";
  if (pct >= 80 && pct <= 120) return "green";
  if (pct >= 50 && pct <= 150) return "yellow";
  return "orange";
}

const pct = (actual: number | null | undefined, planned: number | null | undefined) =>
  actual != null && planned ? Math.round((actual / planned) * 100) : null;

/** Absolute target range for a step, in the units the lap data uses. */
export function absoluteTarget(
  step: Step,
  profile: Profile,
  sport: SportT,
): { metric: "power" | "heart_rate" | "pace" | "cadence"; low: number; high: number; unit: string } | null {
  const t = step.target;
  if (!t) return null;
  const range = (a: number, b: number) => [Math.min(a, b), Math.max(a, b)] as const;
  switch (t.type) {
    case "power": {
      if (t.unit === "watts") return { metric: "power", ...lh(range(t.min, t.max)), unit: "W" };
      if (!profile.ftpWatts) return null;
      return { metric: "power", ...lh(range((t.min * profile.ftpWatts) / 100, (t.max * profile.ftpWatts) / 100)), unit: "W" };
    }
    case "heart_rate_zone": {
      const z = HR_ZONES_PCT_LTHR[t.zone];
      if (!profile.lthrBpm || !z) return null;
      return { metric: "heart_rate", ...lh(range((z[0] * profile.lthrBpm) / 100, (z[1] * profile.lthrBpm) / 100)), unit: "bpm" };
    }
    case "heart_rate": {
      if (t.unit === "bpm") return { metric: "heart_rate", ...lh(range(t.min, t.max)), unit: "bpm" };
      if (!profile.lthrBpm) return null;
      return { metric: "heart_rate", ...lh(range((t.min * profile.lthrBpm) / 100, (t.max * profile.lthrBpm) / 100)), unit: "bpm" };
    }
    case "pace": {
      if (t.unit === "sec_per_km") return { metric: "pace", ...lh(range(t.min, t.max)), unit: "s/km" };
      if (t.unit === "sec_per_100m") return { metric: "pace", ...lh(range(t.min, t.max)), unit: "s/100m" };
      const thr = sport === "swim" ? profile.swimCssSecPer100m : profile.thresholdPaceSecPerKm;
      if (!thr) return null;
      // % of threshold *speed*: 100% = threshold pace, higher % = faster = fewer seconds.
      return {
        metric: "pace",
        ...lh(range((thr * 100) / t.max, (thr * 100) / t.min)),
        unit: sport === "swim" ? "s/100m" : "s/km",
      };
    }
    case "cadence":
      return { metric: "cadence", ...lh(range(t.min, t.max)), unit: "rpm" };
  }
}

function lh([low, high]: readonly [number, number]) {
  return { low: Math.round(low * 10) / 10, high: Math.round(high * 10) / 10 };
}

/**
 * Compares a completed activity with its planned workout: totals and, when the Watch
 * recorded one segment per step, each interval's actual value against its target.
 */
export function compareWorkout(workout: Workout, activity: Activity | null, profile: Profile) {
  const durationPct = pct(activity?.durationSec, workout.plannedDurationSec);
  const distancePct = pct(activity?.distanceM, workout.plannedDistanceM);
  const tssPct = pct(activity?.tss, workout.plannedTss);
  const color: ComplianceColor = activity ? complianceColor(durationPct ?? distancePct ?? tssPct) : "red";

  const planned = flatten(workout.steps);
  const laps = activity?.laps ?? [];
  const lapsMatchSteps = laps.length > 0 && laps.length === planned.length;

  const steps = planned.map((step, i) => {
    const lap = lapsMatchSteps ? laps[i] : undefined;
    const target = absoluteTarget(step, profile, workout.sport);
    let actual: number | null = null;
    if (lap && target) {
      switch (target.metric) {
        case "power":
          actual = lap.avgPower ?? null;
          break;
        case "heart_rate":
          actual = lap.avgHr ?? null;
          break;
        case "cadence":
          actual = lap.avgCadence ?? null;
          break;
        case "pace":
          if (lap.distanceM) actual = lap.durationSec / (lap.distanceM / (target.unit === "s/100m" ? 100 : 1000));
          break;
      }
    }
    let result: "in_range" | "below" | "above" | null = null;
    if (actual != null && target) {
      const tol = (target.high - target.low) * 0.1 + (target.metric === "heart_rate" ? 2 : 0);
      if (actual < target.low - tol) result = target.metric === "pace" ? "above" : "below"; // lower sec/km = faster
      else if (actual > target.high + tol) result = target.metric === "pace" ? "below" : "above";
      else result = "in_range";
    }
    return {
      index: i,
      kind: step.kind,
      name: step.name ?? null,
      plannedDurationSec: stepSeconds(step, workout.sport, profile),
      plannedDistanceM: step.duration.type === "distance" ? step.duration.meters : null,
      target,
      actualDurationSec: lap?.durationSec ?? null,
      actualDistanceM: lap?.distanceM ?? null,
      actualAvgHr: lap?.avgHr ?? null,
      actualAvgPower: lap?.avgPower ?? null,
      actualValue: actual != null ? Math.round(actual * 10) / 10 : null,
      // "below"/"above" mean too easy / too hard (for pace: slower / faster than target).
      result,
    };
  });

  const judged = steps.filter((s) => s.result !== null);
  return {
    workoutId: workout.id,
    activityId: activity?.id ?? null,
    color,
    totals: {
      planned: {
        durationSec: workout.plannedDurationSec,
        distanceM: workout.plannedDistanceM,
        tss: workout.plannedTss,
      },
      actual: activity
        ? {
            durationSec: activity.durationSec,
            distanceM: activity.distanceM ?? null,
            tss: activity.tss,
            avgHr: activity.avgHr ?? null,
            avgPower: activity.avgPower ?? null,
            rpe: activity.rpe ?? null,
          }
        : null,
      durationPct,
      distancePct,
      tssPct,
    },
    intervalsMatched: lapsMatchSteps,
    intervalsInRangePct: judged.length
      ? Math.round((judged.filter((s) => s.result === "in_range").length / judged.length) * 100)
      : null,
    steps,
  };
}
