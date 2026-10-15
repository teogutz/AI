import type { Activity, ActivityInput, Lap, Profile, SportT, Step, StepOrRepeat, Target, Workout } from "./schema.js";

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

/**
 * Moves an alertMetric given inside `target` (where API clients naturally put it) onto the step,
 * which is where the app and the Watch read it. Step-level wins if both are set.
 */
export function normalizeSteps(steps: StepOrRepeat[]): StepOrRepeat[] {
  const fix = (s: Step): Step => {
    const t = s.target as (Target & { alertMetric?: "average" | "current" }) | undefined;
    if (!t || t.alertMetric === undefined) return s;
    const { alertMetric, ...target } = t;
    return { ...s, alertMetric: s.alertMetric ?? alertMetric, target: target as Target };
  };
  return steps.map((s) => (s.type === "repeat" ? { ...s, steps: s.steps.map(fix) } : fix(s)));
}

/** The alert metric the Watch will actually use for a step (mirrors WorkoutKitBridge.swift). */
export function effectiveAlertMetric(s: Step, profile?: Pick<Profile, "workPaceAlertMetric">) {
  if (!s.target) return null;
  if (s.alertMetric) return s.alertMetric;
  if (s.target.type === "pace" && s.kind === "work") return profile?.workPaceAlertMetric ?? "average";
  return "current";
}

export function describeSteps(steps: StepOrRepeat[], profile?: Pick<Profile, "workPaceAlertMetric">): string {
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
  // Pace/power alerts can use the step average or the current value: show which.
  const alertTag = (s: Step) =>
    s.target && (s.target.type === "pace" || s.target.type === "power")
      ? ` (${effectiveAlertMetric(s, profile) === "average" ? "avg" : "current"} alert${s.alertMetric ? "" : ", default"})`
      : "";
  const line = (s: Step) => `${s.name ?? s.kind} ${fmtDur(s)}${fmtTarget(s.target)}${alertTag(s)}`;
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
 * Recorded segments line up with planned steps when, in order, each segment's length is
 * plausible for its step. Extra segments at the end (kept going after the plan) and a
 * missing tail (stopped early) are allowed; auto km/mile splits are rejected because their
 * lengths don't follow the plan.
 */
export function lapsAlignWithSteps(planned: Step[], laps: Lap[], sport: SportT, profile: Profile): boolean {
  if (laps.length === 0 || planned.length === 0) return false;
  const n = Math.min(planned.length, laps.length);
  if (n < Math.min(planned.length, 2)) return false;
  for (let i = 0; i < n; i++) {
    const step = planned[i];
    const lap = laps[i];
    if (step.duration.type === "open") continue;
    if (step.duration.type === "distance" && lap.distanceM) {
      const r = lap.distanceM / step.duration.meters;
      if (r < 0.6 || r > 1.5) return false;
      continue;
    }
    const sec = stepSeconds(step, sport, profile);
    if (!sec) continue;
    const r = lap.durationSec / sec;
    // The last matched step may have been cut short.
    if (r > 1.5 || (r < 0.6 && !(i === n - 1 && laps.length <= planned.length))) return false;
  }
  return true;
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
  const lapsMatchSteps = lapsAlignWithSteps(planned, laps, workout.sport, profile);
  // Extra recorded segments after the plan (e.g. kept running after the cool-down).
  const extraLaps = lapsMatchSteps ? laps.slice(planned.length) : [];

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
    // result is about effort: too_hard / too_easy. deviation names the direction in the
    // metric's own terms, so pace reads naturally (fewer s/km = too_fast = too_hard).
    let result: "in_range" | "too_hard" | "too_easy" | null = null;
    let deviation: "too_fast" | "too_slow" | "too_high" | "too_low" | null = null;
    let deviationPct: number | null = null;
    if (actual != null && target) {
      const tol = (target.high - target.low) * 0.1 + (target.metric === "heart_rate" ? 2 : 0);
      const isPace = target.metric === "pace";
      if (actual < target.low - tol) {
        result = isPace ? "too_hard" : "too_easy";
        deviation = isPace ? "too_fast" : "too_low";
        deviationPct = Math.round(((actual - target.low) / target.low) * 1000) / 10;
      } else if (actual > target.high + tol) {
        result = isPace ? "too_easy" : "too_hard";
        deviation = isPace ? "too_slow" : "too_high";
        deviationPct = Math.round(((actual - target.high) / target.high) * 1000) / 10;
      } else result = "in_range";
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
      result,
      deviation,
      deviationPct,
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
    recordedSegments: laps.length,
    stepsNotDone: lapsMatchSteps ? Math.max(0, planned.length - laps.length) : 0,
    extraLaps,
    intervalsInRangePct: judged.length
      ? Math.round((judged.filter((s) => s.result === "in_range").length / judged.length) * 100)
      : null,
    execution: executionSummary(judged.map((s) => s.result!)),
    repConsistency: repConsistency(steps),
    steps,
  };
}

/**
 * Over- vs under-execution call for opposite coaching, so they're counted separately.
 * verdict: on_target | over_executed (all misses too hard) | under_executed | mixed.
 */
export function executionSummary(results: ("in_range" | "too_hard" | "too_easy")[]) {
  const inRange = results.filter((r) => r === "in_range").length;
  const tooHard = results.filter((r) => r === "too_hard").length;
  const tooEasy = results.filter((r) => r === "too_easy").length;
  const verdict =
    results.length === 0
      ? null
      : tooHard === 0 && tooEasy === 0
        ? "on_target"
        : tooEasy === 0
          ? "over_executed"
          : tooHard === 0
            ? "under_executed"
            : "mixed";
  return { judged: results.length, inRange, tooHard, tooEasy, verdict };
}

/**
 * Spread between the fastest and slowest work rep (same metric as the targets), the most
 * coachable number for repeated intervals. For pace, "best" is the fastest (fewest s/km).
 */
export function repConsistency(
  steps: { kind: string; target: { metric: string; unit: string } | null; actualValue: number | null }[],
) {
  const work = steps.filter((s) => s.kind === "work" && s.target && s.actualValue != null);
  if (work.length < 2) return null;
  const metric = work[0].target!.metric;
  const values = work.filter((s) => s.target!.metric === metric).map((s) => s.actualValue!);
  if (values.length < 2) return null;
  const lowerIsBetter = metric === "pace";
  const best = lowerIsBetter ? Math.min(...values) : Math.max(...values);
  const worst = lowerIsBetter ? Math.max(...values) : Math.min(...values);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length);
  const r1 = (n: number) => Math.round(n * 10) / 10;
  return {
    metric,
    unit: work[0].target!.unit,
    reps: values.length,
    values: values.map(r1),
    best: r1(best),
    worst: r1(worst),
    spread: r1(Math.abs(worst - best)),
    cvPct: r1((sd / mean) * 100),
    // Positive = later reps slower/weaker than the first ones (fade); negative = negative split.
    fade: r1(lowerIsBetter ? values[values.length - 1] - values[0] : values[0] - values[values.length - 1]),
  };
}

/**
 * Aerobic decoupling (Pa:HR): efficiency (speed or power per heartbeat) in the second half vs the
 * first. Under ~5 % on a steady long run means good aerobic durability. Needs half splits.
 */
export function decoupling(a: Activity) {
  const h1 = a.firstHalf;
  const h2 = a.secondHalf;
  if (!h1 || !h2 || !h1.avgHr || !h2.avgHr || a.durationSec < 40 * 60) return null;
  const out = (h: NonNullable<Activity["firstHalf"]>) =>
    h.avgPower && (a.sport === "bike" || a.sport === "row") ? h.avgPower : h.distanceM ? h.distanceM / h.durationSec : null;
  const o1 = out(h1);
  const o2 = out(h2);
  if (!o1 || !o2) return null;
  const ef1 = o1 / h1.avgHr;
  const ef2 = o2 / h2.avgHr;
  const pct = Math.round(((ef1 - ef2) / ef1) * 1000) / 10;
  return {
    decouplingPct: pct,
    rating: pct < 5 ? "good" : pct < 8 ? "moderate" : "high",
    firstHalf: { avgHr: Math.round(h1.avgHr), paceSecPerKm: h1.distanceM ? Math.round(h1.durationSec / (h1.distanceM / 1000)) : null },
    secondHalf: { avgHr: Math.round(h2.avgHr), paceSecPerKm: h2.distanceM ? Math.round(h2.durationSec / (h2.distanceM / 1000)) : null },
    note: "Meaningful for steady efforts; interval sessions and hilly routes inflate it.",
  };
}
