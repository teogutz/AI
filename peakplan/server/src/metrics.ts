import type { ActivityInput, Profile, SportT, Step, StepOrRepeat, Target } from "./schema.js";

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
