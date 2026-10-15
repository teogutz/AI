import { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import {
  Activity,
  ActivityInput,
  Profile,
  Workout,
  WorkoutInput,
} from "./schema.js";
import { activityTss, compareWorkout, complianceColor, decoupling, normalizeSteps, plannedMetrics } from "./metrics.js";

export class Store {
  private db: DatabaseSync;

  constructor(readonly path: string) {
    if (path !== ":memory:") mkdirSync(dirname(path), { recursive: true });
    this.db = new DatabaseSync(path);
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS workouts (
        id TEXT PRIMARY KEY,
        date TEXT,
        json TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS workouts_date ON workouts(date);
      CREATE TABLE IF NOT EXISTS activities (
        id TEXT PRIMARY KEY,
        external_id TEXT UNIQUE NOT NULL,
        start_date TEXT NOT NULL,
        json TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS activities_start ON activities(start_date);
      CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY CHECK (id = 1), json TEXT NOT NULL);
    `);
  }

  /** Counts and freshness for the status screen. */
  stats() {
    const one = (sql: string) => (this.db.prepare(sql).get() as any) ?? {};
    const today = new Date().toISOString().slice(0, 10);
    return {
      workouts: one("SELECT COUNT(*) AS n FROM workouts").n as number,
      upcomingWorkouts: (this.db.prepare("SELECT COUNT(*) AS n FROM workouts WHERE date >= ?").get(today) as any).n as number,
      activities: one("SELECT COUNT(*) AS n FROM activities").n as number,
      lastWorkoutChange: (one("SELECT MAX(updated_at) AS t FROM workouts").t as string) ?? null,
      lastActivity: (one("SELECT MAX(start_date) AS t FROM activities").t as string) ?? null,
      lastImport:
        (one("SELECT MAX(json_extract(json, '$.createdAt')) AS t FROM activities").t as string) ?? null,
    };
  }

  // ---------- profile ----------

  getProfile(): Profile {
    const row = this.db.prepare("SELECT json FROM profile WHERE id = 1").get() as { json: string } | undefined;
    return Profile.parse(row ? JSON.parse(row.json) : {});
  }

  updateProfile(patch: Partial<Profile>): Profile {
    const next = Profile.parse({ ...this.getProfile(), ...patch });
    this.db
      .prepare("INSERT INTO profile (id, json) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET json = excluded.json")
      .run(JSON.stringify(next));
    // Thresholds changed, so planned TSS/duration of future workouts changes too.
    const today = new Date().toISOString().slice(0, 10);
    for (const w of this.listWorkouts({ from: today })) this.saveWorkout(w, next);
    for (const w of this.listWorkouts({ library: true })) this.saveWorkout(w, next);
    return next;
  }

  // ---------- workouts ----------

  private saveWorkout(w: Workout, profile = this.getProfile()): Workout {
    const steps = normalizeSteps(w.steps);
    const full: Workout = {
      ...w,
      steps,
      ...plannedMetrics(steps, w.sport, profile),
      updatedAt: new Date().toISOString(),
    };
    this.db
      .prepare(
        `INSERT INTO workouts (id, date, json, updated_at) VALUES (?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET date = excluded.date, json = excluded.json, updated_at = excluded.updated_at`,
      )
      .run(full.id, full.date, JSON.stringify(full), full.updatedAt);
    return full;
  }

  createWorkout(input: unknown, id?: string): Workout {
    const parsed = WorkoutInput.parse(input);
    const now = new Date().toISOString();
    const created = this.saveWorkout({
      ...parsed,
      id: id ?? randomUUID(),
      plannedDurationSec: null,
      plannedDistanceM: null,
      plannedTss: null,
      completedActivityId: null,
      createdAt: now,
      updatedAt: now,
    });
    // A plan added after the activity was already imported (e.g. planned retroactively).
    if (created.date) this.autoMatch(created.date);
    return this.getWorkout(created.id)!;
  }

  getWorkout(id: string): Workout | null {
    const row = this.db.prepare("SELECT json FROM workouts WHERE id = ?").get(id) as { json: string } | undefined;
    return row ? JSON.parse(row.json) : null;
  }

  /** Partial update: any WorkoutInput field may be supplied. */
  updateWorkout(id: string, patch: Record<string, unknown>): Workout {
    const existing = this.getWorkout(id);
    if (!existing) throw new NotFound(`Workout ${id} not found`);
    const merged = WorkoutInput.parse({ ...existing, ...patch });
    const saved = this.saveWorkout({ ...existing, ...merged });
    if (saved.date && saved.date !== existing.date) this.autoMatch(saved.date);
    return this.getWorkout(id)!;
  }

  deleteWorkout(id: string): boolean {
    const w = this.getWorkout(id);
    if (w?.completedActivityId) {
      const a = this.getActivity(w.completedActivityId);
      if (a && a.workoutId === id) this.writeActivity({ ...a, workoutId: null });
    }
    return Number(this.db.prepare("DELETE FROM workouts WHERE id = ?").run(id).changes) > 0;
  }

  listWorkouts(opts: { from?: string; to?: string; library?: boolean } = {}): Workout[] {
    let rows: { json: string }[];
    if (opts.library) {
      rows = this.db.prepare("SELECT json FROM workouts WHERE date IS NULL ORDER BY updated_at DESC").all() as any;
    } else {
      rows = this.db
        .prepare("SELECT json FROM workouts WHERE date IS NOT NULL AND date >= ? AND date <= ? ORDER BY date")
        .all(opts.from ?? "0000-00-00", opts.to ?? "9999-99-99") as any;
    }
    return rows.map((r) => JSON.parse(r.json));
  }

  // ---------- activities ----------

  private writeActivity(a: Activity) {
    this.db
      .prepare(
        `INSERT INTO activities (id, external_id, start_date, json) VALUES (?, ?, ?, ?)
         ON CONFLICT(external_id) DO UPDATE SET start_date = excluded.start_date, json = excluded.json`,
      )
      .run(a.id, a.externalId, a.startDate, JSON.stringify(a));
  }

  /** Insert or update by externalId, then try to pair it with a planned workout. */
  upsertActivity(input: unknown): Activity {
    const parsed = ActivityInput.parse(input);
    const profile = this.getProfile();
    const existing = this.db
      .prepare("SELECT json FROM activities WHERE external_id = ?")
      .get(parsed.externalId) as { json: string } | undefined;
    const prev: Activity | null = existing ? JSON.parse(existing.json) : null;

    // Impossible segment speeds (GPS dropouts, overlapping split queries) are dropped at the door.
    const maxSpeed = parsed.sport === "bike" ? 25 : parsed.sport === "swim" ? 2.5 : 7; // m/s
    const laps = parsed.laps.map((l) =>
      l.distanceM && l.durationSec > 0 && l.distanceM / l.durationSec > maxSpeed ? { ...l, distanceM: null } : l,
    );
    const activity: Activity = {
      ...parsed,
      laps,
      // Re-imports must not wipe the athlete's feedback or a manual pairing.
      rpe: parsed.rpe ?? prev?.rpe ?? null,
      notes: parsed.notes || prev?.notes || "",
      workoutId: prev?.manualLink ? (prev.workoutId ?? null) : (parsed.workoutId ?? prev?.workoutId ?? null),
      shoeId:
        parsed.shoeId ??
        prev?.shoeId ??
        (["run", "walk", "hike"].includes(parsed.sport) ? profile.defaultShoeId : null) ??
        null,
      manualLink: prev?.manualLink ?? false,
      id: prev?.id ?? randomUUID(),
      tss: activityTss(parsed, profile),
      createdAt: prev?.createdAt ?? new Date().toISOString(),
    };
    this.writeActivity(activity);

    if (activity.workoutId) {
      const w = this.getWorkout(activity.workoutId);
      if (w) this.saveWorkout({ ...w, status: "completed", completedActivityId: activity.id });
    } else {
      this.autoMatch(localDay(activity.startDate));
    }
    return this.getActivity(activity.id)!;
  }

  getActivity(id: string): Activity | null {
    const row = this.db.prepare("SELECT json FROM activities WHERE id = ?").get(id) as { json: string } | undefined;
    return row ? JSON.parse(row.json) : null;
  }

  updateActivity(
    id: string,
    patch: { rpe?: number | null; notes?: string; workoutId?: string | null; shoeId?: string | null },
  ): Activity {
    const a = this.getActivity(id);
    if (!a) throw new NotFound(`Activity ${id} not found`);
    const { workoutId, ...feedback } = patch;
    this.writeActivity({ ...a, ...feedback });
    if (workoutId !== undefined && workoutId !== a.workoutId) return this.linkActivity(id, workoutId).activity;
    return this.getActivity(id)!;
  }

  // ---------- pairing completed activities with planned workouts ----------

  /**
   * Automatic pairing for one day: each unpaired activity goes to an unpaired planned workout of
   * the same sport that day, preferring the closest planned duration. Activities the athlete
   * paired or unpaired by hand are left alone.
   */
  autoMatch(day: string) {
    const free = this.listActivities({ from: day, to: day }).filter((a) => !a.workoutId && !a.manualLink);
    for (const a of free) {
      const candidates = this.listWorkouts({ from: day, to: day }).filter(
        (w) => w.sport === a.sport && !w.completedActivityId && w.status !== "skipped",
      );
      if (candidates.length === 0) continue;
      candidates.sort(
        (x, y) => Math.abs((x.plannedDurationSec ?? 0) - a.durationSec) - Math.abs((y.plannedDurationSec ?? 0) - a.durationSec),
      );
      this.pair(a, candidates[0], false);
    }
  }

  /** Re-run automatic pairing over all history (cheap; used at startup). */
  backfillMatches() {
    const days = new Set(this.listActivities().filter((a) => !a.workoutId && !a.manualLink).map((a) => localDay(a.startDate)));
    for (const d of days) this.autoMatch(d);
  }

  private pair(a: Activity, w: Workout, manual: boolean) {
    this.writeActivity({ ...a, workoutId: w.id, manualLink: manual || a.manualLink });
    this.saveWorkout({ ...w, status: "completed", completedActivityId: a.id });
  }

  /**
   * Manually pair an activity with a planned workout (or unpair with workoutId = null).
   * Undoes any previous pairing on either side. With moveWorkout, a plan done on another day is
   * moved to the day it was actually done, as TrainingPeaks does.
   */
  linkActivity(activityId: string, workoutId: string | null, opts: { moveWorkout?: boolean } = {}) {
    const a = this.getActivity(activityId);
    if (!a) throw new NotFound(`Activity ${activityId} not found`);

    // Release the activity's current plan.
    if (a.workoutId) {
      const old = this.getWorkout(a.workoutId);
      if (old && old.completedActivityId === a.id) this.saveWorkout({ ...old, status: "planned", completedActivityId: null });
    }

    if (workoutId === null) {
      this.writeActivity({ ...a, workoutId: null, manualLink: true });
      return { activity: this.getActivity(activityId)!, workout: null };
    }

    let w = this.getWorkout(workoutId);
    if (!w) throw new NotFound(`Workout ${workoutId} not found`);
    // Release the plan's current activity.
    if (w.completedActivityId && w.completedActivityId !== a.id) {
      const other = this.getActivity(w.completedActivityId);
      if (other) this.writeActivity({ ...other, workoutId: null, manualLink: true });
    }
    const day = localDay(a.startDate);
    if ((opts.moveWorkout ?? true) && w.date !== day) w = { ...w, date: day };
    this.pair({ ...a, workoutId: null }, w, true);
    return { activity: this.getActivity(activityId)!, workout: this.getWorkout(workoutId)! };
  }

  /** Planned workouts within ±3 days that this activity could be paired with, best first. */
  workoutCandidates(activityId: string) {
    const a = this.getActivity(activityId);
    if (!a) throw new NotFound(`Activity ${activityId} not found`);
    const day = localDay(a.startDate);
    return this.listWorkouts({ from: addDays(day, -3), to: addDays(day, 3) })
      .filter((w) => w.id !== a.workoutId)
      .map((w) => ({
        workout: w,
        dayOffset: dayDiff(day, w.date!),
        sameSport: w.sport === a.sport,
        pairedActivityId: w.completedActivityId,
      }))
      .sort(
        (x, y) =>
          Number(y.sameSport) - Number(x.sameSport) ||
          Number(!!x.pairedActivityId) - Number(!!y.pairedActivityId) ||
          Math.abs(x.dayOffset) - Math.abs(y.dayOffset) ||
          Math.abs((x.workout.plannedDurationSec ?? 0) - a.durationSec) -
            Math.abs((y.workout.plannedDurationSec ?? 0) - a.durationSec),
      );
  }

  /** Completed activities within ±3 days that could fulfil this planned workout, best first. */
  activityCandidates(workoutId: string) {
    const w = this.getWorkout(workoutId);
    if (!w || !w.date) throw new NotFound(`Workout ${workoutId} not found or not scheduled`);
    return this.listActivities({ from: addDays(w.date, -3), to: addDays(w.date, 3) })
      .filter((a) => a.id !== w.completedActivityId)
      .map((a) => ({
        activity: a,
        dayOffset: dayDiff(w.date!, localDay(a.startDate)),
        sameSport: a.sport === w.sport,
        pairedWorkoutId: a.workoutId ?? null,
      }))
      .sort(
        (x, y) =>
          Number(y.sameSport) - Number(x.sameSport) ||
          Number(!!x.pairedWorkoutId) - Number(!!y.pairedWorkoutId) ||
          Math.abs(x.dayOffset) - Math.abs(y.dayOffset) ||
          Math.abs(x.activity.durationSec - (w.plannedDurationSec ?? 0)) -
            Math.abs(y.activity.durationSec - (w.plannedDurationSec ?? 0)),
      );
  }

  listActivities(opts: { from?: string; to?: string } = {}): Activity[] {
    const rows = this.db
      .prepare("SELECT json FROM activities WHERE start_date >= ? AND start_date < ? ORDER BY start_date")
      .all(opts.from ?? "0000", nextDay(opts.to) ?? "9999") as { json: string }[];
    return rows.map((r) => JSON.parse(r.json));
  }

  // ---------- coaching insights ----------

  /** Activity plus derived insights (cardiac drift, shoe) for API/MCP output. */
  enrich(a: Activity) {
    const shoe = this.getProfile().shoes.find((s) => s.id === a.shoeId) ?? null;
    return { ...a, shoe, decoupling: decoupling(a) };
  }

  /**
   * Checks the profile's thresholds against recent hard efforts (last 6 weeks). Stale thresholds
   * silently distort %-based targets and TSS, so this flags when recent work is clearly better.
   */
  thresholdCheck(asOf = new Date().toISOString().slice(0, 10)) {
    const profile = this.getProfile();
    const from = addDays(asOf, -42);
    const acts = this.listActivities({ from, to: asOf });
    const shoes = new Map(profile.shoes.map((s) => [s.id, s]));
    const out: any[] = [];

    // Run threshold pace: fastest *plausible* sustained segment (laps 6-60 min, whole runs 20-70 min).
    // GPS/segment glitches produce impossible paces, so every candidate goes through guards:
    //   1. absolute bound: no faster than ~15 % off world-record pace for that duration
    //   2. internal consistency: a segment can't be >40 % faster than its own run unless HR backs it up
    //   3. heart rate: a threshold-level effort has HR near threshold; fast pace at easy HR is fake
    //   4. relative to the athlete: >10 % faster than the current threshold is rejected outright,
    //      >5 % is suggested but flagged as a large change that needs confirming
    type Ev = {
      est: number; date: string; title: string; durationMin: number; paceSecPerKm: number;
      superShoe: boolean; avgHr: number | null;
    };
    const runEv: Ev[] = [];
    const rejected: { date: string; title: string; durationMin: number; paceSecPerKm: number; reason: string }[] = [];
    const lthr = profile.lthrBpm ?? (profile.maxHrBpm ? profile.maxHrBpm * 0.89 : null);
    const current = profile.thresholdPaceSecPerKm;
    for (const a of acts.filter((x) => x.sport === "run")) {
      const superShoe = !!(a.shoeId && shoes.get(a.shoeId)?.superShoe);
      const runPace = a.distanceM ? a.durationSec / (a.distanceM / 1000) : null;
      const segs = [
        ...(a.laps ?? []).map((l) => ({ sec: l.durationSec, m: l.distanceM ?? 0, hr: l.avgHr ?? null })),
        { sec: a.durationSec, m: a.distanceM ?? 0, hr: a.avgHr ?? null },
      ];
      for (const seg of segs) {
        if (seg.sec < 6 * 60 || seg.sec > 70 * 60 || seg.m < 1000) continue;
        const pace = seg.sec / (seg.m / 1000);
        const info = { date: localDay(a.startDate), title: a.title, durationMin: Math.round(seg.sec / 60), paceSecPerKm: Math.round(pace) };
        const reject = (reason: string) => rejected.push({ ...info, reason });
        if (pace > 600) continue; // walking pace: not a threshold effort, not an error
        // 1. ~World-record pace: 2:30/km up to 15 min, 2:35 to 30 min, 2:45 to 70 min.
        const wr = seg.sec <= 15 * 60 ? 150 : seg.sec <= 30 * 60 ? 155 : 165;
        if (pace < wr * 1.15) {
          reject("faster than world-class pace for this duration: GPS/segment error");
          continue;
        }
        // 3. Heart rate must look like a hard effort.
        const hardHr = seg.hr != null && lthr != null && seg.hr >= lthr * 0.88;
        if (seg.hr != null && lthr != null && !hardHr) {
          // Easy-HR segments can't set a threshold; only reject loudly if the pace is suspiciously fast.
          if (current && pace < current) reject(`pace not matched by heart rate (${Math.round(seg.hr)} bpm)`);
          continue;
        }
        // 2. A segment much faster than its own run, without HR evidence, is almost certainly a glitch.
        if (runPace && pace < runPace * 0.6 && !hardHr) {
          reject(`${Math.round((1 - pace / runPace) * 100)} % faster than the rest of this run`);
          continue;
        }
        const factor = seg.sec < 15 * 60 ? 1.03 : seg.sec < 30 * 60 ? 1.02 : seg.sec < 45 * 60 ? 1.01 : 1.0;
        const est = pace * factor * (superShoe ? 1.03 : 1);
        // 4. Relative to the athlete.
        if (current && est < current * 0.9) {
          reject(`implies a threshold ${Math.round((1 - est / current) * 100)} % faster than the current one`);
          continue;
        }
        runEv.push({ ...info, est, superShoe, avgHr: seg.hr });
      }
    }
    if (runEv.length || rejected.length) {
      const best = runEv.length ? runEv.reduce((x, y) => (y.est < x.est ? y : x)) : null;
      const suggested = best ? Math.round(best.est) : null;
      const diffPct = current && suggested ? Math.round(((current - suggested) / current) * 1000) / 10 : null;
      out.push({
        field: "thresholdPaceSecPerKm",
        current,
        suggested,
        diffPct, // positive = recent work is faster than the profile
        stale: suggested != null && (current == null || (diffPct != null && diffPct > 3)),
        largeChange: diffPct != null && diffPct > 5, // confirm before applying
        basis: best,
        rejectedEvidence: rejected.slice(0, 10),
        note: "Estimate from the fastest plausible sustained effort; confirm with a 30-min test if unsure.",
      });
    }

    // Bike FTP: best 20-60 min power.
    const bikeEv: { est: number; date: string; title: string; durationMin: number; watts: number }[] = [];
    for (const a of acts.filter((x) => x.sport === "bike")) {
      const segs = [...(a.laps ?? []).map((l) => ({ sec: l.durationSec, w: l.avgPower })), { sec: a.durationSec, w: a.normalizedPower ?? a.avgPower }];
      for (const seg of segs) {
        if (!seg.w || seg.sec < 20 * 60 || seg.sec > 70 * 60) continue;
        const est = seg.w * (seg.sec < 30 * 60 ? 0.95 : seg.sec < 45 * 60 ? 0.97 : 1.0);
        bikeEv.push({ est, date: localDay(a.startDate), title: a.title, durationMin: Math.round(seg.sec / 60), watts: Math.round(seg.w) });
      }
    }
    if (bikeEv.length) {
      const best = bikeEv.reduce((x, y) => (y.est > x.est ? y : x));
      const suggested = Math.round(best.est);
      const current = profile.ftpWatts;
      const diffPct = current ? Math.round(((suggested - current) / current) * 1000) / 10 : null;
      out.push({ field: "ftpWatts", current, suggested, diffPct, stale: current == null || (diffPct != null && diffPct > 3), basis: best });
    }
    return out;
  }

  /** Rep consistency and cardiac drift over time: the trends worth coaching from. */
  qualityTrends(from: string, to: string, sport?: string) {
    const profile = this.getProfile();
    const shoes = new Map(profile.shoes.map((s) => [s.id, s]));
    const reps: any[] = [];
    const drift: any[] = [];
    for (const a of this.listActivities({ from, to })) {
      if (sport && a.sport !== sport) continue;
      const shoe = a.shoeId ? (shoes.get(a.shoeId) ?? null) : null;
      if (a.workoutId) {
        const w = this.getWorkout(a.workoutId);
        if (w) {
          const c = compareWorkout(w, a, profile);
          if (c.repConsistency)
            reps.push({ date: localDay(a.startDate), title: w.title, ...c.repConsistency, execution: c.execution, shoe: shoe?.name ?? null, superShoe: !!shoe?.superShoe });
        }
      }
      const d = decoupling(a);
      if (d) drift.push({ date: localDay(a.startDate), title: a.title, durationMin: Math.round(a.durationSec / 60), ...d, shoe: shoe?.name ?? null });
    }
    return { repConsistency: reps, aerobicDecoupling: drift };
  }

  // ---------- review (planned vs. completed) ----------

  /** Detailed planned-vs-actual comparison for one planned workout. */
  comparison(workoutId: string) {
    const w = this.getWorkout(workoutId);
    if (!w) throw new NotFound(`Workout ${workoutId} not found`);
    const a = w.completedActivityId ? this.getActivity(w.completedActivityId) : null;
    const profile = this.getProfile();
    return {
      ...compareWorkout(w, a, profile),
      decoupling: a ? decoupling(a) : null,
      shoe: a?.shoeId ? (profile.shoes.find((s) => s.id === a.shoeId) ?? null) : null,
    };
  }

  /**
   * Everything that happened (or should have) between two days: completed activities with their
   * plan and compliance colour, missed workouts (past, still "planned") and skipped ones.
   */
  review(from: string, to: string) {
    const today = new Date().toISOString().slice(0, 10);
    const workouts = this.listWorkouts({ from, to });
    const activities = this.listActivities({ from, to });
    const byId = new Map(workouts.map((w) => [w.id, w]));
    const items: any[] = [];

    for (const a of activities) {
      const w = a.workoutId ? (byId.get(a.workoutId) ?? this.getWorkout(a.workoutId)) : null;
      const durationPct = w?.plannedDurationSec ? Math.round((a.durationSec / w.plannedDurationSec) * 100) : null;
      const tssPct = w?.plannedTss && a.tss != null ? Math.round((a.tss / w.plannedTss) * 100) : null;
      items.push({
        date: localDay(a.startDate),
        state: w ? "completed" : "unplanned",
        color: w ? complianceColor(durationPct ?? tssPct) : null,
        durationPct,
        tssPct,
        workout: w ?? null,
        activity: a,
      });
    }
    for (const w of workouts) {
      if (w.completedActivityId) continue;
      if (w.status === "skipped") items.push({ date: w.date, state: "skipped", color: "red", workout: w, activity: null });
      else if (w.status === "completed") items.push({ date: w.date, state: "manual", color: null, workout: w, activity: null });
      else if (w.date! < today) items.push({ date: w.date, state: "missed", color: "red", workout: w, activity: null });
    }
    return items.sort((x, y) => y.date.localeCompare(x.date));
  }

  // ---------- analysis ----------

  /**
   * Weekly planned vs. completed volume plus a performance-management chart
   * (CTL = 42-day fitness, ATL = 7-day fatigue, TSB = form), as in TrainingPeaks.
   */
  trainingSummary(from: string, to: string) {
    const workouts = this.listWorkouts({ from, to });
    const activities = this.listActivities({ from, to });
    const weeks = new Map<string, any>();
    const week = (d: string) => weekStart(d);
    const bucket = (k: string) => {
      if (!weeks.has(k))
        weeks.set(k, {
          weekOf: k,
          plannedTss: 0,
          plannedHours: 0,
          completedTss: 0,
          completedHours: 0,
          plannedWorkouts: 0,
          completedWorkouts: 0,
          skippedWorkouts: 0,
          bySport: {} as Record<string, { hours: number; km: number; tss: number }>,
        });
      return weeks.get(k);
    };
    for (const w of workouts) {
      const b = bucket(week(w.date!));
      b.plannedWorkouts++;
      b.plannedTss += w.plannedTss ?? 0;
      b.plannedHours += (w.plannedDurationSec ?? 0) / 3600;
      if (w.status === "skipped") b.skippedWorkouts++;
    }
    for (const a of activities) {
      const b = bucket(week(localDay(a.startDate)));
      b.completedWorkouts++;
      b.completedTss += a.tss ?? 0;
      b.completedHours += a.durationSec / 3600;
      const s = (b.bySport[a.sport] ??= { hours: 0, km: 0, tss: 0 });
      s.hours += a.durationSec / 3600;
      s.km += (a.distanceM ?? 0) / 1000;
      s.tss += a.tss ?? 0;
    }
    const round = (o: any): any =>
      typeof o === "number"
        ? Math.round(o * 10) / 10
        : o && typeof o === "object"
          ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, round(v)]))
          : o;

    return {
      weeks: [...weeks.values()].sort((a, b) => a.weekOf.localeCompare(b.weekOf)).map(round),
      fitness: this.performanceChart(to),
    };
  }

  /** CTL/ATL/TSB as of `asOf`, computed from all activity history. */
  performanceChart(asOf: string) {
    const all = this.listActivities({ to: asOf });
    if (all.length === 0) return { ctl: 0, atl: 0, tsb: 0 };
    const daily = new Map<string, number>();
    for (const a of all) daily.set(localDay(a.startDate), (daily.get(localDay(a.startDate)) ?? 0) + (a.tss ?? 0));
    let ctl = 0;
    let atl = 0;
    let tsb = 0;
    for (let d = localDay(all[0].startDate); d <= asOf; d = nextDay(d)!) {
      tsb = ctl - atl; // form is yesterday's fitness minus fatigue
      const tss = daily.get(d) ?? 0;
      ctl += (tss - ctl) / 42;
      atl += (tss - atl) / 7;
    }
    const r = (n: number) => Math.round(n * 10) / 10;
    return { ctl: r(ctl), atl: r(atl), tsb: r(tsb) };
  }
}

export class NotFound extends Error {}

function localDay(iso: string): string {
  // Timestamps from the phone carry the local offset (e.g. 2026-10-13T06:30:00-06:00),
  // so the first 10 characters are the athlete's local calendar day.
  return iso.slice(0, 10);
}

function nextDay(d?: string): string | undefined {
  if (!d) return undefined;
  const t = new Date(d + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + 1);
  return t.toISOString().slice(0, 10);
}

function addDays(d: string, n: number): string {
  const t = new Date(d + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}

/** Whole days from `from` to `to` (positive = later). */
function dayDiff(from: string, to: string): number {
  return Math.round((Date.parse(to + "T00:00:00Z") - Date.parse(from + "T00:00:00Z")) / 86_400_000);
}

function weekStart(d: string): string {
  const t = new Date(d + "T00:00:00Z");
  const dow = (t.getUTCDay() + 6) % 7; // Monday = 0
  t.setUTCDate(t.getUTCDate() - dow);
  return t.toISOString().slice(0, 10);
}
