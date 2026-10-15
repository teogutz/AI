import { test } from "node:test";
import assert from "node:assert/strict";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { AddressInfo } from "node:net";
import { Store } from "../src/store.js";
import { createApp } from "../src/app.js";
import { plannedMetrics } from "../src/metrics.js";

const TOKEN = "test-token-0123456789abcdef";

const sweetSpot = {
  title: "Sweet Spot 3x12",
  sport: "bike",
  location: "indoor",
  date: "2026-10-14",
  steps: [
    { type: "step", kind: "warmup", duration: { type: "time", seconds: 600 }, target: { type: "power", unit: "percent_ftp", min: 55, max: 65 } },
    {
      type: "repeat",
      count: 3,
      steps: [
        { type: "step", kind: "work", duration: { type: "time", seconds: 720 }, target: { type: "power", unit: "percent_ftp", min: 88, max: 94 } },
        { type: "step", kind: "recovery", duration: { type: "time", seconds: 300 }, target: { type: "power", unit: "percent_ftp", min: 50, max: 60 } },
      ],
    },
    { type: "step", kind: "cooldown", duration: { type: "time", seconds: 600 } },
  ],
};

async function startServer() {
  const store = new Store(":memory:");
  const server = createApp(store, TOKEN).listen(0);
  await new Promise((r) => server.once("listening", r));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  return { store, server, base };
}

test("planned metrics: duration and TSS", () => {
  const m = plannedMetrics(sweetSpot.steps as any, "bike", { ftpWatts: 250 } as any);
  assert.equal(m.plannedDurationSec, 600 + 3 * 1020 + 600);
  // 10' @0.6, 36' @0.91, 15' @0.55, 10' @0.55 -> ~ 70 TSS
  assert.ok(m.plannedTss! > 60 && m.plannedTss! < 75, `tss ${m.plannedTss}`);
});

test("distance steps use pace target", () => {
  const m = plannedMetrics(
    [{ type: "step", kind: "work", duration: { type: "distance", meters: 5000 }, target: { type: "pace", unit: "sec_per_km", min: 290, max: 310 } }],
    "run",
    { thresholdPaceSecPerKm: 280 } as any,
  );
  assert.equal(m.plannedDurationSec, 1500);
  assert.equal(m.plannedDistanceM, 5000);
});

test("REST API requires token and round-trips workouts", async () => {
  const { server, base } = await startServer();
  try {
    assert.equal((await fetch(`${base}/api/workouts`)).status, 401);
    const h = { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" };
    const created = await (await fetch(`${base}/api/workouts`, { method: "POST", headers: h, body: JSON.stringify(sweetSpot) })).json();
    assert.ok(created.id);
    assert.equal(created.plannedDurationSec, 4260);

    const bad = await fetch(`${base}/api/workouts`, { method: "POST", headers: h, body: JSON.stringify({ title: "x" }) });
    assert.equal(bad.status, 400);

    const moved = await (
      await fetch(`${base}/api/workouts/${created.id}`, { method: "PUT", headers: h, body: JSON.stringify({ date: "2026-10-15" }) })
    ).json();
    assert.equal(moved.date, "2026-10-15");
    assert.equal(moved.steps.length, 3);

    const list = await (await fetch(`${base}/api/workouts?from=2026-10-15&to=2026-10-15`, { headers: h })).json();
    assert.equal(list.length, 1);

    // Importing a completed ride on that day links it to the plan.
    const [act] = await (
      await fetch(`${base}/api/activities`, {
        method: "POST",
        headers: h,
        body: JSON.stringify([{ externalId: "hk-1", sport: "bike", startDate: "2026-10-15T07:00:00-06:00", durationSec: 4200, avgPower: 200, avgHr: 145 }]),
      })
    ).json();
    assert.equal(act.workoutId, created.id);
    const done = await (await fetch(`${base}/api/workouts/${created.id}`, { headers: h })).json();
    assert.equal(done.status, "completed");
  } finally {
    server.close();
  }
});

test("profile update keeps unspecified fields", async () => {
  const store = new Store(":memory:");
  store.updateProfile({ ftpWatts: 250, lthrBpm: 165 });
  store.updateProfile({ ftpWatts: 260 });
  assert.equal(store.getProfile().lthrBpm, 165);
});

test("MCP: Claude can read and write workouts", async () => {
  const { server, base } = await startServer();
  const client = new Client({ name: "test", version: "1" });
  try {
    await client.connect(new StreamableHTTPClientTransport(new URL(`${base}/mcp/${TOKEN}`)));
    const tools = (await client.listTools()).tools.map((t) => t.name);
    for (const t of ["create_workout", "list_workouts", "list_activities", "get_training_summary", "update_athlete_profile"])
      assert.ok(tools.includes(t), t);

    await client.callTool({ name: "update_athlete_profile", arguments: { ftpWatts: 250, lthrBpm: 168 } });
    await client.callTool({ name: "update_athlete_profile", arguments: { goals: "Gran Fondo in March" } });
    const profile = JSON.parse(((await client.callTool({ name: "get_athlete_profile", arguments: {} })).content as any)[0].text);
    assert.equal(profile.ftpWatts, 250);
    assert.equal(profile.goals, "Gran Fondo in March");
    assert.deepEqual(profile.heartRateZonesBpm.z2, [143, 150]);

    const res = await client.callTool({ name: "create_workout", arguments: sweetSpot });
    const w = JSON.parse((res.content as any)[0].text);
    assert.match(w.structure, /3x \[ work 12 min @ 88-94% FTP/);

    const upd = await client.callTool({ name: "update_workout", arguments: { id: w.id, status: "skipped" } });
    assert.equal(JSON.parse((upd.content as any)[0].text).status, "skipped");

    const bad = await client.callTool({ name: "create_workout", arguments: { title: "x", sport: "bike" } });
    assert.equal(bad.isError, true);
  } finally {
    await client.close();
    server.close();
  }
});

test("MCP rejects wrong token", async () => {
  const { server, base } = await startServer();
  try {
    const r = await fetch(`${base}/mcp/wrong`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
    assert.equal(r.status, 401);
  } finally {
    server.close();
  }
});

test("fitness chart (CTL/ATL)", () => {
  const store = new Store(":memory:");
  store.updateProfile({ lthrBpm: 160 });
  for (let i = 1; i <= 28; i++) {
    const d = `2026-09-${String(i).padStart(2, "0")}`;
    store.upsertActivity({ externalId: d, sport: "run", startDate: `${d}T07:00:00-06:00`, durationSec: 3600, avgHr: 144 });
  }
  const { ctl, atl } = store.performanceChart("2026-09-28");
  assert.ok(atl > ctl && ctl > 0, `ctl ${ctl} atl ${atl}`);
});

test("clients can clear date (move to library) with explicit null", async () => {
  const store = new Store(":memory:");
  const w = store.createWorkout({ ...sweetSpot, startTime: "06:30" });
  const moved = store.updateWorkout(w.id, { date: null, startTime: null });
  assert.equal(moved.date, null);
  assert.equal(moved.startTime, null);
  assert.equal(store.listWorkouts({ library: true }).length, 1);
});

test("review & interval comparison", () => {
  const store = new Store(":memory:");
  store.updateProfile({ ftpWatts: 250, lthrBpm: 165 });
  const w = store.createWorkout(sweetSpot);
  const missed = store.createWorkout({ ...sweetSpot, title: "Missed", date: "2026-10-01" });
  const laps = [
    { startOffsetSec: 0, durationSec: 600, avgPower: 150 },
    ...[0, 1, 2].flatMap((i) => [
      { startOffsetSec: 0, durationSec: 720, avgPower: i === 2 ? 205 : 228 }, // last rep faded
      { startOffsetSec: 0, durationSec: 300, avgPower: 140 },
    ]),
    { startOffsetSec: 0, durationSec: 600 },
  ];
  store.upsertActivity({ externalId: "x", sport: "bike", startDate: "2026-10-14T06:30:00-06:00", durationSec: 4260, avgPower: 190, laps });

  const c = store.comparison(w.id);
  assert.equal(c.color, "green");
  assert.equal(c.intervalsMatched, true);
  assert.equal(c.steps[1].result, "in_range"); // 228 W vs 220-235
  assert.equal(c.steps[5].result, "too_easy"); // 205 W
  assert.equal(c.steps[5].deviation, "too_low");
  assert.equal(c.execution.verdict, "under_executed");
  assert.equal(c.steps[1].target?.low, 220);

  const review = store.review("2026-09-28", "2026-10-20");
  assert.equal(review.find((r) => r.workout?.id === missed.id)?.state, "missed");
  assert.equal(review.find((r) => r.workout?.id === w.id)?.state, "completed");
});

test("pairing: plan added after import, cross-day manual pairing, unpairing", () => {
  const store = new Store(":memory:");
  // Activity first, plan created later on the same day -> auto-paired.
  const run = store.upsertActivity({ externalId: "r1", sport: "run", startDate: "2026-10-12T14:54:00-06:00", durationSec: 2980 });
  assert.equal(run.workoutId, null);
  const easy = store.createWorkout({ title: "Easy Run", sport: "run", date: "2026-10-12", steps: [] });
  assert.equal(easy.completedActivityId, run.id);

  // Plan for Tuesday, done on Wednesday -> not auto-paired, manual pairing moves the plan.
  const tue = store.createWorkout({ title: "Tempo", sport: "run", date: "2026-10-13", steps: [] });
  const wed = store.upsertActivity({ externalId: "r2", sport: "run", startDate: "2026-10-14T07:00:00-06:00", durationSec: 3000 });
  assert.equal(wed.workoutId, null);
  const cands = store.workoutCandidates(wed.id);
  assert.equal(cands[0].workout.id, tue.id);
  assert.equal(cands[0].dayOffset, -1);
  const { workout } = store.linkActivity(wed.id, tue.id);
  assert.equal(workout?.date, "2026-10-14");
  assert.equal(workout?.status, "completed");

  // Re-import keeps the manual pairing and feedback.
  store.updateActivity(wed.id, { rpe: 7, notes: "legs heavy" });
  const again = store.upsertActivity({ externalId: "r2", sport: "run", startDate: "2026-10-14T07:00:00-06:00", durationSec: 3000 });
  assert.equal(again.workoutId, tue.id);
  assert.equal(again.rpe, 7);
  assert.equal(again.notes, "legs heavy");

  // Unpair -> plan back to planned, and auto-matching won't re-pair it.
  store.linkActivity(wed.id, null);
  assert.equal(store.getWorkout(tue.id)?.status, "planned");
  store.backfillMatches();
  assert.equal(store.getActivity(wed.id)?.workoutId, null);

  // Pairing a plan that already has an activity releases the old one.
  store.linkActivity(wed.id, easy.id, { moveWorkout: false });
  assert.equal(store.getActivity(run.id)?.workoutId, null);
  assert.equal(store.getWorkout(easy.id)?.completedActivityId, wed.id);
});

test("status endpoint", async () => {
  const { server, base } = await startServer();
  try {
    const s = await (await fetch(`${base}/api/status`, { headers: { authorization: `Bearer ${TOKEN}` } })).json();
    assert.equal(s.ok, true);
    assert.equal(typeof s.uptimeSec, "number");
    assert.equal(s.database.workouts, 0);
  } finally {
    server.close();
  }
});

test("intervals align with an extra open segment and reject km splits", () => {
  const store = new Store(":memory:");
  store.updateProfile({ thresholdPaceSecPerKm: 290, lthrBpm: 165 });
  const q = store.createWorkout({
    title: "Quality Run 3x8",
    sport: "run",
    date: "2026-10-15",
    steps: [
      { type: "step", kind: "warmup", duration: { type: "time", seconds: 720 } },
      {
        type: "repeat",
        count: 3,
        steps: [
          { type: "step", kind: "work", duration: { type: "time", seconds: 480 }, target: { type: "pace", unit: "sec_per_km", min: 295, max: 310 } },
          { type: "step", kind: "recovery", duration: { type: "time", seconds: 180 } },
        ],
      },
      { type: "step", kind: "cooldown", duration: { type: "time", seconds: 480 } },
    ],
  });
  // As recorded by the Watch (Apple Fitness "Intervals"), plus 3:18 of extra running.
  const lap = (sec: number, m: number, hr: number) => ({ startOffsetSec: 0, durationSec: sec, distanceM: m, avgHr: hr });
  const laps = [lap(720, 1830, 141), lap(480, 1590, 160), lap(180, 464, 152), lap(480, 1530, 158), lap(180, 426, 155),
    lap(480, 1550, 165), lap(180, 415, 162), lap(480, 1270, 157), lap(198, 447, 159)];
  store.upsertActivity({ externalId: "q", sport: "run", startDate: "2026-10-15T06:00:00-06:00", durationSec: 3345, distanceM: 9550, laps });
  const c = store.comparison(q.id);
  assert.equal(c.intervalsMatched, true);
  assert.equal(c.extraLaps.length, 1);
  assert.equal(c.steps[1].result, "in_range"); // 5:01/km vs 4:55-5:10
  assert.equal(c.steps[3].result, "too_easy"); // 5:13/km is slower than 5:10
  assert.equal(c.steps[3].deviation, "too_slow");
  assert.equal(c.repConsistency?.spread, 11.8); // 5:02 vs 5:14 /km

  // Ten 1 km auto-splits must not be mistaken for the steps.
  const splits = Array.from({ length: 10 }, () => lap(360, 1000, 155));
  store.upsertActivity({ externalId: "q", sport: "run", startDate: "2026-10-15T06:00:00-06:00", durationSec: 3345, distanceM: 9550, laps: splits });
  assert.equal(store.comparison(q.id).intervalsMatched, false);
});

test("pace faster than target is too_fast / too_hard; over-execution and rep spread", () => {
  const store = new Store(":memory:");
  const w = store.createWorkout({
    title: "3x8 @ T",
    sport: "run",
    date: "2026-10-15",
    steps: [1, 2, 3].map(() => ({
      type: "step",
      kind: "work",
      duration: { type: "time", seconds: 480 },
      target: { type: "pace", unit: "sec_per_km", min: 320, max: 330 },
    })),
  });
  const lap = (paceSec: number) => ({ startOffsetSec: 0, durationSec: 480, distanceM: (480 / paceSec) * 1000 });
  store.upsertActivity({ externalId: "t", sport: "run", startDate: "2026-10-15T06:00:00-06:00", durationSec: 1440, laps: [lap(301), lap(305), lap(313)] });
  const c = store.comparison(w.id);
  assert.equal(c.steps[0].result, "too_hard");
  assert.equal(c.steps[0].deviation, "too_fast");
  assert.ok(c.steps[0].deviationPct! < 0);
  assert.equal(c.execution.verdict, "over_executed");
  assert.equal(c.execution.tooHard, 3);
  assert.equal(c.repConsistency?.spread, 12);
  assert.equal(c.repConsistency?.best, 301);
  assert.equal(c.repConsistency?.fade, 12);
  assert.equal(store.qualityTrends("2026-10-01", "2026-10-31").repConsistency.length, 1);
});

test("threshold check flags a stale profile; super shoes are discounted", () => {
  const store = new Store(":memory:");
  store.updateProfile({ thresholdPaceSecPerKm: 348, shoes: [{ id: "vf", name: "Vaporfly", superShoe: true, retired: false }] });
  const day = new Date().toISOString().slice(0, 10);
  const lap = (paceSec: number) => ({ startOffsetSec: 0, durationSec: 480, distanceM: (480 / paceSec) * 1000 });
  store.upsertActivity({ externalId: "x", sport: "run", startDate: `${day}T06:00:00-06:00`, durationSec: 1500, laps: [lap(310), lap(312)] });
  const [t] = store.thresholdCheck();
  assert.equal(t.field, "thresholdPaceSecPerKm");
  assert.equal(t.stale, true);
  assert.equal(t.suggested, Math.round(310 * 1.03));
  // Same effort in super shoes suggests a 3 % slower threshold.
  store.upsertActivity({ externalId: "x", sport: "run", startDate: `${day}T06:00:00-06:00`, durationSec: 1500, shoeId: "vf", laps: [lap(310), lap(312)] });
  assert.equal(store.thresholdCheck()[0].suggested, Math.round(310 * 1.03 * 1.03));
});

test("aerobic decoupling from half splits", () => {
  const store = new Store(":memory:");
  const a = store.upsertActivity({
    externalId: "lr", sport: "run", startDate: "2026-10-10T07:00:00-06:00", durationSec: 6000, distanceM: 17000,
    firstHalf: { durationSec: 3000, distanceM: 8600, avgHr: 145 },
    secondHalf: { durationSec: 3000, distanceM: 8400, avgHr: 152 },
  });
  const d = store.enrich(a).decoupling!;
  assert.ok(d.decouplingPct > 6 && d.decouplingPct < 8, String(d.decouplingPct));
  assert.equal(d.rating, "moderate");
});
