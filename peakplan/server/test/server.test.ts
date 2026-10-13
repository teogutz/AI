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
