import express, { NextFunction, Request, Response } from "express";
import { timingSafeEqual } from "node:crypto";
import { readdirSync, readFileSync, statSync, statfsSync } from "node:fs";
import { dirname, join } from "node:path";
import os from "node:os";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { ZodError } from "zod";
import { NotFound, Store } from "./store.js";
import { buildMcpServer } from "./mcp.js";
import { hrZonesBpm } from "./metrics.js";
import { ProfilePatch } from "./schema.js";

const STARTED_AT = new Date().toISOString();
const VERSION = (() => {
  try {
    return JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version as string;
  } catch {
    return "unknown";
  }
})();

export function createApp(store: Store, token: string) {
  const app = express();
  app.use(express.json({ limit: "2mb" }));

  const tokenOk = (given: string | undefined) => {
    if (!given) return false;
    const a = Buffer.from(given);
    const b = Buffer.from(token);
    return a.length === b.length && timingSafeEqual(a, b);
  };
  const auth = (req: Request, res: Response, next: NextFunction) => {
    const bearer = req.headers.authorization?.replace(/^Bearer\s+/i, "");
    const pathToken = typeof req.params.token === "string" ? req.params.token : undefined;
    if (tokenOk(bearer) || tokenOk(pathToken)) return next();
    res.status(401).json({ error: "unauthorized" });
  };

  app.get("/health", (_req, res) => {
    res.json({ ok: true });
  });

  // ---------- MCP (Claude) ----------
  // Claude.ai custom connectors take a plain URL, so the token may also be
  // embedded in the path: https://host/mcp/<token>. Claude Code / Desktop can
  // use https://host/mcp with an "Authorization: Bearer <token>" header.
  const handleMcp = async (req: Request, res: Response) => {
    const server = buildMcpServer(store);
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    res.on("close", () => {
      transport.close();
      server.close();
    });
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  };
  const mcpNotAllowed = (_req: Request, res: Response) => {
    res.status(405).set("Allow", "POST").json({ error: "Method not allowed" });
  };
  app.post("/mcp", auth, handleMcp);
  app.post("/mcp/:token", auth, handleMcp);
  app.get(["/mcp", "/mcp/:token"], mcpNotAllowed);
  app.delete(["/mcp", "/mcp/:token"], mcpNotAllowed);

  // ---------- REST API (iPhone / Mac / Watch apps) ----------
  const api = express.Router();
  api.use(auth);
  const q = (v: unknown) => (typeof v === "string" && v ? v : undefined);

  api.get("/profile", (_req, res) => {
    const p = store.getProfile();
    res.json({ ...p, heartRateZonesBpm: hrZonesBpm(p.lthrBpm) });
  });
  api.put("/profile", (req, res) => {
    res.json(store.updateProfile(ProfilePatch.parse(req.body)));
  });

  api.get("/workouts", (req, res) => {
    res.json(store.listWorkouts({ from: q(req.query.from), to: q(req.query.to), library: req.query.library === "true" }));
  });
  api.get("/workouts/:id", (req, res) => {
    const w = store.getWorkout(req.params.id);
    if (w) res.json(w);
    else res.status(404).json({ error: "not found" });
  });
  api.post("/workouts", (req, res) => {
    // Clients may supply their own UUID so offline-created workouts keep their id.
    const id = typeof req.body?.id === "string" ? req.body.id : undefined;
    const existing = id ? store.getWorkout(id) : null;
    if (id && existing) {
      // Optimistic concurrency: a client editing an old copy must not overwrite a newer change
      // (e.g. one Claude just made). The app sends the updatedAt it loaded as baseUpdatedAt.
      const base = req.body?.baseUpdatedAt;
      if (typeof base === "string" && base !== existing.updatedAt) {
        res.status(409).json({ error: "conflict", message: "Workout changed since it was loaded", current: existing });
        return;
      }
      res.json(store.updateWorkout(id, req.body));
      return;
    }
    res.status(201).json(store.createWorkout(req.body, id));
  });
  api.put("/workouts/:id", (req, res) => {
    res.json(store.updateWorkout(req.params.id, req.body));
  });
  api.delete("/workouts/:id", (req, res) => {
    res.status(store.deleteWorkout(req.params.id) ? 204 : 404).end();
  });

  api.get("/activities", (req, res) => {
    res.json(store.listActivities({ from: q(req.query.from), to: q(req.query.to) }).map((a) => store.enrich(a)));
  });
  api.post("/activities", (req, res) => {
    const items = Array.isArray(req.body) ? req.body : [req.body];
    res.json(items.map((a) => store.upsertActivity(a)));
  });
  api.post("/activities/:id/link", (req, res) => {
    const workoutId = typeof req.body?.workoutId === "string" ? req.body.workoutId : null;
    res.json(store.linkActivity(req.params.id, workoutId, { moveWorkout: req.body?.moveWorkout !== false }));
  });
  api.get("/activities/:id/candidates", (req, res) => {
    res.json(store.workoutCandidates(req.params.id));
  });
  api.get("/workouts/:id/candidates", (req, res) => {
    res.json(store.activityCandidates(req.params.id));
  });
  api.put("/activities/:id", (req, res) => {
    res.json(store.updateActivity(req.params.id, req.body));
  });

  api.get("/review", (req, res) => {
    const to = q(req.query.to) ?? new Date().toISOString().slice(0, 10);
    const from = q(req.query.from) ?? to;
    res.json(store.review(from, to));
  });
  api.get("/workouts/:id/comparison", (req, res) => {
    res.json(store.comparison(req.params.id));
  });

  // Server health for the app's status screen.
  api.get("/status", (_req, res) => {
    const size = (f: string) => {
      try {
        return statSync(f).size;
      } catch {
        return 0;
      }
    };
    let backups: { file: string; sizeBytes: number; modified: string }[] = [];
    if (store.path !== ":memory:") {
      const dir = join(dirname(store.path), "backups");
      try {
        backups = readdirSync(dir)
          .filter((f) => f.endsWith(".db"))
          .map((f) => {
            const st = statSync(join(dir, f));
            return { file: f, sizeBytes: st.size, modified: st.mtime.toISOString() };
          })
          .sort((a, b) => b.modified.localeCompare(a.modified));
      } catch {}
    }
    let disk: { totalBytes: number; freeBytes: number } | null = null;
    try {
      const fs = statfsSync(store.path === ":memory:" ? "." : dirname(store.path));
      disk = { totalBytes: fs.blocks * fs.bsize, freeBytes: fs.bavail * fs.bsize };
    } catch {}
    res.json({
      ok: true,
      version: VERSION,
      node: process.version,
      startedAt: STARTED_AT,
      uptimeSec: Math.round(process.uptime()),
      host: { loadAvg1m: os.loadavg()[0], memTotalBytes: os.totalmem(), memFreeBytes: os.freemem(), disk },
      database: {
        sizeBytes: store.path === ":memory:" ? 0 : size(store.path) + size(store.path + "-wal"),
        ...store.stats(),
      },
      backups: { count: backups.length, latest: backups[0] ?? null },
    });
  });

  api.get("/thresholds", (_req, res) => {
    res.json(store.thresholdCheck());
  });
  api.get("/trends", (req, res) => {
    const to = q(req.query.to) ?? new Date().toISOString().slice(0, 10);
    const from = q(req.query.from) ?? new Date(Date.parse(to) - 84 * 86_400_000).toISOString().slice(0, 10);
    res.json(store.qualityTrends(from, to, q(req.query.sport)));
  });

  api.get("/summary", (req, res) => {
    const to = q(req.query.to) ?? new Date().toISOString().slice(0, 10);
    const from = q(req.query.from) ?? to;
    res.json(store.trainingSummary(from, to));
  });

  app.use("/api", api);

  app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    if (err instanceof ZodError) return void res.status(400).json({ error: "invalid", issues: err.issues });
    if (err instanceof NotFound) return void res.status(404).json({ error: err.message });
    console.error(err);
    res.status(500).json({ error: "internal error" });
  });

  return app;
}
