import express, { NextFunction, Request, Response } from "express";
import { timingSafeEqual } from "node:crypto";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { ZodError } from "zod";
import { NotFound, Store } from "./store.js";
import { buildMcpServer } from "./mcp.js";
import { hrZonesBpm } from "./metrics.js";
import { ProfilePatch } from "./schema.js";

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
    if (id && store.getWorkout(id)) {
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
    res.json(store.listActivities({ from: q(req.query.from), to: q(req.query.to) }));
  });
  api.post("/activities", (req, res) => {
    const items = Array.isArray(req.body) ? req.body : [req.body];
    res.json(items.map((a) => store.upsertActivity(a)));
  });
  api.put("/activities/:id", (req, res) => {
    res.json(store.updateActivity(req.params.id, req.body));
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
