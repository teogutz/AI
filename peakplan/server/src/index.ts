import { Store } from "./store.js";
import { createApp } from "./app.js";

const token = process.env.PEAKPLAN_TOKEN;
if (!token || token.length < 16) {
  console.error("Set PEAKPLAN_TOKEN to a random secret of at least 16 characters (e.g. `openssl rand -hex 24`).");
  process.exit(1);
}
const port = Number(process.env.PORT ?? 8787);
const store = new Store(process.env.PEAKPLAN_DB ?? "./data/peakplan.db");

createApp(store, token).listen(port, () => {
  console.log(`PeakPlan server on :${port}`);
  console.log(`  REST API:  /api   (Authorization: Bearer <token>)`);
  console.log(`  MCP:       /mcp   (Bearer) or /mcp/<token> for Claude.ai custom connectors`);
});
