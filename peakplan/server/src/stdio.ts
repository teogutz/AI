// Local MCP over stdio for Claude Desktop / Claude Code when the database is on this machine.
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Store } from "./store.js";
import { buildMcpServer } from "./mcp.js";

const store = new Store(process.env.PEAKPLAN_DB ?? "./data/peakplan.db");
await buildMcpServer(store).connect(new StdioServerTransport());
