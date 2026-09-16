/**
 * Central backend base URL.
 *
 * Change this ONE constant when the Cloudflare tunnel URL rotates.
 * All callers (backendApi, analysis route, proxy fallback) import from here.
 *
 * NOTE: MCP_API_BASE is a *different* service -- do not put it here.
 */
import { runtimeEnv } from "@/lib/runtime-env";

export const BACKEND_API_BASE =
  runtimeEnv.API_BASE?.replace(/\/$/, "") ||
  "http://127.0.0.1:8010";
