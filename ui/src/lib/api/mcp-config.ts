/**
 * MCP service base URLs — SEPARATE services from BACKEND_API_BASE.
 *  - MCP_API_BASE : onboarding/registration service (local port 8081)
 *  - MCP_TOOL_BASE: MCP tool server, /call_tool     (local port 8082)
 * Each has its own tunnel; they are NOT the backend tunnel.
 * Change these ONE place when the tunnel URLs rotate.
 */
// export const MCP_API_BASE =
//   (import.meta.env.VITE_MCP_API_BASE as string | undefined)?.replace(/\/$/, "") ||
//   "https://participants-one-tmp-scene.trycloudflare.com";

// export const MCP_TOOL_BASE =
//   (import.meta.env.VITE_MCP_TOOL_BASE as string | undefined)?.replace(/\/$/, "") ||
//   "https://broad-sophisticated-select-turned.trycloudflare.com";

import { runtimeEnv } from "@/lib/runtime-env";

export const MCP_API_BASE =
  runtimeEnv.MCP_API_BASE?.replace(/\/$/, "") ||
  "http://127.0.0.1:8081";

export const MCP_TOOL_BASE =
  runtimeEnv.MCP_TOOL_BASE?.replace(/\/$/, "") ||
  "http://127.0.0.1:8082";
