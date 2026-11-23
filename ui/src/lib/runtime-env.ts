const KEYS = [
  "SUPABASE_URL",
  "SUPABASE_INCLUSTER_URL",
  "SUPABASE_ANON_KEY",
  "API_BASE",
  "MCP_API_BASE",
  "MCP_TOOL_BASE",
  "STRIPE_PUBLISHABLE_KEY",
  // "local" for a self-hosted laptop install with no Supabase (the API's
  // AVALOKA_AUTH_MODE=local); empty for Supabase sign-in.
  "AUTH_MODE",
] as const;

type Key = (typeof KEYS)[number];
export type RuntimeEnv = Record<Key, string>;

declare global {
  interface Window {
    __ENV__?: Partial<RuntimeEnv>;
  }
}

function read(key: Key): string {
  if (typeof window !== "undefined") return window.__ENV__?.[key] ?? "";
  if (typeof process !== "undefined" && process.env) return process.env[key] ?? "";
  return "";
}

export function getRuntimeEnv(): RuntimeEnv {
  return {
    SUPABASE_URL: read("SUPABASE_URL"),
    SUPABASE_INCLUSTER_URL: read("SUPABASE_INCLUSTER_URL"),
    SUPABASE_ANON_KEY: read("SUPABASE_ANON_KEY"),
    API_BASE: read("API_BASE"),
    STRIPE_PUBLISHABLE_KEY: read("STRIPE_PUBLISHABLE_KEY"),
    MCP_API_BASE: read("MCP_API_BASE"),
    MCP_TOOL_BASE: read("MCP_TOOL_BASE"),
    AUTH_MODE: read("AUTH_MODE"),
  };
}

/**
 * Keys that name in-cluster addresses. The UI server reads them from its own
 * process.env and proxies to them (routes/api/public/backend-proxy.ts,
 * local-auth.$.ts); the browser never calls them, so they are never published
 * into the page. Rendering them leaked e.g. API_BASE "http://api:9000".
 */
export const SERVER_ONLY_KEYS = ["API_BASE", "SUPABASE_INCLUSTER_URL"] as const satisfies readonly Key[];

/** The runtime env with every server-only key blanked: what the page may carry. */
export function getPublicRuntimeEnv(): RuntimeEnv {
  const env = getRuntimeEnv();
  for (const key of SERVER_ONLY_KEYS) env[key] = "";
  return env;
}

/** The inline script that hands the public env to the browser. */
export function publicEnvScript(env: Partial<RuntimeEnv>): string {
  const safe: Partial<RuntimeEnv> = { ...env };
  for (const key of SERVER_ONLY_KEYS) delete safe[key];
  return `window.__ENV__=${JSON.stringify(safe).replace(/</g, "\\u003c")}`;
}

export const runtimeEnv = getRuntimeEnv();