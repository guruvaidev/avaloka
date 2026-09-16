const KEYS = [
  "SUPABASE_URL",
  "SUPABASE_INCLUSTER_URL",
  "SUPABASE_ANON_KEY",
  "API_BASE",
  "MCP_API_BASE",
  "MCP_TOOL_BASE",
  "STRIPE_PUBLISHABLE_KEY",
  "PAYPAL_CLIENT_ID",
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
    PAYPAL_CLIENT_ID: read("PAYPAL_CLIENT_ID"),
    MCP_API_BASE: read("MCP_API_BASE"),
    MCP_TOOL_BASE: read("MCP_TOOL_BASE"),
  };
}

export const runtimeEnv = getRuntimeEnv();