// AUTH_MODE=local (the quickstart): the Supabase client must talk to the API's
// local-auth routes through this origin, never to the kind NodePort fallback
// (localhost:30091) that made every _authenticated page bounce to "/".
// runtimeEnv is read once at import, so each case sets window.__ENV__ and
// imports the modules fresh.
import { afterEach, describe, expect, it, vi } from "vitest";

async function load(env: Record<string, string>) {
  window.__ENV__ = env;
  vi.resetModules();
  return {
    config: await import("@/integrations/supabase/config"),
    backend: await import("@/lib/api/backend-config"),
  };
}

afterEach(() => {
  delete window.__ENV__;
  vi.resetModules();
});

describe("Supabase config in local auth mode", () => {
  it("points the browser client at the same-origin local-auth proxy", async () => {
    const { config } = await load({ AUTH_MODE: "local" });
    expect(config.isLocalAuth()).toBe(true);
    expect(config.SUPABASE_URL).toBe(`${window.location.origin}/api/public/local-auth`);
    expect(config.SUPABASE_URL).not.toContain("30091");
    expect(config.SUPABASE_ANON_KEY).toBe("avaloka-local-auth");
  });

  it("is case-insensitive about the mode", async () => {
    const { config } = await load({ AUTH_MODE: " Local " });
    expect(config.isLocalAuth()).toBe(true);
  });

  it("refuses local mode beside a real Supabase project", async () => {
    await expect(
      load({ AUTH_MODE: "local", SUPABASE_URL: "https://x.supabase.co", SUPABASE_ANON_KEY: "k" }),
    ).rejects.toThrow(/mutually exclusive/);
  });

  it("leaves Supabase mode exactly as it was", async () => {
    const { config } = await load({ SUPABASE_URL: "https://x.supabase.co", SUPABASE_ANON_KEY: "k" });
    expect(config.isLocalAuth()).toBe(false);
    expect(config.SUPABASE_URL).toBe("https://x.supabase.co");
    expect(config.SUPABASE_ANON_KEY).toBe("k");
  });
});

describe("API base default", () => {
  it("is the port the API serves on (9000), not 8010", async () => {
    const { backend } = await load({});
    expect(backend.BACKEND_API_BASE).toBe("http://127.0.0.1:9000");
  });

  it("still honours API_BASE", async () => {
    const { backend } = await load({ API_BASE: "http://api:9000/" });
    expect(backend.BACKEND_API_BASE).toBe("http://api:9000");
  });
});
