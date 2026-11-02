// UI-13 (Master Test Plan): the Supabase URL is a RUNTIME setting of the UI
// image. Changing the deployment's env (window.__ENV__, injected by the root
// route from the server's process.env) points the UI at another project
// without a rebuild. Supersedes tests/contract/test_c4_ui_wiring.py
// ::test_e9_15a/e9_16, which look for files that do not exist on 1.6 and SKIP.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

declare global {
  interface Window {
    __ENV__?: Record<string, string>;
  }
}

beforeEach(() => {
  vi.resetModules();
  delete window.__ENV__;
});
afterEach(() => {
  delete window.__ENV__;
});

describe("UI-13 Supabase URL is read at runtime", () => {
  it("UI-13: getRuntimeEnv reads values set after the module was imported (not inlined at build)", async () => {
    const { getRuntimeEnv } = await import("@/lib/runtime-env");
    expect(getRuntimeEnv().SUPABASE_URL).toBe("");
    window.__ENV__ = { SUPABASE_URL: "https://project-two.example.org" };
    expect(getRuntimeEnv().SUPABASE_URL).toBe("https://project-two.example.org");
  });

  it("UI-13: the Supabase client config uses the deployment's runtime URL and key", async () => {
    window.__ENV__ = { SUPABASE_URL: "https://project-two.example.org", SUPABASE_ANON_KEY: "anon-test" };
    const config = await import("@/integrations/supabase/config");
    expect(config.SUPABASE_URL).toBe("https://project-two.example.org");
    expect(config.SUPABASE_ANON_KEY).toBe("anon-test");
  });

  it("UI-13: with nothing configured the UI falls back to the local self-hosted Supabase, not a hosted project", async () => {
    const config = await import("@/integrations/supabase/config");
    expect(config.SUPABASE_URL).toBe("http://localhost:30091");
    expect(config.SUPABASE_URL).not.toMatch(/supabase\.co/);
  });
});
