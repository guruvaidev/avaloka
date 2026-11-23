// The page shell asks no third party for anything, and publishes no in-cluster
// address.
//
// It used to: every page view loaded a stylesheet (and then fonts) from
// api.fontshare.com, which hands the viewer's IP to a third party and leaves an
// offline install with fallback type; og:image/twitter:image pointed at a
// public r2.dev bucket; and window.__ENV__ carried API_BASE ("http://api:9000"),
// the in-cluster address the UI server proxies to.
//
// Allowed third-party hosts: none. (There is no basemap in this UI; if one is
// added, its configured tile host is the only thing that belongs here.)
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const THIRD_PARTY_ALLOWLIST: readonly string[] = [];

// vitest runs from ui/ (vitest.config.ts); jsdom gives import.meta.url a non-file scheme.
const UI_ROOT = process.cwd();

// Any absolute or protocol-relative URL naming a host.
const ABSOLUTE_URL = /(?:https?:)?\/\/([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?::\d+)?/gi;

function thirdPartyHosts(values: string[]): string[] {
  const hosts: string[] = [];
  for (const v of values) {
    for (const m of v.matchAll(ABSOLUTE_URL)) {
      const host = m[1].toLowerCase();
      if (!THIRD_PARTY_ALLOWLIST.includes(host)) hosts.push(`${host} (in ${JSON.stringify(v)})`);
    }
  }
  return hosts;
}

vi.mock("@tanstack/react-start", () => {
  // The root route's loader is a server function, and (since the team-invite
  // claim) its imports build middleware-backed server functions; the head
  // needs none of them, so every builder step returns a builder and handler
  // returns the function as-is.
  const builder = (): Record<string, unknown> => ({
    handler: (fn: unknown) => fn,
    middleware: () => builder(),
    inputValidator: () => builder(),
    validator: () => builder(),
    server: () => builder(),
    client: () => builder(),
  });
  return { createServerFn: () => builder(), createMiddleware: () => builder() };
});

afterEach(() => {
  delete window.__ENV__;
  vi.resetModules();
});

describe("root route head", () => {
  it("references no third-party host in any meta or link", async () => {
    const { Route } = await import("@/routes/__root");
    const head = (Route.options.head as () => {
      meta: Record<string, unknown>[];
      links: Record<string, unknown>[];
    })();
    const values = [...head.meta, ...head.links].flatMap((tag) =>
      Object.values(tag).filter((v): v is string => typeof v === "string"),
    );
    expect(values.length).toBeGreaterThan(5);
    expect(thirdPartyHosts(values)).toEqual([]);
    // Positive checks: the replacements are local.
    expect(head.meta).toContainEqual({ property: "og:image", content: "/og-image.png" });
    expect(values.some((v) => v.includes("fontshare"))).toBe(false);
  });
});

describe("the env published into the page", () => {
  it("never carries the in-cluster API or Supabase address", async () => {
    // What the UI server's process.env holds in the quickstart / chart.
    window.__ENV__ = {
      API_BASE: "http://api:9000",
      SUPABASE_INCLUSTER_URL: "http://avaloka-supabase-kong:8000",
      AUTH_MODE: "local",
    };
    const env = await import("@/lib/runtime-env");
    // The server itself still sees them (it proxies to them) ...
    expect(env.getRuntimeEnv().API_BASE).toBe("http://api:9000");
    // ... the page does not, whichever way the shell is fed.
    const fromLoader = env.getPublicRuntimeEnv();
    expect(fromLoader.API_BASE).toBe("");
    expect(fromLoader.SUPABASE_INCLUSTER_URL).toBe("");
    expect(fromLoader.AUTH_MODE).toBe("local");
    for (const script of [
      env.publicEnvScript(fromLoader),
      env.publicEnvScript(env.getRuntimeEnv()),
    ]) {
      expect(script).not.toContain("api:9000");
      expect(script).not.toContain("supabase-kong");
      expect(script).toContain('"AUTH_MODE":"local"');
    }
  });

  it("escapes markup in values", async () => {
    const env = await import("@/lib/runtime-env");
    expect(env.publicEnvScript({ AUTH_MODE: "</script><b>" })).not.toContain("</script>");
  });
});

describe("static assets", () => {
  function walk(dir: string): string[] {
    return readdirSync(dir).flatMap((name) => {
      const p = join(dir, name);
      return statSync(p).isDirectory() ? walk(p) : [p];
    });
  }

  it("CSS loads fonts and images from this origin only", () => {
    const css = [join(UI_ROOT, "src/styles.css"), ...walk(join(UI_ROOT, "src/styles"))].filter((p) =>
      p.endsWith(".css"),
    );
    const loads: string[] = [];
    for (const file of css) {
      // Comments are not requests.
      const text = readFileSync(file, "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
      for (const m of text.matchAll(/url\(\s*(['"]?)([^'")]+)\1\s*\)|@import\s+(['"])([^'"]+)\3/g)) {
        loads.push(`${relative(UI_ROOT, file)}: ${m[2] ?? m[4]}`);
      }
    }
    expect(loads.some((l) => l.includes("/fonts/figtree/"))).toBe(true);
    expect(thirdPartyHosts(loads)).toEqual([]);
  });

  it("public HTML/JS/CSS reference no third-party host", () => {
    const files = walk(join(UI_ROOT, "public")).filter((p) => /\.(html?|m?js|css|webmanifest|json)$/i.test(p));
    const values = files.map((f) => `${relative(UI_ROOT, f)}: ${readFileSync(f, "utf8")}`);
    expect(thirdPartyHosts(values)).toEqual([]);
  });

  it("bundles the font files and their licence", () => {
    const dir = join(UI_ROOT, "public/fonts/figtree");
    const names = readdirSync(dir);
    expect(names).toEqual(
      expect.arrayContaining([
        "figtree-latin-wght-normal.woff2",
        "figtree-latin-ext-wght-normal.woff2",
        "OFL.txt",
      ]),
    );
    expect(readFileSync(join(dir, "OFL.txt"), "utf8")).toMatch(/SIL Open Font License/);
    expect(statSync(join(UI_ROOT, "public/og-image.png")).size).toBeGreaterThan(1000);
  });
});
