// @vitest-environment node
//
// A source-level fence for the four things this change removed from the edge
// functions, so none of them can come back unnoticed:
//   - quickchart.io, a third-party chart renderer that received real data
//     values and column labels (no egress inventory entry, no DPA);
//   - the Lovable-hosted watermark asset, fetched on every report email, which
//     no air-gapped or self-hosted install can reach;
//   - any bare fetch() of a customer-supplied URL (the logo SSRF sink).
//
// The plaintext SMTP subject and the send-smtp-email auth-gate ordering are
// owned elsewhere and are deliberately not asserted here.
//
// These are assertions about the committed source, read from disk. They are not
// a substitute for the behavioural tests in ssrf-guard.test.ts; they catch the
// case where a future edit reintroduces the call rather than the case where the
// guard is wrong.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const FUNCTIONS_DIR = fileURLToPath(new URL("../../supabase/functions", import.meta.url));

function allSources(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...allSources(full));
    else if (entry.endsWith(".ts")) out.push(full);
  }
  return out;
}

const sources = allSources(FUNCTIONS_DIR);
const read = (relative: string) => readFileSync(join(FUNCTIONS_DIR, relative), "utf8");

describe("edge function egress surface", () => {
  it("finds the function sources it means to check", () => {
    // Guards against the whole suite passing because the glob found nothing.
    const names = sources.map((s) => s.slice(FUNCTIONS_DIR.length + 1));
    expect(names).toContain("send-report-email/index.ts");
    expect(names).toContain("send-analysis-email/index.ts");
    expect(names).toContain("send-smtp-email/index.ts");
    expect(names).toContain("_shared/ssrf.ts");
    expect(names).toContain("_shared/chart-render.ts");
  });

  it("names no third-party chart renderer in any string literal", () => {
    for (const file of sources) {
      const text = readFileSync(file, "utf8");
      expect(text, file).not.toMatch(/["'`]https:\/\/quickchart\.io/);
      expect(text, file).not.toMatch(/["'`]https?:\/\/[^"'`]*quickchart/);
    }
  });

  it("names no Lovable-hosted asset in any string literal", () => {
    for (const file of sources) {
      const text = readFileSync(file, "utf8");
      expect(text, file).not.toMatch(/["'`]https?:\/\/[^"'`]*lovable\.app/);
    }
  });

  it("makes no fetch() call from either email function", () => {
    // The only outbound HTTP left in the tree is fetchImageSafely, in _shared.
    for (const relative of ["send-report-email/index.ts", "send-analysis-email/index.ts"]) {
      expect(read(relative), relative).not.toMatch(/\bfetch\s*\(/);
    }
  });

  it("fetches the org logo only through the SSRF guard", () => {
    const report = read("send-report-email/index.ts");
    expect(report).toMatch(/import \{ fetchImageSafely \} from "\.\.\/_shared\/ssrf\.ts"/);
    expect(report).toMatch(/await fetchImageSafely\(orgLogoUrl\)/);
    // Bytes may only be embedded on the ok branch.
    expect(report).toMatch(/if \(fetched\.ok\)/);
    expect(report).not.toMatch(/fetchLogoBytes/);
  });

  it("draws charts with the local renderer in both email functions", () => {
    for (const relative of ["send-report-email/index.ts", "send-analysis-email/index.ts"]) {
      const text = read(relative);
      expect(text, relative).toMatch(/from "\.\.\/_shared\/chart-render\.ts"/);
      expect(text, relative).toMatch(/drawChartPlan\(/);
      expect(text, relative).not.toMatch(/renderChartPng/);
    }
  });

});
