// @vitest-environment node
// Release-suite wiring guard: a test folder missing from ui/vitest.config.ts
// "include" never runs and never fails. Every *.test.ts(x) under ui/tests/
// (except e2e/, which Playwright runs) must match one of the include globs.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const uiRoot = join(__dirname, "..", "..");

function listTests(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return name === "node_modules" ? [] : listTests(path);
    return /\.test\.tsx?$/.test(name) ? [relative(uiRoot, path)] : [];
  });
}

/** "tests/unit/**\/*.test.{ts,tsx}" -> RegExp. Supports **, *, and {a,b}. */
function globToRegExp(glob: string): RegExp {
  let re = "";
  for (let i = 0; i < glob.length; i++) {
    const ch = glob[i];
    if (ch === "*" && glob[i + 1] === "*") {
      re += "(?:.*/)?";
      i += glob[i + 2] === "/" ? 2 : 1;
    } else if (ch === "*") re += "[^/]*";
    else if (ch === "{") {
      const end = glob.indexOf("}", i);
      re += `(?:${glob.slice(i + 1, end).split(",").join("|")})`;
      i = end;
    } else re += ch.replace(/[.+?^$()|[\]\\]/g, "\\$&");
  }
  return new RegExp(`^${re}$`);
}

describe("vitest include covers every unit/component/release test", () => {
  it("1.6 suite wiring: every *.test.ts(x) under ui/tests (outside e2e/) is matched by ui/vitest.config.ts include", () => {
    const config = readFileSync(join(uiRoot, "vitest.config.ts"), "utf8");
    const block = config.match(/include:\s*\[([\s\S]*?)\]/);
    expect(block, "include list not found in vitest.config.ts").not.toBeNull();
    const globs = Array.from(block![1].matchAll(/"([^"]+)"/g), (m) => m[1]);
    expect(globs.length).toBeGreaterThan(0);
    const patterns = globs.map(globToRegExp);
    const tests = listTests(join(uiRoot, "tests")).filter((p) => !p.startsWith("tests/e2e/"));
    expect(tests).toContain("tests/release_1_6/vitest-include.test.ts");
    const uncovered = tests.filter((p) => !patterns.some((re) => re.test(p)));
    expect(uncovered).toEqual([]);
  });
});
