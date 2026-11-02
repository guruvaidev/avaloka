// @vitest-environment node
// 1.6 release suite wiring guard, kept OUTSIDE tests/release_1_6 on purpose:
// if the release_1_6 glob were dropped from ui/vitest.config.ts, the guard in
// tests/release_1_6/vitest-include.test.ts would stop running with it. This
// copy lives under tests/unit (an original glob) and checks that the
// release_1_6 glob is present and matches every test file in that folder.
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const uiRoot = join(__dirname, "..", "..");

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

describe("vitest include covers the 1.6 release suite", () => {
  it("1.6 suite wiring: ui/vitest.config.ts includes a glob that matches every ui/tests/release_1_6 test file", () => {
    const config = readFileSync(join(uiRoot, "vitest.config.ts"), "utf8");
    const block = config.match(/include:\s*\[([\s\S]*?)\]/);
    expect(block, "include list not found in vitest.config.ts").not.toBeNull();
    const globs = Array.from(block![1].matchAll(/"([^"]+)"/g), (m) => m[1]);
    const releaseGlobs = globs.filter((g) => g.startsWith("tests/release_1_6/"));
    expect(releaseGlobs, "no tests/release_1_6 glob in vitest include").not.toHaveLength(0);
    const patterns = releaseGlobs.map(globToRegExp);
    const files = readdirSync(join(uiRoot, "tests", "release_1_6"))
      .filter((f) => /\.test\.tsx?$/.test(f))
      .map((f) => `tests/release_1_6/${f}`);
    expect(files.length).toBeGreaterThan(0);
    expect(files.filter((f) => !patterns.some((re) => re.test(f)))).toEqual([]);
  });
});
