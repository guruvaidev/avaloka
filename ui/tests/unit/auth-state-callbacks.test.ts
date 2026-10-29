// @vitest-environment node
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

// vitest runs from ui/, so src/ is relative to the working directory.
const SRC = resolve(process.cwd(), "src");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.(ts|tsx)$/.test(name) ? [full] : [];
  });
}

describe("supabase.auth.onAuthStateChange callbacks", () => {
  it("finds the source files", () => {
    expect(sourceFiles(SRC).length).toBeGreaterThan(0);
  });

  it("are never async (async callbacks deadlock during a token refresh)", () => {
    const offenders = sourceFiles(SRC).flatMap((file) =>
      readFileSync(file, "utf8")
        .split("\n")
        .map((line, i) => ({ line, i }))
        .filter(({ line }) => /onAuthStateChange\(\s*async\b/.test(line))
        .map(({ i }) => `${file.slice(SRC.length + 1)}:${i + 1}`),
    );
    expect(offenders).toEqual([]);
  });
});