// Unit + component test config.
//
// Deliberately NOT derived from vite.config.ts: that file loads the Lovable
// TanStack Start wrapper (router codegen, nitro, SSR entry), none of which a
// unit test needs and all of which slow or break a test run. Vitest still uses
// Vite's own transform, so the two things the source actually relies on -- the
// React JSX transform and the "@/..." tsconfig path alias -- are wired
// here. The alias is spelled out rather than read by vite-tsconfig-paths,
// which only applies to files inside tsconfig "include" (src/), not tests/.
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "jsdom",
    // tests/release_1_6 holds the 1.6 release-readiness suite (see
    // docs/test-reports/1.6-traceability.csv). A folder missing from this list
    // never runs and never fails, so tests/release_1_6/vitest-include.test.ts
    // checks that every *.test.ts(x) under tests/ matches one of these globs.
    include: [
      "tests/unit/**/*.test.{ts,tsx}",
      "tests/component/**/*.test.{ts,tsx}",
      "tests/release_1_6/**/*.test.{ts,tsx}",
    ],
    setupFiles: ["tests/setup.ts"],
    restoreMocks: true,
  },
});
