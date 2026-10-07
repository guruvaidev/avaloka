// End-to-end config. There is no `webServer` block on purpose: the app is only
// meaningful with a backend, Supabase and a signed-in user behind it, so the
// spec targets an already-running deployment named by E2E_BASE_URL and skips
// itself -- with the reason -- when that environment is not there.
// See tests/README.md for the variables.
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "tests/e2e",
  outputDir: "test-results",
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  // One prompt is a full agent run: plan, execute, validate, visualise.
  timeout: 5 * 60_000,
  use: {
    baseURL: process.env.E2E_BASE_URL,
    storageState: process.env.E2E_STORAGE_STATE || undefined,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
