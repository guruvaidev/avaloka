// The one end-to-end check: a signed-in user opens the dashboard, asks a
// question, and gets a drawn chart back.
//
// It needs a whole running stack. When any part of that is absent the spec
// SKIPS and says which part, rather than failing: a red test that only means
// "nothing was running" teaches people to ignore red tests.
import { existsSync } from "node:fs";
import { expect, test } from "@playwright/test";

const BASE_URL = process.env.E2E_BASE_URL;
const API_URL = process.env.E2E_API_URL;
const STORAGE_STATE = process.env.E2E_STORAGE_STATE;
const ANALYSIS_PATH = process.env.E2E_ANALYSIS_PATH ?? "/analysis";
const PROMPT =
  process.env.E2E_PROMPT ?? "Show the number of records for each category as a bar chart";

/** Any HTTP answer counts: this asks "is something there", not "is it healthy". */
async function reachable(url: string): Promise<boolean> {
  try {
    await fetch(url, { signal: AbortSignal.timeout(5_000) });
    return true;
  } catch {
    return false;
  }
}

async function missingEnvironment(): Promise<string | null> {
  if (!BASE_URL) return "E2E_BASE_URL is not set (no running UI to test against)";
  if (!API_URL) return "E2E_API_URL is not set (no backend to answer the prompt)";
  if (!STORAGE_STATE)
    return "E2E_STORAGE_STATE is not set (no signed-in session; login is by emailed link)";
  if (!existsSync(STORAGE_STATE)) return `E2E_STORAGE_STATE file not found: ${STORAGE_STATE}`;
  if (!(await reachable(BASE_URL))) return `UI is not reachable at ${BASE_URL}`;
  if (!(await reachable(API_URL))) return `backend is not reachable at ${API_URL}`;
  return null;
}

test.describe("prompt to chart", () => {
  test.beforeAll(async () => {
    const reason = await missingEnvironment();
    // The list reporter prints "skipped" without the annotation, so say why.
    if (reason) console.warn(`SKIPPED: e2e environment unavailable: ${reason}`);
    test.skip(reason !== null, `e2e environment unavailable: ${reason}`);
  });

  test("a prompt submitted from the dashboard renders a chart", async ({ page }) => {
    await page.goto("/dashboard");
    // An expired session is redirected to the login page. That is a broken
    // fixture, not an absent environment, so it fails loudly instead of skipping.
    await expect(page, "E2E_STORAGE_STATE no longer holds a valid session").toHaveURL(
      /\/dashboard/,
    );

    await page.goto(ANALYSIS_PATH);
    const composer = page.getByPlaceholder("Ask me anything...");
    await composer.fill(PROMPT);
    await composer.press("Enter");
    await expect(page.getByText(PROMPT).first()).toBeVisible();

    const chart = page.locator(".recharts-wrapper svg.recharts-surface").first();
    await expect(chart).toBeVisible({ timeout: 4 * 60_000 });
    // A frame with axes but no marks is not a rendered chart.
    const marks = page.locator(
      ".recharts-bar-rectangle, .recharts-line-curve, .recharts-area-area, .recharts-pie-sector, .recharts-scatter-symbol",
    );
    await expect(marks.first()).toBeVisible();
    await expect(
      page.getByText("This chart couldn't be drawn from the available data."),
    ).toHaveCount(0);
  });
});
