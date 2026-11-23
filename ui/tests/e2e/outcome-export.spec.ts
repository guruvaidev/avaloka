// B7 (docs/design/2.0-OUTCOMES.md §4.7, §7): the Outcome Report HTML export,
// shown the way the UI shows it, in <iframe sandbox="allow-scripts" srcdoc>,
// runs under its own CSP with no violation, no console error, no network
// request, and draws its charts.
//
// Needs no running stack: the export is a committed fixture, the analyst
// export of the report built from the seeded retail table (US state map,
// bridge, sell-through, basket affinity). tests/avaloka/outcomes/
// test_render.py::test_e2e_fixture_is_current holds it equal to what the
// renderer produces today; regenerate with
//   python tests/avaloka/outcomes/conftest.py
//
// Bite (recorded in the PR): drop the init script's hash from the CSP in
// render_html.py and regenerate; "Refused to execute inline script" turns
// this red.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const FIXTURE = fileURLToPath(new URL("./fixtures/outcome-export.html", import.meta.url));

test("outcome export renders in a sandboxed srcdoc frame with no CSP violation", async ({ page }) => {
  const html = readFileSync(FIXTURE, "utf8");
  const errors: string[] = [];
  const requests: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" || /Content Security Policy|Refused to/i.test(msg.text())) {
      errors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("request", (req) => {
    const url = req.url();
    if (!url.startsWith("data:") && !url.startsWith("about:")) requests.push(url);
  });

  await page.setContent(
    '<!doctype html><html><body style="margin:0">' +
      '<iframe id="f" sandbox="allow-scripts" referrerpolicy="no-referrer" ' +
      'style="width:1000px;height:2400px;border:0"></iframe></body></html>',
  );
  // The parent accepts only the frame's own height message, as OutcomeFrame will.
  await page.evaluate(() => {
    const frame = document.getElementById("f") as HTMLIFrameElement;
    (window as unknown as { heights: number[] }).heights = [];
    window.addEventListener("message", (event) => {
      if (event.source === frame.contentWindow && event.data?.type === "avaloka-report-height") {
        (window as unknown as { heights: number[] }).heights.push(event.data.h);
      }
    });
  });
  await page.evaluate((doc) => {
    (document.getElementById("f") as HTMLIFrameElement).srcdoc = doc;
  }, html);

  const frame = page.frames().find((f) => f !== page.mainFrame());
  expect(frame, "the srcdoc frame exists").toBeTruthy();
  await frame!.waitForLoadState("load");
  await expect
    .poll(() => page.evaluate(() => (window as unknown as { heights: number[] }).heights.length))
    .toBeGreaterThan(0);

  // The sandbox gives the frame an opaque origin: no storage, no parent access.
  const origin = await frame!.evaluate(() => window.origin);
  expect(origin).toBe("null");

  // Every chart drew something: a canvas per chart, each with painted pixels.
  const painted = await frame!.evaluate(() => {
    const out: number[] = [];
    for (const el of Array.from(document.querySelectorAll("[data-chart]"))) {
      const canvas = el.querySelector("canvas");
      if (!canvas) {
        out.push(0);
        continue;
      }
      const ctx = canvas.getContext("2d")!;
      const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let n = 0;
      for (let i = 3; i < data.length; i += 4) if (data[i] !== 0) n++;
      out.push(n);
    }
    return out;
  });
  expect(painted.length).toBe(4);
  for (const n of painted) expect(n).toBeGreaterThan(1000);

  // The what-if slider recomputes through outcome-runtime.js under the same CSP.
  const before = await frame!.locator('[data-out="y"]').textContent();
  await frame!.locator('input[data-driver="discount"]').fill("20");
  await expect.poll(() => frame!.locator('[data-out="y"]').textContent()).not.toBe(before);
  await expect(frame!.locator('input[data-driver="discount"]')).toHaveAttribute("aria-valuetext", "20.0%");

  expect(errors, errors.join("\n")).toEqual([]);
  expect(requests, requests.join("\n")).toEqual([]);
});
