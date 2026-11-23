// The success toast must not be reachable for a run the server reports as failed.
// Kept apart from save-execute-outcome.test.ts so it loads on a tree without the helper.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// Wiring (structural): the handler in routes/analysis.tsx must decide from the
// response's status before it can reach the success toast.
describe("handleSaveAndExecute wiring", () => {
  const page = readFileSync(join(process.cwd(), "src/routes/analysis.tsx"), "utf8");
  const start = page.indexOf("const handleSaveAndExecute = async");
  const handler = page.slice(start, page.indexOf("\n  };\n", start));

  it("checks the server's outcome, not just HTTP 200, before the success toast", () => {
    expect(start).toBeGreaterThan(-1);
    const success = handler.indexOf('toast.success(');
    expect(success).toBeGreaterThan(-1);
    const decided = handler.indexOf("saveExecuteOutcome(body)");
    expect(decided, "the handler never reads the run's status").toBeGreaterThan(-1);
    const failedReturn = handler.search(/outcome\.kind === "failed"\) \{\s*toast\.error\(outcome\.message\);\s*return;/);
    expect(failedReturn, "a failed run must return with an error toast").toBeGreaterThan(-1);
    expect(decided).toBeLessThan(success);
    expect(failedReturn).toBeLessThan(success);
  });
});
