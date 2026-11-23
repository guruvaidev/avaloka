// Save & Execute must not say "Saved & executed" when the server says the run failed.
// The endpoint answers HTTP 200 for a failed run; the outcome is in `status`.
import { describe, expect, it } from "vitest";
import { saveExecuteOutcome } from "@/lib/save-execute-outcome";

const RAW_ERROR = [
  /Traceback \(most recent call last\)/,
  /File "[^"]*", line \d+/,
  /\b[A-Z]\w*(?:Error|Exception)\((?:'|")/,
  /\b[A-Z]\w*(?:Error|Exception)\s*:\s*\S/,
];

// Response bodies as server.py:save_and_execute_analysis returns them.
const executionFailed = {
  available: false,
  status: "execution_failed",
  validation_passed: true,
  execution_message:
    'Script failed with exit status 1\n\nTraceback (most recent call last):\n  File "/tmp/run/script.py", line 3, in main\nKeyError: \'revenue_2026\'',
  output_json: null,
  checkpoint_created: false,
};
const persistenceFailed = {
  available: false,
  status: "persistence_failed",
  validation_passed: true,
  execution_message: "Execution produced 10 rows, but saving the new output failed.",
  output_json: [{ a: 1 }],
  checkpoint_created: false,
};
const saved = { available: true, status: "success", validation_passed: true, output_json: [{ a: 1 }] };

describe("saveExecuteOutcome", () => {
  it("reports a failed run as failed, in plain words", () => {
    const outcome = saveExecuteOutcome(executionFailed);
    expect(outcome.kind).toBe("failed");
    if (outcome.kind !== "failed") return;
    expect(outcome.message).toMatch(/nothing was saved/);
    for (const rx of RAW_ERROR) expect(outcome.message).not.toMatch(rx);
  });

  it("reports a run whose new version could not be saved as failed", () => {
    const outcome = saveExecuteOutcome(persistenceFailed);
    expect(outcome.kind).toBe("failed");
    if (outcome.kind === "failed") expect(outcome.message).toMatch(/couldn't be saved/);
  });

  it("treats available:false as failure whatever the status says", () => {
    expect(saveExecuteOutcome({ ...saved, available: false }).kind).toBe("failed");
  });

  it("treats an unknown or missing status as failure, not success", () => {
    expect(saveExecuteOutcome({ available: true, status: "error" }).kind).toBe("failed");
    expect(saveExecuteOutcome({}).kind).toBe("failed");
    expect(saveExecuteOutcome(null).kind).toBe("failed");
  });

  it("keeps validation failures on the needs-fix path", () => {
    expect(saveExecuteOutcome({ available: false, status: "validation_failed", validation_passed: false }).kind).toBe(
      "needs_fix",
    );
  });

  it("reports the success statuses the server sends as saved", () => {
    for (const status of ["success", "succeeded", "completed", "dryrun", "SUCCESS"]) {
      expect(saveExecuteOutcome({ ...saved, status }).kind).toBe("saved");
    }
  });
});
