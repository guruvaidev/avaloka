// Guard: no UI-rendered string carries raw error output (a traceback, a stack
// frame, a raw exception). The components that show a failure are rendered with
// an error payload carrying exactly that, and the whole rendered text is swept.
// The patterns live here, not in src/lib/user-facing-error, so the guard does
// not share a blind spot with the code it checks.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

const getTaskRuns = vi.fn();
vi.mock("@/lib/api/backendApi", () => ({ backendApi: { getTaskRuns: (...a: unknown[]) => getTaskRuns(...a) } }));

import { RunHistoryDrawer } from "@/components/dashboard/RunHistoryDrawer";

const RAW_ERROR = [
  /Traceback \(most recent call last\)/,
  /File "[^"]*", line \d+/,
  /\b[A-Z]\w*(?:Error|Exception)\((?:'|")/,
  /\b[A-Z]\w*(?:Error|Exception)\s*:\s*\S/,
];

const STDERR = [
  "Traceback (most recent call last):",
  '  File "/tmp/avaloka/run_77/script.py", line 12, in main',
  "    return df['revenue_2026'].astype(float)",
  "ValueError: could not convert string to float: 'n/a'",
].join("\n");

function rawErrorsIn(text: string): string[] {
  return RAW_ERROR.filter((rx) => rx.test(text)).map(String);
}

describe("the guard's patterns", () => {
  it("catch each raw shape and pass plain words", () => {
    for (const s of [STDERR, "ValueError('bad')", "KeyError: 'revenue_2026'", '  File "x.py", line 3, in main']) {
      expect(rawErrorsIn(s).length).toBeGreaterThan(0);
    }
    expect(rawErrorsIn("The scheduled analysis failed: the code asked for a column the data doesn't have.")).toEqual([]);
  });
});

describe("RunHistoryDrawer with failed runs", () => {
  it("shows no traceback in a run's log or error", async () => {
    getTaskRuns.mockResolvedValue([
      {
        run_id: "r2",
        run_number: 2,
        status: "failure",
        error: "ValueError: could not convert string to float: 'n/a'",
        result: { logs: ["[error] could not convert string to float: 'n/a'", STDERR] },
      },
      { run_id: "r1", run_number: 1, status: "failure", error: STDERR, result: null },
    ]);
    const { container } = render(<RunHistoryDrawer taskId="sched-1" sessionId="s-1" onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("Run #2")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Run #2"));
    fireEvent.click(screen.getByText("Run #1"));
    const text = container.textContent ?? "";
    expect(rawErrorsIn(text)).toEqual([]);
    expect(text).toContain("The run stopped with an error");
  });
});

// The two route files are too large to mount in a unit test; their failure
// displays are checked where they read the server's text (structural).
describe("route failure displays", () => {
  const read = (rel: string) => readFileSync(join(process.cwd(), rel), "utf8");

  it("scheduled-analysis shows the plain message, not the Celery traceback", () => {
    const page = read("src/routes/scheduled-analysis.tsx");
    expect(page).not.toMatch(/>Traceback</);
    expect(page).not.toMatch(/setOutputTraceback\(res\?\.traceback/);
    expect(page).toMatch(/setOutputTraceback\(userFacingError\(/);
  });

  it("analysis code panel passes validation feedback through the guard", () => {
    const page = read("src/routes/analysis.tsx");
    expect(page).not.toMatch(/^\s*\{validationFeedback\}\s*$/m);
    expect(page).toMatch(/\{userFacingError\(validationFeedback,/);
  });
});
