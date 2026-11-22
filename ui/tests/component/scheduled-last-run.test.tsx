import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";

// The page's shell, plan gate and API are stand-ins; the table and grid under
// test are the real Scheduled Analysis page.
vi.mock("@/components/dashboard/DashboardShell", () => ({
  DashboardShell: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));
vi.mock("@/components/dashboard/UpgradeGate", () => ({
  useUpgradeGate: () => ({ blocked: false, dialog: null }),
}));
vi.mock("@/components/dashboard/RunHistoryDrawer", () => ({ RunHistoryDrawer: () => null }));
vi.mock("@/lib/api/scheduled-tasks", () => ({
  sendScheduledChat: vi.fn(),
  getActiveSession: () => ({ sessionId: "sess-1" }),
}));
const listTasks = vi.fn();
vi.mock("@/lib/api/backendApi", () => ({
  backendApi: {
    listTasks: (...a: unknown[]) => listTasks(...a),
    getTaskStatus: async () => ({ status: "scheduled" }),
  },
}));

import { Route } from "../../src/routes/scheduled-analysis";
import * as cache from "../../src/lib/api/scheduled-prompt-cache";

const Page = (Route as unknown as { options: { component: () => JSX.Element } }).options
  .component;

// A just-created task as GET /tasks reported it on develop-1.6 @ 7fd77f09:
// RedBeat's creation-time anchor in last_run_at, and no runs.
const NEVER_RAN = {
  id: "389885c2-c9c5-443a-ba44-7134d0a0adbc",
  schedule: { minute: "0", hour: "9", day_of_week: "1" },
  max_runs: -1,
  total_run_count: 0,
  last_run_at: new Date().toISOString(),
};

async function renderPage(task: Record<string, unknown>) {
  listTasks.mockResolvedValue([task]);
  await act(async () => {
    render(<Page />);
  });
  await screen.findByText(/Ran \d+ time/);
}

describe("Scheduled Analysis: a never-run task shows no last-run time", () => {
  it("table view: Last Run says Never next to Ran 0 times", async () => {
    await renderPage(NEVER_RAN);
    const row = screen.getByText("Ran 0 times").closest("tr")!;
    expect(within(row).queryByText(/\bago\b|^in \d/)).toBeNull();
    expect(within(row).getByText("Never")).toBeTruthy();
  });

  it("grid view: the card's last-run line says Never", async () => {
    await renderPage(NEVER_RAN);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /grid/i }));
    });
    const line = screen.getByText(/^Last run:/);
    expect(line.textContent).toBe("Last run: Never");
  });

  it("a task that ran shows how long ago", async () => {
    await renderPage({
      ...NEVER_RAN,
      total_run_count: 2,
      last_run_at: new Date(Date.now() - 3 * 3600_000).toISOString(),
    });
    const row = screen.getByText("Ran 2 times").closest("tr")!;
    expect(within(row).getByText("3h ago")).toBeTruthy();
  });
});

describe("lastRunLabel", () => {
  it("is Never without a run, whatever timestamp came with it", () => {
    const lastRunLabel = (cache as Record<string, unknown>).lastRunLabel as typeof cache.lastRunLabel;
    expect(typeof lastRunLabel).toBe("function");
    expect(lastRunLabel(null, 0)).toBe("Never");
    expect(lastRunLabel(new Date().toISOString(), 0)).toBe("Never");
    expect(lastRunLabel(undefined, 3)).toBe("Never");
  });
});
