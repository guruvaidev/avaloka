// rc18b live regrade (rweb18b) items on the UI side.
// - The trace step list said "safe" for the claim verifier while its detail said "checked".
// - After a fresh upload the Auto Insights modal kept the upload-time charts: it reads each
//   dataset's chip, and the finished profile only reached the page's own config.
// - With no saved analysis (quickstart) the chart explainer told the user to reopen it
//   "from your analyses list", which local mode never has.
import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import * as deferred from "@/lib/deferred-profile";
import { AutoInsightsModal } from "@/components/dashboard/AutoInsightsModal";
import { TraceTimeline } from "@/components/agent-trace/TraceSteps";
import { InsightsAvatar } from "@/components/insights-avatar/InsightsAvatar";
import { InsightsAvatarProvider } from "@/components/insights-avatar/InsightsAvatarContext";
import type { TraceTurn } from "@/lib/agent-trace/types";
import profile from "../fixtures/commentary/rweb18b-finished-profile.json";

// AutoInsightsModal reads the plan for its auto_insights gate (every plan has it).
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: "free" }, isLoading: false }) }));
vi.mock("@/integrations/supabase/external-client", () => ({
  supabase: {
    auth: {
      getSession: vi.fn(async () => ({ data: { session: { access_token: "t" } } })),
      onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })),
    },
  },
}));
vi.mock("@/integrations/supabase/client", () => ({
  supabase: {
    auth: {
      getSession: vi.fn(async () => ({ data: { session: { access_token: "t" } } })),
      onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })),
    },
  },
}));

describe("Trace step list", () => {
  it("says the verifier checked its claims, not that they are safe", () => {
    const turn = {
      turn_id: "t1",
      handoffs: [],
      steps: [
        {
          kind: "agent", step_id: "v", parent_id: null, turn_id: "t1", seq: 4, node: "verify_claims",
          agent: "Claim verifier", decision: { type: "evidence", value: "safe", reason: "Checked 5 claim(s) against the numbers" },
        },
        {
          kind: "agent", step_id: "x", parent_id: null, turn_id: "t1", seq: 5, node: "execute_code",
          agent: "Executor", decision: { type: "check", value: "pass" },
        },
      ],
    } as unknown as TraceTurn;
    const { container } = render(<TraceTimeline turn={turn} onSelect={() => {}} />);
    const text = container.textContent ?? "";
    expect(text).toContain("Claim verifier");
    expect(text).toContain("checked");
    expect(text).not.toMatch(/\bsafe\b/);
    expect(text).toContain("pass"); // other decisions keep their value
  });
});

describe("Auto Insights after a deferred profile", () => {
  const upload = profile.upload_config;
  const finished = profile.finished_config;
  const chips = [{ id: "ds1", name: "stroke.csv", visualizationConfig: upload as unknown, visualizationStatus: "ready" }];
  const swap = (deferred as Record<string, unknown>).chipsWithFinishedProfile as
    | ((c: typeof chips, id: string, uploadCfg: unknown, next: unknown) => typeof chips)
    | undefined;

  it("shows the finished profile's badges and column meanings in the modal", () => {
    // What the page does when the profile lands: its own config is the finished one;
    // the modal reads the chips.
    const next = swap ? swap(chips, "ds1", upload, finished) : chips;
    const datasets = next.map((c) => ({ id: c.id, name: c.name, config: c.visualizationConfig, status: c.visualizationStatus }));
    const { baseElement } = render(
      <AutoInsightsModal open onOpenChange={() => {}} config={finished} status="ready" datasets={datasets} primaryDatasetId="ds1" />,
    );
    // The modal opens on the Stack view; the profile's badges sit on Overview and
    // the column meanings in the Column statistics tab.
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    const overview = baseElement.textContent ?? "";
    expect(overview).toContain("Data readiness");
    expect(overview).toContain(String(finished.profile_overview.readiness_score_segments[0].text));
    fireEvent.click(screen.getByRole("button", { name: "Column statistics" }));
    const shown = finished.column_stats.find((c) => c.role !== "identifier")!; // ID columns are hidden by default
    expect(baseElement.textContent ?? "").toContain(shown.meaning_segments![0].text);
  });

  it("leaves a chip whose config another flow already replaced", () => {
    expect(swap).toBeTypeOf("function");
    const regenerated = { charts: [], visualization_status: "ready" };
    const own = [{ ...chips[0], visualizationConfig: regenerated as unknown }];
    expect(swap!(own, "ds1", upload, finished)[0].visualizationConfig).toBe(regenerated);
    expect(swap!(chips, "other", upload, finished)).toBe(chips);
  });
});

describe("Chart explainer without a saved analysis", () => {
  // The avatar's panel is controlled by Auto Insights (each tab applies its own default).
  function Host() {
    const [open, setOpen] = useState(false);
    return (
      <InsightsAvatarProvider>
        <InsightsAvatar aid={null} datasetId="ds1" datasetName="stroke.csv" charts={[]} open={open} onOpenChange={setOpen} />
      </InsightsAvatarProvider>
    );
  }
  it("says the page is not a saved analysis instead of pointing at an analyses list", async () => {
    render(<Host />);
    // The launcher's accessible name is "Open Insights Assistant"; "Ask about these charts" is its title.
    fireEvent.click(screen.getByRole("button", { name: "Open Insights Assistant" }));
    fireEvent.change(screen.getByLabelText("Ask about the charts"), { target: { value: "Which group is highest?" } });
    fireEvent.click(screen.getByLabelText("Send question"));
    const log = await screen.findByRole("log");
    expect(log.textContent).not.toContain("analyses list");
    expect(log.textContent).toContain("saved analysis");
  });
});
