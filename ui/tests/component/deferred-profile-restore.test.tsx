// rweb18c: a fresh stroke upload, the API at full_profile with a profile
// overview, 360 s and a reopened modal later the Auto Insights modal still
// showed the 6 upload-time charts. The chips the modal reads had been restored
// from the sessionStorage batch: JSON copies, never the upload's own config
// object, and the swap matched by identity. This goes through that restore
// path: the route's key ("analysis:batch"), JSON in and out.
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { chipsWithFinishedProfile } from "@/lib/deferred-profile";
import { AutoInsightsModal } from "@/components/dashboard/AutoInsightsModal";
import profile from "../fixtures/commentary/rweb18b-finished-profile.json";

// AutoInsightsModal reads the plan for its auto_insights gate (every plan has it).
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: "free" }, isLoading: false }) }));
vi.mock("@/integrations/supabase/external-client", () => ({
  supabase: { auth: { getSession: vi.fn(), onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })) } },
}));
vi.mock("@/integrations/supabase/client", () => ({
  supabase: { auth: { getSession: vi.fn(), onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })) } },
}));

/** What the upload handler stores (analysis.tsx storeDatasetBatch) and the restore effect reads back. */
function throughTheBatch<C>(chips: C[], selectedId: string): C[] {
  sessionStorage.setItem("analysis:batch", JSON.stringify({ chips, selectedId }));
  return JSON.parse(sessionStorage.getItem("analysis:batch")!).chips as C[];
}

describe("finished profile after a restored batch", () => {
  afterEach(() => sessionStorage.clear());
  const upload = profile.upload_config;          // the upload response's own config object
  const finished = profile.finished_config;

  it("reaches the modal when the chips are the sessionStorage copies", () => {
    const uploaded = [{ id: "ds1", name: "stroke.csv", visualizationConfig: upload as unknown, visualizationStatus: "ready" }];
    const restored = throughTheBatch(uploaded, "ds1");
    expect(restored[0].visualizationConfig).not.toBe(upload); // a copy, as live
    const chips = chipsWithFinishedProfile(restored, "ds1", upload, finished);
    expect(chips[0].visualizationConfig).toBe(finished);
    const datasets = chips.map((c) => ({ id: c.id, name: c.name, config: c.visualizationConfig, status: c.visualizationStatus }));
    const { baseElement } = render(
      <AutoInsightsModal open onOpenChange={() => {}} config={finished} status="ready" datasets={datasets} primaryDatasetId="ds1" />,
    );
    // The modal opens on the Stack view; the profile's readiness sits on Overview.
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    expect(baseElement.textContent).toContain("Data readiness");
    expect(baseElement.textContent).toContain(finished.profile_overview.readiness_score_segments[0].text);
  });

  it("still leaves a restored chip whose config another flow changed", () => {
    const regenerated = { charts: [], visualization_status: "ready", note: "user regenerated" };
    const restored = throughTheBatch([{ id: "ds1", name: "stroke.csv", visualizationConfig: regenerated as unknown }], "ds1");
    const chips = chipsWithFinishedProfile(restored, "ds1", upload, finished);
    expect(chips[0].visualizationConfig).toEqual(regenerated);
  });
});
