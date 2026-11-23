// @vitest-environment happy-dom
// Owner ruling 2026-11-03: the Free upgrade prompt says only what a paid plan
// adds according to the edition matrix. Renders the real dialog.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: "free" }, isLoading: false }) }));

import { UpgradePlanDialog } from "@/components/dashboard/UpgradeGate";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
beforeEach(() => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

describe("Upgrade your plan dialog", () => {
  it("names the blocked feature, then only what Professional adds over Free", () => {
    act(() => root.render(<UpgradePlanDialog open feature="Scheduling analyses" onClose={() => {}} />));
    const text = document.body.textContent!;
    expect(text).toContain(
      "Scheduling analyses is available on paid plans. Upgrade to invite people and form teams, comment on analyses, share analyses, schedule analyses and use Reports.",
    );
  });

  it.each(["insights", "projects", "datasets"])("no longer promises %s, which Free has or no capability backs", (word) => {
    act(() => root.render(<UpgradePlanDialog open feature="Scheduling analyses" onClose={() => {}} />));
    expect(document.body.textContent!.toLowerCase()).not.toContain(`unlock ${word}`);
    expect(document.body.textContent!.toLowerCase()).not.toMatch(new RegExp(`upgrade[^.]*\\b${word}\\b`));
  });
});
