// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { TopChartsSection, TopKeyFindings } from "../../src/components/dashboard/multi-dataset/MultiDatasetInsights";
import { dedupeFindingGroups } from "../../src/lib/multi-dataset-view";
import { normalizeFinding, splitAtKeyFindings } from "../../src/lib/workspace-findings";

(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | null = null;
let host: HTMLElement | null = null;

function mount(node: React.ReactNode) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(node));
  return host;
}
const button = (el: HTMLElement, text: string) =>
  Array.from(el.querySelectorAll("button")).find((b) => b.textContent?.trim().startsWith(text)) as HTMLButtonElement | undefined;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
});

describe("chat Key findings View more / View less", () => {
  const picks = [1, 2, 3, 4, 5].map((i) => ({ text: `Top ${i}`, datasetName: `${i}. Sheet` }));
  const groups = dedupeFindingGroups([
    { id: "a", name: "3. Agents", findings: ["Agents finding", "Shared finding."] },
    { id: "b", name: "4. Billing", findings: ["shared   finding", "Billing finding"] },
  ]);

  it("shows the 5 picks with sheet names and collapses by default", () => {
    const el = mount(<TopKeyFindings picks={picks} groups={groups} />);
    expect(el.querySelectorAll("li")).toHaveLength(5);
    expect(el.textContent).toContain("· 1. Sheet");
    expect(el.textContent).not.toContain("Agents finding");
    expect(button(el, "View more by sheet")).toBeTruthy();
  });

  it("expands grouped findings and collapses again with View less", () => {
    const el = mount(<TopKeyFindings picks={picks} groups={groups} />);
    act(() => button(el, "View more by sheet")!.click());
    expect(el.textContent).toContain("3. Agents");
    act(() => button(el, "3. Agents")!.click());
    expect(el.textContent).toContain("Agents finding");
    act(() => button(el, "View less")!.click());
    expect(el.textContent).not.toContain("3. Agents");
    expect(button(el, "View more by sheet")).toBeTruthy();
  });

  it("never duplicates a finding across sheets (normalised text)", () => {
    const all = groups.flatMap((g) => g.findings).map(normalizeFinding);
    expect(new Set(all).size).toBe(all.length);
    expect(all).toHaveLength(3);
  });

  it("keeps Watch out for / Try asking after the key findings", () => {
    const parts = splitAtKeyFindings("Hello\n\n**Key findings**\n- one\n- two\n\n**Watch out for**\n- x\n\n**Try asking**\n- y");
    expect(parts.found).toBe(true);
    expect(parts.before.trim()).toBe("Hello");
    expect(parts.after.startsWith("**Watch out for**")).toBe(true);
  });
});

describe("Chart tab View all charts by sheet / View less", () => {
  const groups = [
    { id: "a", name: "3. Agents", count: 3, emptyText: "", cards: <p>agents-cards</p> },
    { id: "s", name: "Summary", count: 0, emptyText: "No charts for this sheet.", cards: null },
  ];

  it("shows the top heading and keeps sheet groups collapsed by default", () => {
    const el = mount(<TopChartsSection heading="Top insights across 10 sheets" topCards={<p>top-cards</p>} groups={groups} total={3} />);
    expect(el.textContent).toContain("Top insights across 10 sheets");
    expect(el.textContent).toContain("top-cards");
    expect(el.textContent).not.toContain("3. Agents · 3 charts");
    expect(button(el, "View all charts by sheet (3)")).toBeTruthy();
  });

  it("expands per-sheet groups and collapses with View less", () => {
    const el = mount(<TopChartsSection heading="Top insights across 10 sheets" topCards={<p>top-cards</p>} groups={groups} total={3} />);
    act(() => button(el, "View all charts by sheet")!.click());
    expect(el.textContent).toContain("3. Agents · 3 charts");
    act(() => button(el, "Summary")!.click());
    expect(el.textContent).toContain("No charts for this sheet.");
    act(() => button(el, "View less")!.click());
    expect(el.textContent).not.toContain("3. Agents · 3 charts");
  });
});
