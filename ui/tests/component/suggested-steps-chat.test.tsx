// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

vi.mock("@/components/dashboard/dynamicChart", () => ({ DynamicChart: () => null, normalizeVizConfig: () => [] as never[] }));
vi.mock("@/components/dashboard/auto-insights/AutoInsightsPanels", async (importOriginal) => ({ ...(await importOriginal<object>()), ColumnStatisticsTab: () => null }));

import { SuggestedStepList, askInChatAndClose } from "../../src/components/dashboard/AutoInsightsModal";

(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | null = null;
let host: HTMLDivElement | null = null;
afterEach(() => { act(() => root?.unmount()); host?.remove(); root = null; host = null; });

const items = [
  { type: "Q", text: "Correlate subscriber growth with monthly earnings across categories" },
  { type: "C", text: " Remove constant Abbreviation column " },
];

function mount(onAsk: (t: string) => boolean | void) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<SuggestedStepList items={items} onAsk={onAsk} />));
  return Array.from(host.querySelectorAll("button"));
}

describe("Suggested next steps hand-off", () => {
  it("Q step: closes Auto Insights and asks once with the text (no badge)", () => {
    const send = vi.fn(() => true);
    const close = vi.fn();
    const rows = mount(askInChatAndClose(send, close));
    expect(rows[0].getAttribute("aria-label")).toBe(`Ask in chat: ${items[0].text}`);
    act(() => rows[0].click());
    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith(items[0].text);
    expect(close).toHaveBeenCalledWith(false);
  });

  it("C step: sends trimmed text without the C badge", () => {
    const send = vi.fn(() => true);
    const close = vi.fn();
    const rows = mount(askInChatAndClose(send, close));
    act(() => rows[1].click());
    expect(send).toHaveBeenCalledWith("Remove constant Abbreviation column");
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("double click sends once and disables every row", () => {
    const send = vi.fn(() => true);
    const rows = mount(send);
    act(() => { rows[0].click(); rows[0].click(); rows[1].click(); });
    expect(send).toHaveBeenCalledTimes(1);
    expect(rows.every((r) => r.disabled)).toBe(true);
  });

  it("refused hand-off keeps Auto Insights open and rows usable", () => {
    const close = vi.fn();
    const rows = mount(askInChatAndClose(() => false, close));
    act(() => rows[0].click());
    expect(close).not.toHaveBeenCalled();
    expect(rows[0].disabled).toBe(false);
  });
});
