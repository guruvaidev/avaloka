// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { ChartEmptyState } from "../../src/components/dashboard/data-view/ChartEmptyState";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

function mount(onAsk: () => void = () => {}) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<ChartEmptyState onAsk={onAsk} />));
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

describe("Chart tab empty state", () => {
  it("does not render a 'Build a chart' button", () => {
    const el = mount();
    expect(el.textContent).not.toContain("Build a chart");
    expect(el.querySelector("button")).not.toBeNull();
    expect([...el.querySelectorAll("button")].some((b) => b.textContent?.includes("Build a chart"))).toBe(false);
  });

  it("tells the user to ask in the chat", () => {
    const el = mount();
    expect(el.textContent).toContain("No chart yet. Ask in the chat to create one.");
  });

  it("keeps the 'Ask in chat' button and wires it up", () => {
    const onAsk = vi.fn();
    const el = mount(onAsk);
    const ask = [...el.querySelectorAll("button")].find((b) => b.textContent?.includes("Ask in chat"));
    expect(ask).toBeDefined();
    act(() => {
      ask!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(onAsk).toHaveBeenCalledTimes(1);
  });
});
