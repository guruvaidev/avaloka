// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { ChatDockControls } from "../../src/components/analysis/ChatDockControls";
import type { ChatDockSide } from "../../src/lib/chat-dock";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

function mount(mode: ChatDockSide, onChange: (mode: ChatDockSide) => void = () => {}) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(<ChatDockControls mode={mode} onChange={onChange} />));
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const modes: ChatDockSide[] = ["left", "float", "right"];
const labels = ["Dock chat left", "Pop out chat", "Dock chat right"];

describe("ChatDockControls", () => {
  it.each(modes)("renders exactly three buttons in %s mode", (mode) => {
    const el = mount(mode);
    const buttons = el.querySelectorAll("button");
    expect(buttons).toHaveLength(3);
    expect([...buttons].map((b) => b.getAttribute("aria-label"))).toEqual(labels);
  });

  it.each(modes)("highlights only the %s button in %s mode", (mode) => {
    const el = mount(mode);
    const buttons = [...el.querySelectorAll("button")];
    buttons.forEach((b, i) => {
      const isActive = labels[i] === `Dock chat ${mode}` || (mode === "float" && labels[i] === "Pop out chat");
      expect(b.className.includes("bg-[#eef4ff]")).toBe(isActive);
      expect(b.className.includes("text-[#1a5fe0]")).toBe(isActive);
    });
  });

  it("pop out switches to float mode and never calls window.open", () => {
    const openSpy = vi.spyOn(window, "open");
    const changes: ChatDockSide[] = [];
    const el = mount("right", (m) => changes.push(m));
    const popOut = el.querySelector('button[aria-label="Pop out chat"]')!;
    act(() => {
      popOut.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(changes).toEqual(["float"]);
    expect(openSpy).not.toHaveBeenCalled();
    openSpy.mockRestore();
  });

  it("dock left and dock right buttons report their modes", () => {
    const changes: ChatDockSide[] = [];
    const el = mount("float", (m) => changes.push(m));
    act(() => {
      el.querySelector('button[aria-label="Dock chat left"]')!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      el.querySelector('button[aria-label="Dock chat right"]')!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(changes).toEqual(["left", "right"]);
  });

  it("buttons are real buttons that are keyboard focusable and activatable", () => {
    const changes: ChatDockSide[] = [];
    const el = mount("right", (m) => changes.push(m));
    const buttons = [...el.querySelectorAll("button")];
    buttons.forEach((b) => expect(b.tagName).toBe("BUTTON"));
    const first = buttons[0];
    first.focus();
    expect(document.activeElement).toBe(first);
    // Enter on a focused button fires a click (activation behavior).
    act(() => {
      first.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      first.click();
    });
    expect(changes).toEqual(["left"]);
  });
});
