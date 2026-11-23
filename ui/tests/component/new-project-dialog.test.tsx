// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { fireEvent, getByRole, getByLabelText } from "@testing-library/dom";
import { NewProjectDialog } from "../../src/components/dashboard/NewProjectDialog";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));
let root: Root | undefined;
let host: HTMLDivElement | undefined;
afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = undefined;
  host = undefined;
});

function mount(onCreate = vi.fn(async () => {})) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  const onOpenChange = vi.fn();
  act(() => root?.render(<NewProjectDialog open onOpenChange={onOpenChange} onCreate={onCreate} />));
  return { onCreate, onOpenChange };
}

function typeName(value: string) {
  const input = getByLabelText(document.body, "Project name") as HTMLInputElement;
  act(() => {
    fireEvent.change(input, { target: { value } });
  });
  return input;
}

describe("New project submission", () => {
  it("labels Create and disables it for empty and whitespace-only names", () => {
    mount();
    const button = getByRole(document.body, "button", { name: "Create" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.getAttribute("aria-disabled")).toBe("true");
    typeName("   ");
    expect(button.disabled).toBe(true);
  });

  it("enables a valid name, submits once, and shows Creating while pending", async () => {
    let finish: (() => void) | undefined;
    const onCreate = vi.fn(() => new Promise<void>((resolve) => { finish = resolve; }));
    const { onOpenChange } = mount(onCreate);
    typeName(" test ");
    const button = getByRole(document.body, "button", { name: "Create" }) as HTMLButtonElement;
    expect(button.disabled).toBe(false);
    act(() => button.click());
    const pending = getByRole(document.body, "button", { name: "Creating…" }) as HTMLButtonElement;
    expect(pending.disabled).toBe(true);
    expect(pending.getAttribute("aria-disabled")).toBe("true");
    act(() => pending.click());
    expect(onCreate).toHaveBeenCalledTimes(1);
    expect(onCreate).toHaveBeenCalledWith("test", null);
    await act(async () => { finish?.(); });
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("uses the same submission handler for form submission and retains input on failure", async () => {
    const onCreate = vi.fn(async () => { throw new Error("creation failed"); });
    const { onOpenChange } = mount(onCreate);
    const input = typeName("test");
    await act(async () => { input.form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); });
    expect(onCreate).toHaveBeenCalledWith("test", null);
    expect(onCreate).toHaveBeenCalledTimes(1);
    expect(onOpenChange).not.toHaveBeenCalled();
    expect(input.value).toBe("test");
    expect((getByRole(document.body, "button", { name: "Create" }) as HTMLButtonElement).disabled).toBe(false);
  });
});