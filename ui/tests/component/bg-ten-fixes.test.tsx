// @vitest-environment happy-dom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, screen } from "@testing-library/dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UserMessageEditor } from "../../src/components/analysis/MessageActions";
import { ChatResizeHandle } from "../../src/components/analysis/ChatResizeHandle";
import { KeyFindingsCard, KEY_FINDINGS_OPEN_KEY } from "../../src/components/dashboard/KeyFindingsCard";
import { NewProjectDialog } from "../../src/components/dashboard/NewProjectDialog";
import { InferencePanel } from "../../src/components/dashboard/InferencePanel";
import { RegisterModelModal } from "../../src/components/dashboard/RegisterModelModal";
import { CHAT_WIDTH_STEP } from "../../src/lib/chat-width";

vi.mock("react-remove-scroll", () => ({ RemoveScroll: ({ children }: { children: React.ReactNode }) => <>{children}</> }));

let root: Root;
let host: HTMLDivElement;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  localStorage.clear();
});
afterEach(() => {
  act(() => root.unmount());
  document.body.innerHTML = "";
});

describe("user message edit (issue 5)", () => {
  it("Edit + Save calls send with the new text", async () => {
    const send = vi.fn();
    await act(async () =>
      root.render(
        <UserMessageEditor content="old question" onResend={send}>
          {({ editButton }) => <div>{editButton}</div>}
        </UserMessageEditor>,
      ),
    );
    await act(async () => fireEvent.click(screen.getByLabelText("Edit")));
    const box = screen.getByLabelText("Edit message") as HTMLTextAreaElement;
    await act(async () => fireEvent.change(box, { target: { value: "new question" } }));
    await act(async () => fireEvent.click(screen.getByText("Save")));
    expect(send).toHaveBeenCalledWith("new question");
  });

  it("renders no Edit button without a resend action (AI messages)", async () => {
    await act(async () =>
      root.render(
        <UserMessageEditor content="ai text">
          {({ editButton }) => <div>{editButton}</div>}
        </UserMessageEditor>,
      ),
    );
    expect(screen.queryByLabelText("Edit")).toBeNull();
  });
});

describe("chat resize handle (issue 2)", () => {
  it("exposes separator semantics, steps with arrows and resets on double-click", async () => {
    const onResize = vi.fn();
    const onReset = vi.fn();
    await act(async () => root.render(<ChatResizeHandle width={400} viewport={1920} chatSide="right" onResize={onResize} onReset={onReset} />));
    const sep = screen.getByRole("separator");
    expect(sep.getAttribute("aria-orientation")).toBe("vertical");
    expect(sep.getAttribute("aria-valuenow")).toBe("400");
    fireEvent.keyDown(sep, { key: "ArrowLeft" });
    expect(onResize).toHaveBeenLastCalledWith(400 + CHAT_WIDTH_STEP);
    fireEvent.keyDown(sep, { key: "ArrowRight" });
    expect(onResize).toHaveBeenLastCalledWith(400 - CHAT_WIDTH_STEP);
    fireEvent.doubleClick(sep);
    expect(onReset).toHaveBeenCalled();
  });
});

describe("key findings card (issue 1)", () => {
  it("is expanded by default and remembers collapse", async () => {
    await act(async () => root.render(<KeyFindingsCard findings={["One", "Two"]} />));
    expect(screen.getByText("One")).toBeTruthy();
    await act(async () => fireEvent.click(screen.getByRole("button", { name: /Key findings/ })));
    expect(screen.queryByText("One")).toBeNull();
    expect(localStorage.getItem(KEY_FINDINGS_OPEN_KEY)).toBe("false");
  });
});

describe("new project dialog (issues 6 and 7)", () => {
  it("shows a loading state, then closes on success", async () => {
    let resolve!: () => void;
    const onCreate = vi.fn(() => new Promise<void>((r) => (resolve = r)));
    const onOpenChange = vi.fn();
    await act(async () => root.render(<NewProjectDialog open onOpenChange={onOpenChange} onCreate={onCreate} />));
    const name = screen.getByPlaceholderText("Project name") as HTMLInputElement;
    await act(async () => fireEvent.change(name, { target: { value: "test" } }));
    await act(async () => fireEvent.click(screen.getByText("Create")));
    expect(onCreate).toHaveBeenCalledWith("test", null);
    expect(name.readOnly).toBe(true);
    expect(screen.getByLabelText("Creating project")).toBeTruthy();
    await act(async () => resolve());
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("keeps the typed values when creation fails", async () => {
    const onCreate = vi.fn(() => Promise.reject(new Error("Backend said no")));
    const onOpenChange = vi.fn();
    await act(async () => root.render(<NewProjectDialog open onOpenChange={onOpenChange} onCreate={onCreate} />));
    const name = screen.getByPlaceholderText("Project name") as HTMLInputElement;
    await act(async () => fireEvent.change(name, { target: { value: "test" } }));
    await act(async () => fireEvent.click(screen.getByText("Create")));
    expect(onOpenChange).not.toHaveBeenCalledWith(false);
    expect((screen.getByPlaceholderText("Project name") as HTMLInputElement).value).toBe("test");
  });
});

describe("dark theme roots (issue 10)", () => {
  it("Register Model dialog has no hard-coded white background", async () => {
    await act(async () => root.render(<RegisterModelModal open onOpenChange={() => undefined} />));
    const html = document.body.innerHTML;
    expect(html).toContain("Register Model");
    expect(html).not.toMatch(/bg-white|#fff\b|text-gray-|bg-gray-/);
  });

  it("Inference panel root has no hard-coded white background", async () => {
    await act(async () => root.render(<QueryClientProvider client={new QueryClient()}><InferencePanel runId="run-1" raw={{}} /></QueryClientProvider>));
    const html = host.innerHTML;
    expect(html).toContain("bg-card");
    expect(html).not.toMatch(/bg-white|#fff\b|#ffffff|text-gray-|bg-gray-/);
  });
});
