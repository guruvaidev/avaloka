// @vitest-environment happy-dom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { fireEvent, screen } from "@testing-library/dom";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { FidelitySwitch } from "../../src/components/analysis/FidelitySwitch";
import { fidelityApi, FidelityError, FIDELITY_POLL_MS, fidelityReplyDiffers, needsFidelityPolling, isFullDatasetTask, type FidelityState } from "../../src/lib/analysis-fidelity";
import { backendApi } from "../../src/lib/api/backendApi";

const mocks = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }));
vi.mock("sonner", () => ({ toast: mocks }));
vi.mock("../../src/integrations/supabase/external-client", () => ({ supabase: { auth: { getSession: async () => ({ data: { session: { access_token: "test-token" } } }) } } }));
// Native controls exercise selection behavior without Radix pointer geometry in happy-dom.
vi.mock("../../src/components/ui/select", async () => {
  const { createContext, useContext } = await import("react");
  const Context = createContext({ value: "", change: (_v: string) => {}, disabled: false });
  return {
    Select: ({ children, value, onValueChange, disabled = false }: { children: ReactNode; value: string; onValueChange: (v: string) => void; disabled?: boolean }) => <Context.Provider value={{ value, change: onValueChange, disabled }}>{children}</Context.Provider>,
    SelectTrigger: ({ children, ...props }: { children: ReactNode }) => { const c = useContext(Context); return <select {...props} value={c.value} disabled={c.disabled} onChange={(e) => c.change(e.target.value)}>{children}<option value={c.value}>{c.value}</option></select>; },
    SelectValue: () => null,
    SelectContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
    SelectItem: ({ children, value, disabled }: { children: ReactNode; value: string; disabled?: boolean }) => { const c = useContext(Context); return <button disabled={disabled} onClick={() => c.change(value)}>{children}</button>; },
  };
});
const fixture = (patch: Partial<FidelityState> = {}): FidelityState => ({ dataset_id: "d1", analysis_fidelity: "quick_sample", selected_sample_name: null, file_size_bytes: 100, is_large_dataset: false, analysis_task_id: null, options: [
  { value: "quick_sample", label: "Quick sample", enabled: true, detail: "Fast" },
  { value: "portfolio_samples", label: "Portfolio samples", enabled: true, status: "ready", samples: ["balanced", "tail"], detail: "Representative" },
  { value: "entire_dataset", label: "Full dataset", enabled: true, runs_as_job: true, detail: "All rows" },
], ...patch });
let root: Root;
const sample = vi.fn(async () => {});
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
beforeEach(() => { const host = document.createElement("div"); document.body.appendChild(host); root = createRoot(host); vi.clearAllMocks(); });
afterEach(() => { act(() => root.unmount()); document.body.innerHTML = ""; vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });
const render = async (props = {}) => { await act(async () => root.render(<FidelitySwitch datasetId="d1" sessionId="s1" threadId="t1" onSampleChange={sample} {...props} />)); };
const click = async (name: string) => { await act(async () => fireEvent.click(screen.getByRole("button", { name }))); };

describe("fidelity transport and chat payload", () => {
  it("reuses proxy base/auth/session and exact GET/PUT/status paths", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify(fixture()))); vi.stubGlobal("fetch", fetch);
    await fidelityApi.get("d/1", "s1"); await fidelityApi.put("d/1", "s1", { analysis_fidelity: "portfolio_samples", selected_sample_name: "tail" }); await fidelityApi.status("d/1", "s1");
    expect(fetch.mock.calls).toHaveLength(3);
    const calls = fetch.mock.calls as unknown as [string, RequestInit][];
    expect(new Headers(calls[0][1].headers).get("X-Backend-Path")).toBe("/api/datasets/d%2F1/fidelity");
    expect(new Headers(calls[0][1].headers).get("X-Avaloka-Session")).toBe("s1");
    expect(new Headers(calls[0][1].headers).get("Authorization")).toBe("Bearer test-token");
    expect(new Headers(calls[0][1].headers).get("X-Backend-Base")).toBeTruthy();
    expect(calls[1][1].method).toBe("PUT");
    expect(JSON.parse(String(calls[1][1].body))).toEqual({ analysis_fidelity: "portfolio_samples", selected_sample_name: "tail" });
    expect(new Headers(calls[2][1].headers).get("X-Backend-Path")).toBe("/api/datasets/d%2F1/background-task-status");
  });
  it("removes mode and sample even when legacy callers supply them, preserving other fields", async () => {
    const fetch = vi.fn(async () => new Response("{}")); vi.stubGlobal("fetch", fetch);
    await backendApi.sendMessage("t1", "s1", "question", ["d1"], "entire_dataset", "tail", { dataset_id: "d1" });
    const calls = fetch.mock.calls as unknown as [string, RequestInit][];
    expect(JSON.parse(String(calls[0][1].body))).toEqual({ role: "user", content: "question", stream: false, dataset_ids: ["d1"], metadata: { dataset_id: "d1" } });
  });
  it("preserves backend error detail and status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Not ready" }), { status: 422 })));
    await expect(fidelityApi.put("d1", "s1", { analysis_fidelity: "entire_dataset" })).rejects.toMatchObject({ status: 422, message: "Not ready" });
  });
});
describe("backend-owned selector", () => {
  it("hides silently on collaborator 404", async () => {
    vi.spyOn(fidelityApi, "get").mockRejectedValue(new FidelityError("Unavailable", 404)); const log = vi.spyOn(console, "error").mockImplementation(() => {});
    await render(); expect(screen.queryByRole("combobox")).toBeNull(); expect(log).not.toHaveBeenCalled();
  });
  it("hides and logs other GET failures", async () => {
    vi.spyOn(fidelityApi, "get").mockRejectedValue(new FidelityError("Offline", 500)); const log = vi.spyOn(console, "error").mockImplementation(() => {});
    await render(); expect(screen.queryByRole("combobox")).toBeNull(); expect(log).toHaveBeenCalledOnce();
  });
  it("loads again when the dataset changes", async () => {
    const get = vi.spyOn(fidelityApi, "get").mockResolvedValue(fixture()); await render(); await render({ datasetId: "d2" });
    expect(get).toHaveBeenLastCalledWith("d2", "s1", expect.any(AbortSignal));
  });
  it("updates optimistically and replaces state on success", async () => {
    vi.spyOn(fidelityApi, "get").mockResolvedValue(fixture()); let complete: ((v: FidelityState) => void) | undefined;
    const put = vi.spyOn(fidelityApi, "put").mockImplementation(() => new Promise((resolve) => { complete = resolve; })); await render(); await click("Full dataset All rows");
    expect((screen.getByRole("combobox", { name: "Analysis on" }) as HTMLSelectElement).value).toBe("entire_dataset");
    expect(put).toHaveBeenCalledWith("d1", "s1", { analysis_fidelity: "entire_dataset" }, expect.any(AbortSignal));
    await act(async () => complete?.(fixture({ analysis_fidelity: "entire_dataset" })));
    expect(mocks.success).toHaveBeenCalledWith("Full dataset selected");
  });
  it.each([409, 422])("rolls back on %s and displays backend detail", async (status) => {
    vi.spyOn(fidelityApi, "get").mockResolvedValue(fixture()); vi.spyOn(fidelityApi, "put").mockRejectedValue(new FidelityError("Build first", status)); await render(); await click("Full dataset All rows");
    expect((screen.getByRole("combobox", { name: "Analysis on" }) as HTMLSelectElement).value).toBe("quick_sample"); expect(mocks.error).toHaveBeenCalledWith("Build first");
  });
  it("refreshes preview once after a successful selected sample change", async () => {
    vi.spyOn(fidelityApi, "get").mockResolvedValue(fixture({ analysis_fidelity: "portfolio_samples", selected_sample_name: "balanced" }));
    vi.spyOn(fidelityApi, "put").mockResolvedValue(fixture({ analysis_fidelity: "portfolio_samples", selected_sample_name: "tail" })); await render(); await click("tail"); expect(sample).toHaveBeenCalledWith("d1");
  });
  it("refetches after a chat reply changes the mode or clears the sample", async () => {
    const get = vi.spyOn(fidelityApi, "get").mockResolvedValue(fixture({ selected_sample_name: "tail" })); await render(); await render({ chatResponse: { selected_sample_name: null } }); expect(get).toHaveBeenCalledTimes(2);
  });
  it.each(["SUCCESS", "FAILURE", "NONE"] as const)("polls every 10s, refetches on %s and stops", async (state) => {
    vi.useFakeTimers(); const building = fixture({ options: fixture().options.map((o) => o.value === "portfolio_samples" ? { ...o, status: "building" } : o) });
    const get = vi.spyOn(fidelityApi, "get").mockResolvedValueOnce(building).mockResolvedValue(fixture()); const status = vi.spyOn(fidelityApi, "status").mockResolvedValue({ state }); await render();
    await act(async () => vi.advanceTimersByTimeAsync(9999)); expect(status).not.toHaveBeenCalled(); await act(async () => vi.advanceTimersByTimeAsync(1)); expect(status).toHaveBeenCalledOnce(); expect(get).toHaveBeenCalledTimes(2);
    await act(async () => vi.advanceTimersByTimeAsync(30000)); expect(status).toHaveBeenCalledOnce(); if (state === "FAILURE") expect(mocks.error).toHaveBeenCalledWith("Portfolio sample build failed");
  });
  it("polls background IDs returned by PUT and cancels on unmount", async () => {
    vi.useFakeTimers(); vi.spyOn(fidelityApi, "get").mockResolvedValue(fixture()); vi.spyOn(fidelityApi, "put").mockResolvedValue(fixture({ analysis_fidelity: "entire_dataset", background_task_id: "b1" })); const status = vi.spyOn(fidelityApi, "status").mockResolvedValue({ state: "RUNNING" }); await render(); await click("Full dataset All rows");
    await act(async () => vi.advanceTimersByTimeAsync(10000)); expect(status).toHaveBeenCalledOnce(); act(() => root.render(null)); await act(async () => vi.advanceTimersByTimeAsync(30000)); expect(status).toHaveBeenCalledOnce();
  });
});
describe("shared rules", () => {
  it("detects null sample changes, avoids redundant refresh, and recognizes full dataset jobs", () => {
    expect(fidelityReplyDiffers(fixture(), { analysis_fidelity: "quick_sample", selected_sample_name: null })).toBe(false);
    expect(fidelityReplyDiffers(fixture({ selected_sample_name: "tail" }), { selected_sample_name: null })).toBe(true);
    expect(FIDELITY_POLL_MS).toBe(10000); expect(needsFidelityPolling(fixture())).toBe(false); expect(needsFidelityPolling(fixture({ background_task_id: "b1" }))).toBe(true);
    expect(isFullDatasetTask({ analysis_task_id: "a1", analysis_fidelity: "entire_dataset" }, "quick_sample")).toBe(true);
    expect(isFullDatasetTask({ analysis_task_id: "a1" }, "entire_dataset")).toBe(true);
    expect(isFullDatasetTask({ analysis_task_id: "a1" }, "quick_sample")).toBe(false);
  });
});