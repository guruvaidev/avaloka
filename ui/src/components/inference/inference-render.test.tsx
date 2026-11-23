// @vitest-environment happy-dom
import { describe, expect, it, vi } from "vitest";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
vi.mock("@tanstack/react-router", () => ({ Link: ({ children, ...p }: { children: unknown }) => createElement("a", p as object, children as never) }));
vi.mock("@/lib/api/backendApi", () => ({ proxyFetch: vi.fn(), getAccessToken: vi.fn() }));
import { normalizeModels, normalizeResult } from "@/api/inference";
import { ModelsList } from "./ModelsList";
import { ModelSummary } from "./ModelSummary";
import { InferenceResults } from "./InferenceResults";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
function render(node: React.ReactElement) {
  const el = document.createElement("div");
  document.body.appendChild(el);
  // ModelSummary reads the run's cross-validation through react-query (2.x); the sections need a client.
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  act(() => { createRoot(el).render(createElement(QueryClientProvider, { client }, node)); });
  return el;
}
const raw = [
  ...Array.from({ length: 100 }, (_, i) => ({ model_name: "median_house_value_model", model_type: "regression", model_version: String(i), mlflow_run_id: `run${i}xxxxxxxx`, created_at: "2026-11-01T00:00:00Z", final_mae: "37043.67", final_rmse: 55006.76, final_r2_score: 0.77, evaluation_report: { baseline: { mae: 90000 } } })),
  { run_id: "onlyid" },
  { run_id: "bad", model_name: "x", created_at: "garbage", final_mae: "abc" },
];
const noop = () => {};

describe("Inference sections render without crashing", () => {
  it("renders the real-sized list with 50 visible, incomplete runs hidden", () => {
    const el = render(createElement(ModelsList, { models: normalizeModels({ models: raw }), loading: false, error: false, selectedRunId: "run1xxxxxxxx", onRetry: noop, onSelect: noop }));
    expect(el.querySelectorAll("button.inf-model")).toHaveLength(50);
    expect(el.textContent).toContain("Show incomplete runs (1)");
    expect(el.textContent).toContain("Load more");
  });
  it("renders the empty state", () => {
    const el = render(createElement(ModelsList, { models: [], loading: false, error: false, selectedRunId: "", onRetry: noop, onSelect: noop }));
    expect(el.textContent).toContain("No trained models yet.");
  });
  it("renders summaries for full, id-only and invalid-date runs", () => {
    for (const m of normalizeModels({ models: raw }).slice(-3)) {
      const el = render(createElement(ModelSummary, { model: m, starting: false, busy: false, onStart: noop, onStop: noop }));
      expect(el.textContent).toContain("Start service");
    }
    const el = render(createElement(ModelSummary, { model: normalizeModels([raw[0]])[0], starting: false, busy: false, onStart: noop, onStop: noop }));
    expect(el.textContent).toContain("37,043.67");
    expect(el.textContent).toContain("Beats baseline (mean prediction) by 58.8%");
  });
  it("renders results for a number, an array and objects", () => {
    const model = normalizeModels([raw[0]])[0];
    for (const [result, n] of [[206855.5, 1], [[1, 2], 2], [[{ prediction: "a", probabilities: { a: 0.9 } }], 1]] as const) {
      const predictions = normalizeResult(result, n, model.target);
      const el = render(createElement(InferenceResults, { run: { id: "x", time: 0, duration: 10, model, rows: Array.from({ length: n }, () => ({})), predictions, warnings: [] }, selected: 0, onSelect: noop }));
      expect(el.textContent).toContain("Row 1");
    }
  });
});
