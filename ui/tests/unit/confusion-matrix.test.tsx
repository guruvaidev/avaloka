// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { ConfusionMatrix } from "@/components/inference/ConfusionMatrix";
import type { CrossValidation } from "@/api/inference";

// TI-36 / TI-41: the UI shows the API's numbers -- counts and row-% -- and the labels in order.
const ORACLE: CrossValidation = {
  run_id: "r", status: "succeeded", k: 4, splitter: "StratifiedKFold", reason: null,
  aggregate: { labels: ["A", "B", "C"], matrix: [[3, 1, 0], [0, 3, 1], [1, 0, 3]], normalized: [[0.75, 0.25, 0], [0, 0.75, 0.25], [0.25, 0, 0.75]], n: 12, k: 4, accuracy_mean: 0.75, accuracy_std: 0 },
  folds: [0, 1, 2, 3].map((fold) => ({ fold, labels: ["A", "B", "C"], matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], normalized: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], n: 3, accuracy: 1, f1_macro: 1 })),
};

function render(cv: CrossValidation) {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => { root.render(createElement(ConfusionMatrix, { cv })); });
  return host;
}

describe("ConfusionMatrix", () => {
  it("renders every count and row-% of the summed matrix, labels on both axes in order", () => {
    const host = render(ORACLE);
    const cell = (i: number, j: number) => host.querySelector(`[data-testid="cv-cell-${i}-${j}"]`)?.textContent ?? "";
    expect(cell(0, 0)).toBe("375.0%"); expect(cell(0, 1)).toBe("125.0%"); expect(cell(0, 2)).toBe("00.0%");
    expect(cell(1, 1)).toBe("375.0%"); expect(cell(1, 2)).toBe("125.0%"); expect(cell(2, 0)).toBe("125.0%"); expect(cell(2, 2)).toBe("375.0%");
    const headers = Array.from(host.querySelectorAll("thead th")).slice(1).map((th) => th.textContent);
    const rows = Array.from(host.querySelectorAll("tbody th")).map((th) => th.textContent);
    expect(headers).toEqual(["A", "B", "C"]); expect(rows).toEqual(["A", "B", "C"]);
    expect(host.textContent).toContain("Total 12 rows across 4 folds");
    expect(host.textContent).toContain("accuracy 75.0%");
  });
  it("shows the fold-level matrix when a fold is selected", () => {
    const host = render(ORACLE);
    const select = host.querySelector("select") as HTMLSelectElement;
    act(() => { select.value = "1"; select.dispatchEvent(new Event("change", { bubbles: true })); });
    expect(host.querySelector('[data-testid="cv-cell-0-0"]')?.textContent).toBe("1100.0%");
    expect(host.textContent).toContain("Total 3 rows");
  });
  it("regression: no heatmap, a fold table and the sentence", () => {
    const host = render({ run_id: "r", status: "succeeded", k: 3, splitter: "KFold", reason: null, aggregate: { k: 3, rmse_mean: 1.2 }, folds: [{ fold: 0, rmse: 1.2, mae: 0.9, r2: 0.5 }] });
    expect(host.querySelector('[data-testid="cv-matrix"]')).toBeNull();
    expect(host.textContent).toContain("Confusion matrices apply to classification models");
    expect(host.textContent).toContain("1.2");
  });
  it("skipped: the reason is shown", () => {
    const host = render({ run_id: "r", status: "skipped", k: null, splitter: null, reason: "only 20 rows; at least 30 are needed", aggregate: null, folds: [] });
    expect(host.querySelector('[data-testid="cv-note"]')?.textContent).toContain("only 20 rows");
  });
});
