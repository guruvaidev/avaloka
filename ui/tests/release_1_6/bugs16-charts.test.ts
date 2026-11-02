// 1.6 release suite: Bugs-1.6 chart-axis rows (BG28, BG36).
// See docs/test-reports/1.6-traceability.csv.
import { describe, expect, it } from "vitest";
import { logarithmicAxis, niceAxis, scatterScale } from "../../src/lib/chart-display";

describe("Bugs-1.6 chart axes", () => {
  // BG28 is an open defect: niceAxis rounds the top of the value axis up to a
  // whole step, so a round maximum (100 -> step 20 -> [0, 100]) gets no space
  // above the tallest bar. dynamicChart.tsx draws the bar/line value axis with
  // niceAxis(yValues, true). it.fails: remove when the axis gains headroom.
  it.fails("BG28 [open defect, strict]: value axis leaves headroom above a round maximum", () => {
    const counts = [12, 40, 73, 100];
    const axis = niceAxis(counts, true);
    expect(axis.domain[0]).toBe(0);
    // The top of the axis must sit strictly above the tallest bar.
    expect(axis.domain[1]).toBeGreaterThan(100);
    expect(axis.ticks[axis.ticks.length - 1]).toBeGreaterThan(100);
  });

  it("BG36: a skewed positive scatter switches to a log axis with decade ticks", () => {
    // Loan amounts spanning four orders of magnitude, mostly small.
    const loanAmounts = [1_000, 2_500, 4_000, 5_000, 7_500, 12_000, 40_000, 8_000_000];
    expect(scatterScale("scatter", loanAmounts)).toBe(true);

    const axis = logarithmicAxis(loanAmounts);
    expect(axis).not.toBeNull();
    expect(axis!.domain).toEqual([1_000, 10_000_000]);
    for (const tick of axis!.ticks) expect(Math.log10(tick) % 1).toBe(0);
  });

  it("BG36: an unskewed scatter, a non-scatter chart and an explicit linear choice stay linear", () => {
    expect(scatterScale("scatter", [10, 20, 35, 50, 90])).toBe(false);
    expect(scatterScale("bar", [1_000, 8_000_000])).toBe(false);
    expect(scatterScale("scatter", [1_000, 8_000_000], "linear")).toBe(false);
    // Zero or negative values cannot be drawn on a log axis.
    expect(scatterScale("scatter", [0, 1_000, 8_000_000])).toBe(false);
  });
});
