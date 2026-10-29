import { describe, it, expect } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { DatasetOverview } from "../../src/components/dashboard/AutoInsightsModal";

const config = {
  dataset: { total_rows: 1200, column_count: 8 },
  column_stats: [
    { name: "rate", dtype: "float64", min: 1, max: 9, median: 4, mean: 5 },
    { name: "month", dtype: "object", top_values: [{ value: "Jan", pct: 30 }] },
  ],
  profile_overview: {
    domain: "Mortgage servicing",
    domain_reasoning: "Each record is one monthly loan performance observation.",
    readiness_score: 72,
    readiness_reasoning: "Two columns need review before modelling.",
    red_flags: ["Missing values in rate"],
    analysis_suggestions: ["Which months had the highest default rate?"],
    quick_wins: ["Chart rate against month"],
  },
};

describe("Auto Insights Overview summary order", () => {
  it("renders the dataset overview card before the action cards", () => {
    const html = renderToStaticMarkup(createElement(DatasetOverview, { config }));
    const order = [
      html.indexOf("Mortgage servicing"),
      html.indexOf('aria-label="Data readiness"'),
      html.indexOf("<span>Rows</span>"),
      html.indexOf("<span>Columns</span>"),
      html.indexOf("Watch out for"),
      html.indexOf("Suggested next steps"),
    ];

    order.forEach((at, i) => expect(at, `marker ${i}`).toBeGreaterThan(-1));
    order.forEach((at, i) => {
      if (i > 0) expect(at, `marker ${i} follows marker ${i - 1}`).toBeGreaterThan(order[i - 1]);
    });
  });

  it("renders nothing when the dataset has no profile overview", () => {
    expect(renderToStaticMarkup(createElement(DatasetOverview, { config: {} }))).toBe("");
  });
});