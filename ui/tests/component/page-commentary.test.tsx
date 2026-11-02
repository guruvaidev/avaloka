import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PageCommentaryCard } from "../../src/components/analysis/PageCommentaryCard";
import { PAGE_COMMENTARY_LABEL, pageCommentary } from "../../src/lib/page-commentary";

// A profile as the visualization agent ships it: computed figures plus the
// profiling model's opinions. The model fields carry distinctive numbers and
// words so any leak into the commentary is caught.
const PROFILE = {
  dataset: { filename: "sales.csv", total_rows: 12345, rows_sampled: 5000, column_count: 4 },
  column_stats: [
    {
      name: "order_date",
      dtype: "datetime64[ns]",
      missing_pct: 0,
      predictive_value: "high",
      meaning: "MODEL MEANING",
    },
    { name: "revenue", dtype: "float64", missing_pct: 2.5 },
    { name: "region", dtype: "object", missing_pct: 41.7 },
    { name: "discount", dtype: "float64", missing_pct: 23 },
  ],
  charts: [{ id: "c1" }, { id: "c2" }, { id: "c3" }],
  profile_overview: {
    domain: "Retail MODELDOMAIN",
    readiness_score: 87,
    readiness_reasoning: "MODEL REASONING 99%",
    red_flags: ["MODEL FLAG 77% duplicates"],
    analysis_suggestions: ["MODEL SUGGESTION forecast churn"],
    quick_wins: ["MODEL QUICKWIN"],
  },
};

describe("page commentary", () => {
  it("is grounded: every figure comes from the computed profile", () => {
    const text = pageCommentary(PROFILE, { datasetCount: 2 })
      .map((l) => l.text)
      .join(" ");
    expect(text).toContain("12,345 rows");
    expect(text).toContain("4 columns");
    expect(text).toContain("2 numeric columns, 1 category column, 1 date column");
    expect(text).toContain(
      "2 columns are more than 20% empty: region (41.7% empty), discount (23% empty)",
    );
    expect(text).toContain("3 charts");
    expect(text).toContain(
      "clean the emptiest columns; join the attached datasets; look at trends over time",
    );
  });

  it("never repeats model-written profile fields", () => {
    const text = pageCommentary(PROFILE, { datasetCount: 2 })
      .map((l) => l.text)
      .join(" ");
    for (const leak of ["MODEL", "87", "99", "77", "Retail", "high", "forecast churn"]) {
      expect(text).not.toContain(leak);
    }
  });

  it("says nothing when the profile has no computed figures", () => {
    expect(pageCommentary({ profile_overview: PROFILE.profile_overview })).toEqual([]);
    expect(pageCommentary(null)).toEqual([]);
  });

  it("marks a sampled row count as sampled", () => {
    const text = pageCommentary({ dataset: { rows_sampled: 500, column_count: 2 } })[0].text;
    expect(text).toBe("This dataset has 500 rows (sampled) and 2 columns.");
  });

  it("the card is labelled as computed from the profile", () => {
    render(<PageCommentaryCard lines={pageCommentary(PROFILE)} datasetName="sales.csv" />);
    const card = screen.getByTestId("page-commentary");
    expect(card.getAttribute("data-source")).toBe("profile");
    expect(screen.getByTestId("page-commentary-label").textContent).toBe(PAGE_COMMENTARY_LABEL);
    expect(PAGE_COMMENTARY_LABEL).toBe("Computed from your data profile");
    expect(card.textContent).toContain("On this page: sales.csv");
  });

  it("the card renders nothing without lines", () => {
    const { container } = render(<PageCommentaryCard lines={[]} />);
    expect(container.innerHTML).toBe("");
  });
});
