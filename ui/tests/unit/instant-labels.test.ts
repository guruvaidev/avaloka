// The report page's closed label maps: every gate the fixtures' recipes wrote
// has a label, every recipe id has a title, and nothing the Evidence panel can
// print carries a digit.
import { describe, expect, it } from "vitest";
import {
  COMPLETION_WORDS,
  GATE_LABELS,
  RECIPE_IDS,
  RECIPE_TITLES,
  STATUS_WORDS,
  VERDICT_LABELS,
  gateLabel,
  recipeTitle,
} from "@/lib/instant-labels";
import describeFx from "../fixtures/instant/describe_clean_by_dimension.report.json";
import trendFx from "../fixtures/instant/trend_gaps.report.json";
import correlateFx from "../fixtures/instant/correlate_simpsons_paradox.report.json";
import abTestFx from "../fixtures/instant/ab_test_underpowered.report.json";

const FIXTURES = [describeFx, trendFx, correlateFx, abTestFx];

describe("instant-labels", () => {
  it("labels every gate the fixtures' data_checks name", () => {
    const gates = new Set(FIXTURES.flatMap((f) => f.data_checks.map((c) => c.gate)));
    expect(gates.size).toBeGreaterThanOrEqual(10);
    for (const g of gates) {
      expect(GATE_LABELS[g], `gate ${g} has no label`).toBeTruthy();
      expect(gateLabel(g)).toBe(GATE_LABELS[g]);
    }
  });

  it("titles the ten recipes, and the fixtures' recipes are among them", () => {
    expect(RECIPE_IDS.length).toBe(10);
    for (const id of RECIPE_IDS) expect(RECIPE_TITLES[id]).toBeTruthy();
    for (const f of FIXTURES) {
      expect(RECIPE_IDS).toContain(f.provenance.recipe.id);
      expect(recipeTitle(f.provenance.recipe.id)).toBe(
        RECIPE_TITLES[f.provenance.recipe.id as (typeof RECIPE_IDS)[number]],
      );
    }
  });

  it("keeps every word the Evidence panel can print digit-free", () => {
    const words = [
      ...Object.values(GATE_LABELS),
      ...Object.values(STATUS_WORDS),
      ...Object.values(VERDICT_LABELS),
      ...Object.values(COMPLETION_WORDS),
      ...Object.values(RECIPE_TITLES),
    ];
    for (const w of words) expect(w, `"${w}" carries a digit`).not.toMatch(/\d/);
  });

  it("never says verified, and spells out an id the map lacks", () => {
    for (const w of Object.values(VERDICT_LABELS)) expect(w).not.toMatch(/verif/i);
    expect(VERDICT_LABELS.pass).toBe("Answered");
    expect(VERDICT_LABELS.qualified).toBe("Answered with caveats");
    expect(gateLabel("some_new_gate")).toBe("Some new gate");
    expect(recipeTitle("forecast")).toBe("Forecast");
  });
});
