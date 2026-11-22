// The Billing comparison's label map, cells and plan cards, against the matrix
// generated from app/core/editions.py by scripts/gen_edition_matrix.py.
import { describe, expect, it } from "vitest";
import {
  COMPARISON_LABELS,
  comparisonSections,
  normalizePlan,
  planCardFeatures,
  planCell,
  planIncludes,
  UPGRADE_PROMPT_PHRASES,
  upgradePromptText,
} from "@/lib/edition-comparison";
import {
  EDITION_CAPABILITIES,
  EDITION_CAPABILITY_STATUS,
  EDITION_MATRIX,
  EDITION_PLANS,
} from "@/lib/edition-matrix.gen";

const allGranted = Object.fromEntries(EDITION_PLANS.map((p) => [p, { x: true }])) as never;

describe("Billing comparison labels", () => {
  it("every capability in the matrix has an entry in the label map", () => {
    const unlabelled = EDITION_CAPABILITIES.filter((c) => !COMPARISON_LABELS[c]);
    expect(unlabelled).toEqual([]);
  });

  it("the label map holds no capability the matrix has dropped", () => {
    expect(Object.keys(COMPARISON_LABELS).sort()).toEqual([...EDITION_CAPABILITIES].sort());
  });

  it("an unlabelled capability throws, naming it, rather than disappearing", () => {
    expect(() => comparisonSections(["brand_new"], {}, allGranted)).toThrow(
      "No Billing comparison label for: brand_new",
    );
  });

  it("no two rows share a label", () => {
    const labels = comparisonSections().flatMap((s) => s.rows.map((r) => r.label));
    expect(new Set(labels).size).toBe(labels.length);
  });

  it("only swarm_intelligence is held back from the table, and it says why", () => {
    const shown = comparisonSections().flatMap((s) => s.rows.map((r) => r.capability));
    const hidden = EDITION_CAPABILITIES.filter((c) => !shown.includes(c));
    expect(hidden).toEqual(["swarm_intelligence"]);
    expect(COMPARISON_LABELS.swarm_intelligence).toHaveProperty("hiddenBecause");
  });

  it("every cell is the matrix's own value, with its status", () => {
    for (const row of comparisonSections().flatMap((s) => s.rows)) {
      for (const plan of EDITION_PLANS) {
        const granted = EDITION_MATRIX[plan][row.capability];
        const expected = !granted ? "no" : EDITION_CAPABILITY_STATUS[row.capability] === "shipped" ? "yes" : "coming";
        expect(row.cells[plan], `${row.capability} on ${plan}`).toBe(expected);
      }
    }
  });
});

describe("a coming capability is never included", () => {
  it("planCell gives 'coming', never 'yes', for a granted capability that has not shipped", () => {
    for (const plan of EDITION_PLANS) {
      expect(planCell(plan, "x", allGranted, { x: "coming" })).toBe("coming");
      expect(planCell(plan, "x", allGranted, { x: "shipped" })).toBe("yes");
    }
  });

  it("no real coming capability is a 'yes' cell or passes a plan gate", () => {
    const coming = EDITION_CAPABILITIES.filter((c) => EDITION_CAPABILITY_STATUS[c] === "coming");
    expect(coming).toContain("sso_audit");
    for (const capability of coming) {
      for (const plan of EDITION_PLANS) {
        expect(planCell(plan, capability), `${capability} on ${plan}`).not.toBe("yes");
        expect(planIncludes(plan, capability), `${capability} on ${plan}`).toBe(false);
      }
    }
  });
});

describe("plan cards", () => {
  it("every card line is a capability that plan has and the plan before it does not", () => {
    const labelOf = new Map(
      EDITION_CAPABILITIES.flatMap((c) => {
        const e = COMPARISON_LABELS[c];
        return "label" in e ? [[e.label, c] as const] : [];
      }),
    );
    EDITION_PLANS.forEach((plan, i) => {
      for (const line of planCardFeatures(plan)) {
        const capability = labelOf.get(line.replace(/ \(coming soon\)$/, ""));
        expect(capability, `${plan} card line "${line}" is a capability label`).toBeTruthy();
        expect(planCell(plan, capability!), line).not.toBe("no");
        if (i > 0) expect(planCell(EDITION_PLANS[i - 1], capability!), line).toBe("no");
        expect(line.endsWith("(coming soon)"), line).toBe(planCell(plan, capability!) === "coming");
      }
    });
  });

  it("Auto Insights is on the Free card, so not again on Professional's", () => {
    expect(planCardFeatures("free")).toContain("Auto Insights");
    expect(planCardFeatures("professional")).not.toContain("Auto Insights");
  });
});

describe("normalizePlan", () => {
  it("keeps known plans, treats unknown names as Free, and an unanswered lookup as unknown", () => {
    expect(normalizePlan("professional")).toBe("professional");
    expect(normalizePlan("pro")).toBe("free");
    expect(normalizePlan(undefined)).toBeNull();
  });
});

describe("the Free upgrade prompt", () => {
  it("names only capabilities Free lacks and Professional includes", () => {
    for (const capability of Object.keys(UPGRADE_PROMPT_PHRASES) as (keyof typeof UPGRADE_PROMPT_PHRASES)[]) {
      expect(planCell("free", capability), capability).toBe("no");
      expect(planCell("professional", capability), capability).toBe("yes");
    }
  });

  it("is built from those phrases, the owner's collaboration items first", () => {
    expect(upgradePromptText()).toBe(
      "Upgrade to invite people and form teams, comment on analyses, share analyses, schedule analyses and use Reports.",
    );
  });
});
