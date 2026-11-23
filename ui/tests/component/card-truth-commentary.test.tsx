// rc18 (c-strict): model-written values the UI showed plain. The fixture is
// what the backend emits for stored rweb17 fields
// (tests/fixtures/commentary/rweb17-card-truth.json): the overview badges, the
// column meanings and predictive values, the trace's model-written reasons.
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AutoInsightsModal, DatasetOverview } from "@/components/dashboard/AutoInsightsModal";
import { StepDetail } from "@/components/agent-trace/TraceSteps";
import { COMMENTARY_SUFFIX, COMMENTARY_TOOLTIP } from "@/lib/reply-segments";
import type { TraceStep, TraceTurn } from "@/lib/agent-trace/types";
import fixture from "../fixtures/commentary/rweb17-card-truth.json";

// AutoInsightsModal reads the plan for its auto_insights gate (every plan has it).
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({ data: { plan: "free" }, isLoading: false }) }));
vi.mock("@/integrations/supabase/external-client", () => ({
  supabase: {
    auth: {
      getSession: vi.fn(async () => ({ data: { session: { access_token: "t" } } })),
      onAuthStateChange: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })),
    },
  },
}));

const marked = (root: ParentNode) => [...root.querySelectorAll("[data-commentary]")];
const markedText = (root: ParentNode) => marked(root).map((e) => e.textContent?.trim());

describe("Dataset overview badges", () => {
  it("marks the profiling model's readiness score, with the commentary tooltip", () => {
    const { container } = render(<DatasetOverview config={{ profile_overview: fixture.overview_vocabulary }} />);
    const score = `${fixture.overview_vocabulary.readiness_score}/100`;
    expect(markedText(container)).toContain(score);
    const el = marked(container).find((e) => e.textContent?.trim() === score)!;
    expect(el.getAttribute("title")).toBe(COMMENTARY_TOOLTIP);
    expect(container.textContent).not.toContain(COMMENTARY_SUFFIX.trim());
  });

  it("shows a domain the vocabulary decided plain, and one a model chose as commentary", () => {
    const vocab = render(<DatasetOverview config={{ profile_overview: fixture.overview_vocabulary }} />).container;
    expect(markedText(vocab)).not.toContain("retail");
    expect(vocab.textContent).toContain("retail");
    const model = render(<DatasetOverview config={{ profile_overview: fixture.overview_model_domain }} />).container;
    expect(markedText(model)).toContain("retail");
    // An older payload records no provenance: the domain is the model's.
    const older = render(<DatasetOverview config={{ profile_overview: { domain: "Healthcare" } }} />).container;
    expect(markedText(older)).toContain("Healthcare");
  });
});

describe("Column statistics", () => {
  it("marks each model-written meaning and predictive value", () => {
    // Through the exported modal (the column table sits under each chart card).
    const chart = {
      id: "c1", type: "bar", title: "Counts by Country", insight: "Records by Country.", insight_source: "data",
      encodings: { x: { field: "Country", type: "nominal" }, y: { aggregate: "count", type: "quantitative" } },
      derived_data: { points: [{ x: "UK", y: 3, Country: "UK", count: 3 }, { x: "EIRE", y: 1, Country: "EIRE", count: 1 }], x_key: "Country", y_key: "count" },
    };
    const config = { charts: [chart], column_stats: fixture.column_stats, visualization_status: "ready" };
    const { baseElement } = render(<AutoInsightsModal open onOpenChange={() => {}} config={config} status="ready" defaultView="chart" />);
    const container = baseElement;
    // The column table is the modal's Column statistics tab (1.6); the stack view
    // shows only the active chart's columns.
    fireEvent.click(screen.getByRole("button", { name: "Column statistics" }));
    const rows = fixture.column_stats.filter((c: any) => c.meaning);
    expect(rows.length).toBeGreaterThan(0);
    for (const c of rows as any[]) {
      expect(markedText(container)).toContain(c.meaning_segments[0].text);
    }
    expect(container.textContent).not.toContain(COMMENTARY_SUFFIX.trim());
    const values = [...container.querySelectorAll("[data-predictive-value]")];
    expect(values.length).toBe(fixture.column_stats.filter((c: any) => c.predictive_value).length);
    for (const v of values) {
      expect(v.querySelector("[data-commentary]")?.getAttribute("title")).toBe(COMMENTARY_TOOLTIP);
      expect(v.querySelector("[data-commentary-label]")).not.toBeNull();
    }
  });
});

describe("Agent trace", () => {
  it("marks a reason a model wrote and leaves code's plain", () => {
    const steps = fixture.trace_steps as unknown as TraceStep[];
    const turn = { turn_id: "t1", steps: [], handoffs: [] } as unknown as TraceTurn;
    const step = (s: object, node: string) => ({ ...s, node, kind: "agent", step_id: node, turn_id: "t1" }) as TraceStep;
    const intent = render(<StepDetail step={step(steps[0], "avaloka_agent")} turn={turn} />).container;
    expect(markedText(intent)).toContain(steps[0].decision!.reason);
    const review = render(<StepDetail step={step(steps[1], "validator_logical")} turn={turn} />).container;
    expect(marked(review).length).toBeGreaterThan(0);
    const code = { ...steps[2], decision: { type: "check", value: "fail", reason: "KeyError" } };
    const plain = render(<StepDetail step={step(code, "execute_code")} turn={turn} />).container;
    expect(plain.textContent).toContain("KeyError");
    expect(marked(plain)).toHaveLength(0);
  });
});

describe("Verification step", () => {
  it("states the count, not a verdict on every claim", () => {
    const turn = { turn_id: "t1", steps: [], handoffs: [] } as unknown as TraceTurn;
    const step = {
      kind: "agent", step_id: "v", turn_id: "t1", node: "verify_claims", agent: "Claim verifier",
      decision: { type: "evidence", value: "safe", reason: "Checked 4 claim(s) against the numbers" },
      calculation: { values: { claims_checked: 4, problems: 0 } },
    } as unknown as TraceStep;
    const { container } = render(<StepDetail step={step} turn={turn} />);
    const text = container.textContent ?? "";
    expect(text).toContain("evidence: checked");
    expect(text).not.toMatch(/evidence: safe|all .* supported/i);
  });
});

describe("Chart explainer answer", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("keeps the model's answer segments, and drops them with the answer", async () => {
    const { explainInsights } = await import("@/lib/insightsExplainApi");
    const segments = [{ text: "Germany returns most.", kind: "commentary" }];
    const answer = `Germany returns most.${COMMENTARY_SUFFIX}`;
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ answer, answer_segments: segments, focus_chart_ids: [] }))));
    const res = await explainInsights("aid", { question: "q", charts: [] } as never, new AbortController().signal);
    expect(res.answer_segments).toEqual(segments);
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ answer: "", answer_segments: segments }))));
    const empty = await explainInsights("aid", { question: "q", charts: [] } as never, new AbortController().signal);
    expect(empty.answer_segments).toBeNull();
  });
});
