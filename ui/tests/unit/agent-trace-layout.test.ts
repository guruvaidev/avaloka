// The planner graph's layout is a pure function of the backend's graph. These
// tests run it on a graph the real backend produced (tests/fixtures, written by
// scripts/gen_agent_trace_fixture.py) and check the picture is readable.
import { describe, expect, it } from "vitest";
import fixture from "../fixtures/agent-trace.json";
import { dataSpine, fmtCost, layoutGraph, nodeTone, LANES } from "@/lib/agent-trace/layout";
import type { PlanGraph } from "@/lib/agent-trace/types";

const graph = fixture.graph as unknown as PlanGraph;
const liveGraph = fixture.graph_live as unknown as PlanGraph;

describe("layoutGraph", () => {
  const layout = layoutGraph(graph);
  const byId = new Map(layout.nodes.map((n) => [n.id, n]));

  it("places every node and every edge", () => {
    expect(layout.nodes).toHaveLength(graph.nodes.length);
    expect(layout.edges).toHaveLength(graph.edges.length);
    for (const n of layout.nodes) {
      expect(n.x + n.w).toBeLessThanOrEqual(layout.width);
      expect(n.y + n.h).toBeLessThanOrEqual(layout.height);
    }
  });

  it("reads top to bottom: every edge points down", () => {
    for (const e of layout.edges) {
      const s = byId.get(e.source)!;
      const t = byId.get(e.target)!;
      expect(t.y, `${s.label} → ${t.label}`).toBeGreaterThan(s.y);
    }
  });

  it("is taller than wide: a long turn scrolls down, not across", () => {
    expect(layout.height).toBeGreaterThan(layout.width);
  });

  it("draws a retry from the failing check to the next attempt, bowing out to the side", () => {
    const retry = layout.edges.find((e) => e.type === "retry");
    expect(retry).toBeDefined();
    const s = byId.get(retry!.source)!;
    const t = byId.get(retry!.target)!;
    expect(s.type).toBe("check");
    expect(s.value).toBe("fail");
    expect(t.label).toBe("Coder · attempt 2");
    expect(retry!.labelX).toBeGreaterThan(Math.max(s.x + s.w, t.x + t.w));
  });

  it("puts lanes side by side, agents then code then data, without overlap", () => {
    expect(layout.lanes.map((l) => l.lane)).toEqual([...LANES]);
    for (let i = 1; i < layout.lanes.length; i++) {
      const prev = layout.lanes[i - 1];
      expect(layout.lanes[i].x).toBeGreaterThan(prev.x + prev.w);
    }
    for (const n of layout.nodes) {
      const lane = layout.lanes.find((l) => l.lane === n.lane)!;
      expect(n.x).toBeGreaterThanOrEqual(lane.x);
      expect(n.x + n.w).toBeLessThanOrEqual(lane.x + lane.w + 0.5);
    }
    for (const a of layout.nodes) {
      for (const b of layout.nodes) {
        if (a.id >= b.id) continue;
        const overlap = a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
        expect(overlap, `${a.label} overlaps ${b.label}`).toBe(false);
      }
    }
  });
});

describe("dataSpine", () => {
  it("keeps the question and the data lane, with only the edges between them", () => {
    const spine = dataSpine(liveGraph);
    expect(spine.nodes.map((n) => n.type)).toEqual(["question", "data", "data", "operation", "operation", "result"]);
    const ids = new Set(spine.nodes.map((n) => n.id));
    for (const e of spine.edges) expect(ids.has(e.source) && ids.has(e.target)).toBe(true);
    expect(spine.edges.some((e) => e.source === "q" && e.target === "data:load")).toBe(true);
  });

  it("closes up the rows the hidden agents left empty", () => {
    const layout = layoutGraph(dataSpine(liveGraph));
    const layers = [...new Set(layout.nodes.map((n) => n.layer))].sort((a, b) => a - b);
    expect(layers).toEqual([0, 1, 2, 3, 4, 5]);
    expect(layout.height).toBeLessThan(layoutGraph(liveGraph).height / 3);
    for (const e of layout.edges) {
      expect(layout.nodes.find((n) => n.id === e.target)!.y).toBeGreaterThan(layout.nodes.find((n) => n.id === e.source)!.y);
    }
  });
});

describe("appearance", () => {
  it("gives every node type its own colour", () => {
    const types = ["question", "agent", "decision", "gate", "data", "operation", "check", "evidence", "train", "evaluate"] as const;
    const fills = new Set(types.map((type) => nodeTone({ type }).fill + nodeTone({ type }).stroke));
    expect(fills.size).toBe(types.length);
  });

  it("marks a failed check red whatever its type", () => {
    expect(nodeTone({ type: "check", value: "fail" }).label).toBe("failed");
    expect(nodeTone({ type: "agent", status: "error" }).label).toBe("failed");
    expect(nodeTone({ type: "check", value: "pass" }).label).toBe("check");
  });

  it("never shows an unknown cost as zero", () => {
    expect(fmtCost(null, 2)).toBe("unpriced");
    expect(fmtCost(null)).toBe("—");
    expect(fmtCost(0.0042)).toBe("$0.0042");
    expect(fmtCost(1.5, 1)).toBe("$1.50 + 1 unpriced");
  });
});
