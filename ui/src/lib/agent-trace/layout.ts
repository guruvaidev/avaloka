// Layered layout for the planner graph.
//
// The backend assigns every node a `layer` (longest path from the question)
// and an `order` within its (layer, lane). This file only turns those into
// coordinates and edge paths, so the picture is a pure function of the data
// and is unit-tested without a DOM.
//
// The flow runs top to bottom, one row per layer, because a turn is a long
// pipeline (twenty-odd layers) and a page scrolls vertically. The lanes are
// side-by-side columns: the agents and their decisions, the coding team's
// attempts and checks, and the data lineage (load → sample → operations →
// result), so a coder attempt sits level with the operations it wrote.

import type { PlanEdge, PlanGraph, PlanNode, PlanNodeType } from "./types";

export const LANES = ["agents", "code", "data"] as const;

export interface Box {
  w: number;
  h: number;
}

const SIZES: Record<PlanNodeType, Box> = {
  question: { w: 260, h: 56 },
  agent: { w: 228, h: 72 },
  decision: { w: 212, h: 64 },
  gate: { w: 200, h: 48 },
  data: { w: 300, h: 64 },
  operation: { w: 300, h: 64 },
  check: { w: 208, h: 56 },
  result: { w: 300, h: 112 },
  evidence: { w: 228, h: 72 },
  train: { w: 300, h: 80 },
  evaluate: { w: 300, h: 80 },
};

/** The lanes that make up the data spine: what was done to the user's data. */
export const DATA_LANES = new Set<string>(["data"]);

/**
 * The graph with only the question and the data lane: how the prompt was acted
 * upon, without the agents, their decisions and the coding team's checks.
 * Edges between the remaining nodes are kept; the rest are dropped.
 */
export function dataSpine(graph: Pick<PlanGraph, "nodes" | "edges">): Pick<PlanGraph, "nodes" | "edges"> {
  const nodes = (graph.nodes ?? []).filter((n) => n.type === "question" || DATA_LANES.has(n.lane));
  const keep = new Set(nodes.map((n) => n.id));
  return { nodes, edges: (graph.edges ?? []).filter((e) => keep.has(e.source) && keep.has(e.target)) };
}

export const GAP_X = 20;
export const GAP_Y = 26;
export const LANE_GAP = 56;
export const PAD = 24;
export const HEADER = 28;

export function nodeSize(type: PlanNodeType | string): Box {
  return SIZES[type as PlanNodeType] ?? { w: 228, h: 60 };
}

export interface PositionedNode extends PlanNode {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PositionedEdge extends PlanEdge {
  path: string;
  labelX: number;
  labelY: number;
}

export interface Layout {
  nodes: PositionedNode[];
  edges: PositionedEdge[];
  width: number;
  height: number;
  lanes: Array<{ lane: string; x: number; w: number }>;
}

export function layoutGraph(graph: Pick<PlanGraph, "nodes" | "edges">): Layout {
  // Close up the layers nothing sits in: hiding the agents leaves the data
  // lane starting several layers down, and those rows would be blank.
  const used = [...new Set((graph.nodes ?? []).map((n) => n.layer))].sort((a, b) => a - b);
  const compact = new Map(used.map((layer, i) => [layer, i]));
  const nodes = (graph.nodes ?? []).map((n) => ({ ...n, layer: compact.get(n.layer) ?? 0 }));
  const laneOrder = [...LANES, ...new Set(nodes.map((n) => n.lane).filter((l) => !LANES.includes(l as never)))];

  // Row heights: the tallest node any lane puts in that layer.
  const rowH: number[] = [];
  for (const n of nodes) rowH[n.layer] = Math.max(rowH[n.layer] ?? 0, nodeSize(n.type).h);
  const rowY: number[] = [];
  let y = PAD + HEADER;
  for (let i = 0; i < rowH.length; i++) {
    rowY[i] = y;
    y += (rowH[i] ?? 0) + (rowH[i] ? GAP_Y : 0);
  }
  const height = Math.max(y - GAP_Y + PAD, 160);

  // Lane widths: the widest row of nodes that lane holds in any one layer.
  const lanes: Layout["lanes"] = [];
  const laneX: Record<string, number> = {};
  const laneW: Record<string, number> = {};
  let x = PAD;
  for (const lane of laneOrder) {
    const inLane = nodes.filter((n) => n.lane === lane);
    if (!inLane.length) continue;
    const rows: Record<number, number> = {};
    for (const n of inLane) {
      rows[n.layer] = Math.max(rows[n.layer] ?? 0, (n.order + 1) * (nodeSize(n.type).w + GAP_X) - GAP_X);
    }
    const w = Math.max(...Object.values(rows));
    laneX[lane] = x;
    laneW[lane] = w;
    lanes.push({ lane, x, w });
    x += w + LANE_GAP;
  }
  const width = Math.max(x - LANE_GAP + PAD, 320);

  const positioned: PositionedNode[] = nodes.map((n) => {
    const box = nodeSize(n.type);
    const peers = nodes.filter((m) => m.lane === n.lane && m.layer === n.layer).length;
    const rowWidth = peers * (box.w + GAP_X) - GAP_X;
    const left = (laneX[n.lane] ?? PAD) + ((laneW[n.lane] ?? box.w) - rowWidth) / 2;
    return {
      ...n,
      x: left + n.order * (box.w + GAP_X),
      y: (rowY[n.layer] ?? PAD) + ((rowH[n.layer] ?? box.h) - box.h) / 2,
      w: box.w,
      h: box.h,
    };
  });
  const byId = new Map(positioned.map((n) => [n.id, n]));
  const edges: PositionedEdge[] = [];
  for (const e of graph.edges ?? []) {
    const s = byId.get(e.source);
    const t = byId.get(e.target);
    if (!s || !t) continue;
    edges.push({ ...e, ...edgePath(s, t, e.type) });
  }
  return { nodes: positioned, edges, width, height, lanes };
}

/** A smooth path between two boxes, top to bottom. Retries bow out to the side. */
export function edgePath(
  s: Pick<PositionedNode, "x" | "y" | "w" | "h">,
  t: Pick<PositionedNode, "x" | "y" | "w" | "h">,
  type: string,
): { path: string; labelX: number; labelY: number } {
  if (type === "retry") {
    const sx = s.x + s.w;
    const sy = s.y + s.h / 2;
    const tx = t.x + t.w;
    const ty = t.y + t.h / 2;
    const bow = Math.max(sx, tx) + 48;
    return {
      path: `M ${sx} ${sy} C ${bow} ${sy}, ${bow} ${ty}, ${tx} ${ty}`,
      labelX: bow - 8,
      labelY: (sy + ty) / 2,
    };
  }
  const sx = s.x + s.w / 2;
  const sy = s.y + s.h;
  const tx = t.x + t.w / 2;
  const ty = t.y;
  const mid = (sy + ty) / 2;
  return {
    path: `M ${sx} ${sy} C ${sx} ${mid}, ${tx} ${mid}, ${tx} ${ty}`,
    labelX: (sx + tx) / 2 + 6,
    labelY: mid,
  };
}

// ---------------------------------------------------------------- appearance

export interface NodeTone {
  fill: string;
  stroke: string;
  accent: string;
  label: string;
}

const v = (name: string) => `var(--color-utility-${name})`;

/** Colour by node type, and by outcome where the node has one. */
export function nodeTone(node: Pick<PlanNode, "type" | "status" | "value"> & { subtitle?: string | null }): NodeTone {
  const failed =
    node.status === "error" ||
    node.value === "fail" ||
    node.value === "failed" ||
    node.value === "unsafe" ||
    (node.type === "agent" && typeof node.subtitle === "string" && node.subtitle.startsWith("artifact: error"));
  if (failed) return { fill: v("red-50"), stroke: v("red-500"), accent: v("red-700"), label: "failed" };
  switch (node.type) {
    case "question":
      return { fill: v("neutral-50"), stroke: v("neutral-500"), accent: v("neutral-700"), label: "question" };
    case "agent":
      return { fill: v("brand-50"), stroke: v("brand-500"), accent: v("brand-700"), label: "agent" };
    case "decision":
      return { fill: v("purple-50"), stroke: v("purple-500"), accent: v("purple-700"), label: "decision" };
    case "gate":
      return { fill: v("orange-50"), stroke: v("orange-500"), accent: v("orange-700"), label: "gate" };
    case "data":
      return { fill: v("blue-50"), stroke: v("blue-500"), accent: v("blue-700"), label: "data" };
    case "operation":
      return { fill: v("indigo-50"), stroke: v("indigo-500"), accent: v("indigo-700"), label: "operation" };
    case "check":
      return { fill: v("green-50"), stroke: v("green-500"), accent: v("green-700"), label: "check" };
    case "result":
      return { fill: v("green-50"), stroke: v("green-500"), accent: v("green-700"), label: "result" };
    case "evidence":
      return { fill: v("yellow-50"), stroke: v("yellow-500"), accent: v("yellow-700"), label: "evidence" };
    case "train":
      return { fill: v("pink-50"), stroke: v("pink-500"), accent: v("pink-700"), label: "train" };
    case "evaluate":
      return { fill: v("fuchsia-50"), stroke: v("fuchsia-500"), accent: v("fuchsia-700"), label: "evaluate" };
    default:
      return { fill: v("neutral-50"), stroke: v("neutral-500"), accent: v("neutral-700"), label: String(node.type) };
  }
}

export const EDGE_STYLE: Record<string, { dash?: string; stroke: string }> = {
  flow: { stroke: v("neutral-400") },
  route: { stroke: v("purple-500") },
  handoff: { stroke: v("brand-500") },
  data: { stroke: v("blue-500"), dash: "5 4" },
  check: { stroke: v("green-500") },
  retry: { stroke: v("red-500"), dash: "6 4" },
  evidence: { stroke: v("yellow-500"), dash: "2 4" },
  gate: { stroke: v("orange-500"), dash: "3 3" },
};

// ---------------------------------------------------------------- formatting

export function fmtMs(ms?: number | null): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

export function fmtTokens(n?: number | null): string {
  if (n == null) return "—";
  return n >= 10_000 ? `${(n / 1000).toFixed(0)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

/** Unknown cost is shown as unknown, never as $0. */
export function fmtCost(usd?: number | null, unpriced = 0): string {
  if (usd == null) return unpriced ? "unpriced" : "—";
  const s = usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
  return unpriced ? `${s} + ${unpriced} unpriced` : s;
}

export function truncate(text: string | null | undefined, n: number): string {
  if (!text) return "";
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}
