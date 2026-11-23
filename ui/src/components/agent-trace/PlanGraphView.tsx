// The planner graph, drawn from the backend's layered layout.
//
// Plain SVG: the UI ships no graph library, and this picture needs no physics —
// the backend already decided each node's layer and order. Every node from a
// recorded step is a button that selects that step, so the graph and the trace
// timeline always point at the same thing.
//
// The default picture is the data spine alone — load, the operations with their
// columns and formulas, the result or the models trained and evaluated — because
// that is how the user's prompt was acted upon. The agents, their decisions and
// the coding team's checks are one toggle away.

import { useMemo, useState } from "react";
import { cx } from "@/lib/utils/cx";
import {
  EDGE_STYLE,
  dataSpine,
  fmtCost,
  fmtMs,
  fmtTokens,
  layoutGraph,
  nodeTone,
  truncate,
  type PositionedNode,
} from "@/lib/agent-trace/layout";
import type { PlanGraph, PlanNodeType } from "@/lib/agent-trace/types";

const LANE_TITLES: Record<string, string> = {
  agents: "Agents and decisions",
  code: "Coding team: attempts and checks",
  data: "What was done to your data",
};

const LEGEND: Array<{ type: PlanNodeType; label: string; spine?: boolean }> = [
  { type: "data", label: "Data", spine: true },
  { type: "operation", label: "Operation", spine: true },
  { type: "result", label: "Result", spine: true },
  { type: "train", label: "Model trained", spine: true },
  { type: "evaluate", label: "Evaluation", spine: true },
  { type: "agent", label: "Agent" },
  { type: "decision", label: "Decision" },
  { type: "gate", label: "Approval gate" },
  { type: "check", label: "Check" },
  { type: "evidence", label: "Evidence" },
];

const SOURCE_TEXT: Record<string, string> = {
  executed: "from the code that ran",
  validated: "from the last validated attempt",
  generated: "from the generated code",
};

function valueText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toPrecision(4);
  return String(v);
}

function NodeShape({ n }: { n: PositionedNode }) {
  const tone = nodeTone(n);
  if (n.type === "decision") {
    const cx0 = n.x + n.w / 2;
    const cy0 = n.y + n.h / 2;
    return (
      <polygon
        points={`${n.x},${cy0} ${cx0},${n.y} ${n.x + n.w},${cy0} ${cx0},${n.y + n.h}`}
        fill={tone.fill}
        stroke={tone.stroke}
        strokeWidth={1.5}
      />
    );
  }
  const radius = n.type === "question" ? n.h / 2 : n.type === "gate" ? 4 : 10;
  return (
    <>
      <rect
        x={n.x}
        y={n.y}
        width={n.w}
        height={n.h}
        rx={radius}
        fill={tone.fill}
        stroke={tone.stroke}
        strokeWidth={1.5}
        strokeDasharray={n.type === "gate" ? "4 3" : undefined}
      />
      {n.type !== "question" && n.type !== "gate" ? (
        <rect x={n.x} y={n.y + 8} width={4} height={n.h - 16} rx={2} fill={tone.stroke} />
      ) : null}
    </>
  );
}

function NodeText({ n }: { n: PositionedNode }) {
  const tone = nodeTone(n);
  const left = n.type === "decision" ? n.x + n.w / 2 : n.x + 14;
  const anchor = n.type === "decision" || n.type === "question" ? "middle" : "start";
  const x = n.type === "question" ? n.x + n.w / 2 : left;
  const maxChars = Math.floor((n.w - 24) / 6.4);
  const lines: Array<{ text: string; cls: string; mono?: boolean }> = [];
  lines.push({ text: truncate(n.label, maxChars), cls: "font-semibold" });

  if (n.type === "operation" && n.formula) {
    lines.push({ text: truncate(n.formula, maxChars + 4), cls: "", mono: true });
  } else if (n.type === "evaluate") {
    // The primary metric against its baseline first; the verdict and the rest after.
    const ev = n.evaluation;
    const primary = ev?.primary_metric ?? undefined;
    const score = primary ? ev?.metrics?.[primary] : undefined;
    if (primary && score != null) {
      const base = ev?.baseline?.[primary];
      lines.push({ text: truncate(`${primary} ${valueText(score)}${base != null ? ` vs baseline ${valueText(base)}` : ""}`, maxChars), cls: "" });
    } else if (n.subtitle) {
      lines.push({ text: truncate(n.subtitle, maxChars), cls: "" });
    }
    const verdict =
      n.value === "beats_baseline" ? "beats the baseline" : n.value === "does_not_beat_baseline" ? "does not beat the baseline" : null;
    const others = Object.entries(ev?.metrics ?? {})
      .filter(([k]) => k !== primary)
      .slice(0, 2)
      .map(([k, v]) => `${k.replace(/_/g, " ")} ${valueText(v)}`);
    if (verdict || others.length) lines.push({ text: truncate([verdict, ...others].filter(Boolean).join(" · "), maxChars), cls: "" });
  } else if (n.type === "train") {
    if (n.subtitle) lines.push({ text: truncate(n.subtitle, maxChars), cls: "" });
    const parts = Object.entries(n.values ?? {})
      .filter(([, v]) => v != null && typeof v !== "object")
      .slice(0, 3)
      .map(([k, v]) => `${k.replace(/_/g, " ")} ${valueText(v)}`);
    if (parts.length) lines.push({ text: truncate(parts.join(" · "), maxChars), cls: "" });
  } else if (n.type === "decision") {
    if (n.value != null) lines.push({ text: `→ ${valueText(n.value)}`, cls: "" });
  } else if (n.type === "check" || n.type === "gate") {
    lines.push({ text: valueText(n.value) || (n.status ?? ""), cls: "uppercase tracking-wide" });
  } else if (n.subtitle) {
    lines.push({ text: truncate(n.subtitle, maxChars), cls: "" });
  } else if (n.values) {
    const parts = Object.entries(n.values)
      .filter(([, v]) => v != null && typeof v !== "object")
      .slice(0, 2)
      .map(([k, v]) => `${k.replace(/_/g, " ")} ${valueText(v)}`);
    if (parts.length) lines.push({ text: truncate(parts.join(" · "), maxChars), cls: "" });
  }
  if (n.columns?.length && n.type !== "train") {
    // The columns this node touches, as a badge line; a train node names its
    // target and features in its values instead.
    lines.push({ text: truncate(`⌗ ${n.columns.join(", ")}`, maxChars), cls: "opacity-80", mono: true });
  }
  const m = n.metrics;
  if (m && (m.latency_ms != null || m.llm_calls)) {
    const bits = [fmtMs(m.latency_ms)];
    if (m.llm_calls) bits.push(`${fmtTokens(m.tokens)} tok`, fmtCost(m.cost_usd));
    lines.push({ text: bits.join(" · "), cls: "opacity-80" });
  }

  const top = n.type === "result" ? n.y + 18 : n.y + n.h / 2 - ((lines.length - 1) * 15) / 2 + 4;
  return (
    <>
      {lines.map((line, i) => (
        <text
          key={i}
          x={x}
          y={top + i * 15}
          textAnchor={anchor}
          className={cx("text-[11px]", line.cls, line.mono && "font-mono text-[10px]")}
          fill={i === 0 ? tone.accent : "var(--color-text-secondary, currentColor)"}
        >
          {line.text}
        </text>
      ))}
      {n.type === "result" && n.table ? (
        <g>
          {n.table.rows.slice(0, 4).map((row, r) => (
            <text
              key={r}
              x={n.x + 14}
              y={top + lines.length * 15 + 2 + r * 15}
              className="font-mono text-[10px]"
              fill="var(--color-text-secondary, currentColor)"
            >
              {truncate(row.map(valueText).join("  ·  "), Math.floor((n.w - 24) / 6))}
            </text>
          ))}
          {n.table.total_rows > 4 ? (
            <text x={n.x + 14} y={n.y + n.h - 8} className="text-[10px]" fill="var(--color-text-tertiary, currentColor)">
              {n.table.total_rows.toLocaleString()} rows
            </text>
          ) : null}
        </g>
      ) : null}
    </>
  );
}

export interface PlanGraphViewProps {
  graph: PlanGraph;
  selectedStepId?: string | null;
  onSelect?: (stepId: string, nodeId: string) => void;
  className?: string;
  /** Start with the agents, decisions and checks shown. Default: the data spine only. */
  showAgents?: boolean;
}

export function PlanGraphView({ graph, selectedStepId, onSelect, className, showAgents = false }: PlanGraphViewProps) {
  const [agentsOn, setAgentsOn] = useState(showAgents);
  const spine = useMemo(() => dataSpine(graph), [graph]);
  // A turn with no data lane (a clarification, a refusal) has only its agents to show.
  const hasSpine = spine.nodes.some((n) => n.type !== "question");
  const shown = agentsOn || !hasSpine ? graph : spine;
  const layout = useMemo(() => layoutGraph(shown), [shown]);
  const [zoom, setZoom] = useState(1);
  const [hover, setHover] = useState<string | null>(null);

  if (!layout.nodes.length) {
    return <p className="text-sm text-tertiary">No steps were recorded for this turn.</p>;
  }
  const legend = LEGEND.filter((item) => item.spine || shown === graph);

  return (
    <div className={cx("flex flex-col gap-2", className)}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <ul className="flex flex-wrap items-center gap-3" aria-label="Node types">
          {legend.map((item) => {
            const tone = nodeTone({ type: item.type });
            return (
              <li key={item.type} className="flex items-center gap-1.5 text-xs text-tertiary">
                <span
                  aria-hidden
                  className={cx("inline-block size-3 border", item.type === "decision" ? "rotate-45" : "rounded-sm")}
                  style={{ background: tone.fill, borderColor: tone.stroke }}
                />
                {item.label}
              </li>
            );
          })}
        </ul>
        <div className="flex items-center gap-2">
          {hasSpine ? (
            <label className="flex items-center gap-1.5 text-xs text-secondary">
              <input
                type="checkbox"
                className="size-3.5"
                checked={agentsOn}
                onChange={(ev) => setAgentsOn(ev.target.checked)}
              />
              Show agents and checks
            </label>
          ) : (
            <span className="text-xs text-tertiary">No data operations were recorded; showing the agents.</span>
          )}
          <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-0.5 shadow-xs">
          <button
            type="button"
            className="rounded-md px-2 py-1 text-xs font-semibold text-secondary hover:bg-primary_hover"
            onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.15).toFixed(2)))}
            aria-label="Zoom out"
          >
            −
          </button>
          <span className="w-12 text-center text-xs text-tertiary">{Math.round(zoom * 100)}%</span>
          <button
            type="button"
            className="rounded-md px-2 py-1 text-xs font-semibold text-secondary hover:bg-primary_hover"
            onClick={() => setZoom((z) => Math.min(1.6, +(z + 0.15).toFixed(2)))}
            aria-label="Zoom in"
          >
            +
          </button>
          </div>
        </div>
      </div>
      <div className="max-h-[80vh] overflow-auto rounded-xl border border-secondary bg-primary">
        <svg
          role="img"
          aria-label="Planner graph"
          width={layout.width * zoom}
          height={layout.height * zoom}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
        >
          <defs>
            {Object.entries(EDGE_STYLE).map(([type, style]) => (
              <marker
                key={type}
                id={`arrow-${type}`}
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill={style.stroke} />
              </marker>
            ))}
          </defs>
          {layout.lanes.map((band) => (
            <g key={band.lane}>
              <rect
                x={band.x - 12}
                y={8}
                width={band.w + 24}
                height={layout.height - 16}
                rx={12}
                fill="var(--color-utility-neutral-50)"
                opacity={0.6}
              />
              <text
                x={band.x + band.w / 2}
                y={30}
                textAnchor="middle"
                className="text-[10px] font-semibold uppercase tracking-wider"
                fill="var(--color-utility-neutral-500)"
              >
                {LANE_TITLES[band.lane] ?? band.lane}
              </text>
            </g>
          ))}
          {layout.edges.map((e) => {
            const style = EDGE_STYLE[e.type] ?? EDGE_STYLE.flow;
            const lit = hover && (e.source === hover || e.target === hover);
            return (
              <g key={e.id} data-edge-type={e.type}>
                <path
                  d={e.path}
                  fill="none"
                  stroke={style.stroke}
                  strokeWidth={lit ? 2.5 : 1.5}
                  strokeDasharray={style.dash}
                  opacity={hover && !lit ? 0.35 : 1}
                  markerEnd={`url(#arrow-${EDGE_STYLE[e.type] ? e.type : "flow"})`}
                />
                {e.label ? (
                  <text
                    x={e.labelX}
                    y={e.labelY}
                    textAnchor="middle"
                    className="text-[10px]"
                    fill="var(--color-utility-neutral-500)"
                  >
                    {truncate(e.label, 28)}
                  </text>
                ) : null}
              </g>
            );
          })}
          {layout.nodes.map((n) => {
            const selected = !!n.step_id && n.step_id === selectedStepId;
            const clickable = !!n.step_id && !!onSelect;
            return (
              <g
                key={n.id}
                data-node-type={n.type}
                data-node-id={n.id}
                role={clickable ? "button" : undefined}
                tabIndex={clickable ? 0 : undefined}
                aria-label={`${n.type}: ${n.label}${n.value != null ? ` (${valueText(n.value)})` : ""}`}
                aria-pressed={clickable ? selected : undefined}
                className={cx(clickable && "cursor-pointer outline-none")}
                onMouseEnter={() => setHover(n.id)}
                onMouseLeave={() => setHover(null)}
                onClick={() => clickable && onSelect?.(n.step_id!, n.id)}
                onKeyDown={(ev) => {
                  if (clickable && (ev.key === "Enter" || ev.key === " ")) {
                    ev.preventDefault();
                    onSelect?.(n.step_id!, n.id);
                  }
                }}
              >
                <title>
                  {[
                    n.label,
                    n.reason,
                    n.formula,
                    n.columns?.length ? `columns: ${n.columns.join(", ")}` : null,
                    n.source ? SOURCE_TEXT[n.source] ?? n.source : null,
                  ]
                    .filter(Boolean)
                    .join("\n")}
                </title>
                {selected ? (
                  <rect
                    x={n.x - 4}
                    y={n.y - 4}
                    width={n.w + 8}
                    height={n.h + 8}
                    rx={14}
                    fill="none"
                    stroke="var(--color-utility-brand-500)"
                    strokeWidth={2}
                  />
                ) : null}
                <NodeShape n={n} />
                <NodeText n={n} />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
