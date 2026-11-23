// The trace as an ordered timeline, and the full detail of one step.
//
// Decisions are shown by what the agent decided and the reason it recorded;
// calculations by the code, the operations with their formulas, the data scope
// and the values. This is the user's own analysis, shown back to them.

import { modelSwapLine } from "@/lib/agent-trace/model-swap";
import { useState } from "react";
import { cx } from "@/lib/utils/cx";
import { fmtCost, fmtMs, fmtTokens, truncate } from "@/lib/agent-trace/layout";
import type { SmallTable, TraceHandoff, TraceStep, TraceTurn } from "@/lib/agent-trace/types";
import { CommentaryText } from "@/components/chat/CommentaryText";

function Chip({ tone = "neutral", children }: { tone?: "neutral" | "good" | "bad" | "brand" | "purple" | "warn"; children: React.ReactNode }) {
  const tones = {
    neutral: "bg-utility-neutral-50 text-utility-neutral-700 ring-utility-neutral-200",
    good: "bg-utility-green-50 text-utility-green-700 ring-utility-green-200",
    bad: "bg-utility-red-50 text-utility-red-700 ring-utility-red-200",
    brand: "bg-utility-brand-50 text-utility-brand-700 ring-utility-brand-200",
    purple: "bg-utility-purple-50 text-utility-purple-700 ring-utility-purple-200",
    warn: "bg-utility-orange-50 text-utility-orange-700 ring-utility-orange-200",
  } as const;
  return (
    <span className={cx("inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset", tones[tone])}>
      {children}
    </span>
  );
}

/**
 * What a step's decision chip says. The claim verifier's "safe" is a count
 * with no problem found, not a verdict on every claim (owner's ruling): it
 * reads "checked", in the step list as in the step detail.
 */
export function decisionLabel(decision: TraceStep["decision"], fallback = "—"): string {
  if (!decision) return fallback;
  if (decision.type === "evidence" && decision.value === "safe") return "checked";
  return String(decision.chosen ?? decision.value ?? fallback);
}

function decisionTone(step: TraceStep): "good" | "bad" | "purple" | "neutral" {
  const v = step.decision?.value;
  if (step.status === "error" || v === "fail" || v === "failed" || v === "unsafe" || v === "refused") return "bad";
  if (typeof v === "string" && v.startsWith("error")) return "bad";
  if (v === "pass" || v === "success" || v === "safe" || v === "beats_baseline") return "good";
  if (step.kind === "decision") return "purple";
  return "neutral";
}

export function ResultTable({ table }: { table: SmallTable }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-secondary">
      <table className="min-w-full text-xs">
        <thead className="bg-secondary">
          <tr>
            {table.columns.map((c) => (
              <th key={c} className="px-2 py-1 text-left font-semibold text-secondary">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-t border-secondary">
              {row.map((cell, j) => (
                <td key={j} className="px-2 py-1 font-mono text-primary">
                  {cell == null ? "—" : String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {table.truncated ? (
        <p className="px-2 py-1 text-xs text-tertiary">
          Showing {table.rows.length} of {table.total_rows.toLocaleString()} rows.
        </p>
      ) : null}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-1.5">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-tertiary">{title}</h4>
      {children}
    </section>
  );
}

function KeyValues({ values }: { values?: Record<string, unknown> | null }) {
  const entries = Object.entries(values ?? {}).filter(([, v]) => v != null && v !== "");
  if (!entries.length) return null;
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
      {entries.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-tertiary">{k.replace(/_/g, " ")}</dt>
          <dd className="font-mono text-primary">{typeof v === "object" ? JSON.stringify(v) : String(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function StepDetail({ step, turn }: { step: TraceStep; turn: TraceTurn }) {
  const calc = step.calculation;
  const into = turn.handoffs.find((h) => h.to_step === step.step_id);
  const out = turn.handoffs.find((h) => h.from_step === step.step_id);
  return (
    <div className="space-y-4" data-testid="step-detail">
      <header className="space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-sm font-semibold text-primary">
            {step.kind === "decision" ? `Decision · ${step.decision?.router?.replace(/_/g, " ")}` : step.agent ?? step.node}
          </h3>
          {step.clone ? <Chip tone="brand">clone: {step.clone}</Chip> : null}
          <Chip tone={step.status === "error" ? "bad" : "neutral"}>{fmtMs(step.latency_ms)}</Chip>
        </div>
        {step.role ? <p className="text-xs text-tertiary">{step.role}</p> : null}
      </header>

      {step.decision ? (
        <Section title="Decision">
          <div className="flex flex-wrap items-center gap-2">
            <Chip tone={decisionTone(step)}>
              {step.decision.type}: {decisionLabel(step.decision)}
            </Chip>
            {step.decision.confidence != null ? (
              <span className="text-xs text-tertiary">confidence {Number(step.decision.confidence).toFixed(2)}</span>
            ) : null}
          </div>
          {step.decision.reason ? (
            <p className="whitespace-pre-wrap text-sm text-secondary">
              {/* A model's reason (intent, plan, clarify, logical review) travels with commentary segments. */}
              <CommentaryText text={step.decision.reason} segments={step.decision.reason_segments} />
            </p>
          ) : null}
          {step.decision.signals ? <KeyValues values={step.decision.signals as Record<string, unknown>} /> : null}
        </Section>
      ) : null}

      {step.gates?.length ? (
        <Section title="Approval gates">
          {step.gates.map((g) => (
            <p key={g.gate} className="text-sm text-secondary">
              <Chip tone="warn">{g.gate.replace(/_/g, " ")}</Chip> {g.status}{" "}
              {g.reason ? <>— <CommentaryText text={g.reason} segments={g.reason_segments} /></> : null}
            </p>
          ))}
        </Section>
      ) : null}

      {calc ? (
        <Section title="Calculation">
          {calc.operation ? <p className="text-sm text-secondary">{calc.operation}</p> : null}
          <KeyValues values={calc.data_scope} />
          <KeyValues values={calc.values ?? undefined} />
          {calc.operations?.length ? (
            <ol className="space-y-1">
              {calc.operations.map((op, i) => (
                <li key={i} className="rounded-md bg-secondary px-2 py-1">
                  <span className="text-xs font-semibold text-primary">{op.op.replace(/_/g, " ")}</span>
                  {op.line ? <span className="ml-2 text-xs text-tertiary">line {op.line}</span> : null}
                  {op.formula ? <code className="mt-0.5 block font-mono text-xs text-secondary">{op.formula}</code> : null}
                </li>
              ))}
            </ol>
          ) : null}
          {calc.result ? <ResultTable table={calc.result} /> : null}
          {calc.findings?.length ? (
            <ul className="space-y-1">
              {calc.findings.map((f, i) => (
                <li key={i} className="text-xs text-secondary">
                  <Chip tone={f.verdict === "supported" ? "good" : "bad"}>{String(f.verdict)}</Chip> {String(f.claim ?? "")}
                </li>
              ))}
            </ul>
          ) : null}
          {calc.charts?.length ? (
            <ul className="space-y-1">
              {calc.charts.map((c, i) => (
                <li key={i} className="text-xs text-secondary">
                  <Chip tone="brand">{c.type ?? "chart"}</Chip> <span className="font-medium text-primary">{c.title}</span>
                  {c.reason ? <span className="block text-tertiary"><CommentaryText text={c.reason} fromSuffix /></span> : null}
                </li>
              ))}
            </ul>
          ) : null}
          {calc.answer ? (
            <p className="whitespace-pre-wrap text-sm text-secondary">
              <CommentaryText text={calc.answer} segments={calc.answer_segments} />
            </p>
          ) : null}
          {calc.code ? (
            <details className="rounded-lg border border-secondary">
              <summary className="cursor-pointer px-2 py-1 text-xs font-semibold text-secondary">Code</summary>
              <pre className="max-h-80 overflow-auto px-3 py-2 font-mono text-xs text-primary">{calc.code}</pre>
            </details>
          ) : null}
        </Section>
      ) : null}

      {step.inference?.length ? (
        <Section title="Inference">
          <table className="w-full text-xs [&_td]:pr-3 [&_th]:pr-3">
            <thead>
              <tr className="text-left text-tertiary">
                <th className="font-medium">Provider / model</th>
                <th className="font-medium">Tokens in / out</th>
                <th className="font-medium">Cost</th>
                <th className="font-medium">Latency</th>
                <th className="font-medium">Outcome</th>
              </tr>
            </thead>
            <tbody>
              {step.inference.flatMap((c, i) => [
                <tr key={i} className="border-t border-secondary">
                  <td className="py-1 font-mono">{c.provider}/{c.model}</td>
                  <td>{fmtTokens(c.tokens_in)} / {fmtTokens(c.tokens_out)}</td>
                  <td title={c.cost_source ? `from ${c.cost_source}` : "no price configured"}>
                    {fmtCost(c.cost_usd, c.cost_usd == null ? 1 : 0)}
                  </td>
                  <td>{fmtMs(c.latency_ms)}</td>
                  <td>
                    <Chip tone={c.outcome === "success" ? "good" : "bad"}>{c.outcome}{c.error_class ? ` · ${c.error_class}` : ""}{c.reason ? ` · ${c.reason}` : ""}</Chip>
                  </td>
                </tr>,
                // A fallback tier answered this call: say which model and why.
                ...(modelSwapLine(c)
                  ? [
                      <tr key={`${i}-swap`} data-testid="trace-model-swap">
                        <td colSpan={5} className="pb-1 text-tertiary">{modelSwapLine(c)}</td>
                      </tr>,
                    ]
                  : []),
              ])}
            </tbody>
          </table>
        </Section>
      ) : null}

      {into || out ? (
        <Section title="Handoffs">
          {into ? <HandoffLine h={into} direction="from" /> : null}
          {out ? <HandoffLine h={out} direction="to" /> : null}
        </Section>
      ) : null}
    </div>
  );
}

function HandoffLine({ h, direction }: { h: TraceHandoff; direction: "from" | "to" }) {
  const other = direction === "from" ? h.from : h.to;
  return (
    <p className="text-xs text-secondary">
      {direction === "from" ? "Received from" : "Handed to"} <span className="font-semibold">{other}</span>
      {h.route ? <> via route <Chip tone="purple">{h.route}</Chip></> : null}
      {h.passed.length ? <> — passed {h.passed.slice(0, 8).join(", ")}{h.passed.length > 8 ? "…" : ""}</> : null}
    </p>
  );
}

export interface TraceTimelineProps {
  turn: TraceTurn;
  selectedStepId?: string | null;
  onSelect: (stepId: string) => void;
}

export function TraceTimeline({ turn, selectedStepId, onSelect }: TraceTimelineProps) {
  const top = turn.steps.filter((s) => !s.parent_id);
  const kids = (id: string) => turn.steps.filter((s) => s.parent_id === id);
  const longest = Math.max(1, ...turn.steps.map((s) => s.latency_ms ?? 0));
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const row = (s: TraceStep, depth: number) => {
    const children = s.kind === "agent" ? kids(s.step_id) : [];
    const expanded = open[s.step_id] ?? true;
    const calls = s.inference?.length ?? 0;
    return (
      <li key={s.step_id}>
        <div
          className={cx(
            "group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-primary_hover",
            selectedStepId === s.step_id && "bg-secondary ring-1 ring-utility-brand-200",
          )}
          style={{ paddingLeft: 8 + depth * 18 }}
        >
          {children.length ? (
            <button
              type="button"
              aria-label={expanded ? "Collapse" : "Expand"}
              aria-expanded={expanded}
              className="w-4 text-xs text-tertiary"
              onClick={() => setOpen((o) => ({ ...o, [s.step_id]: !expanded }))}
            >
              {expanded ? "▾" : "▸"}
            </button>
          ) : (
            <span className="w-4" />
          )}
          <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => onSelect(s.step_id)}>
            <span className="w-6 shrink-0 text-right font-mono text-[10px] text-tertiary">{s.seq}</span>
            <span className="truncate text-sm font-medium text-primary">
              {s.kind === "decision" ? `↳ ${s.decision?.router?.replace(/_/g, " ")}` : s.agent ?? s.node}
            </span>
            {s.decision ? (
              <Chip tone={decisionTone(s)}>
                {truncate(decisionLabel(s.decision, s.decision.type), 48)}
              </Chip>
            ) : null}
            {calls ? <span className="text-xs text-tertiary">{calls} LLM</span> : null}
          </button>
          {s.kind === "agent" ? (
            <div className="hidden w-32 shrink-0 items-center gap-1.5 sm:flex" title={fmtMs(s.latency_ms)}>
              <div className="h-1.5 flex-1 rounded-full bg-tertiary">
                <div
                  className={cx("h-1.5 rounded-full", s.status === "error" ? "bg-error-solid" : "bg-brand-solid")}
                  style={{ width: `${Math.max(3, ((s.latency_ms ?? 0) / longest) * 100)}%` }}
                />
              </div>
              <span className="w-12 text-right text-[10px] text-tertiary">{fmtMs(s.latency_ms)}</span>
            </div>
          ) : null}
        </div>
        {children.length && expanded ? <ul>{children.map((c) => row(c, depth + 1))}</ul> : null}
      </li>
    );
  };

  return (
    <ul className="space-y-0.5" aria-label="Agent steps">
      {top.map((s) => row(s, 0))}
    </ul>
  );
}
