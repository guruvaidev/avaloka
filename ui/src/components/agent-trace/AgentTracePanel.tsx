// The agent trace of one analysis: the graph, the ordered steps, the agents.
//
// Used by the /agent-trace page and, collapsed, on the analysis page. It reads
// three endpoints: the trace, the granular planner graph of the same turn, and
// the swarm capability (so an edition limit is stated, never implied).

import { useEffect, useMemo, useState } from "react";
import { cx } from "@/lib/utils/cx";
import { backendApi } from "@/lib/api/backendApi";
import { fmtCost, fmtMs, fmtTokens } from "@/lib/agent-trace/layout";
import type { AgentMetric, AgentTrace, PlanGraph, SwarmCapability, TraceTurn } from "@/lib/agent-trace/types";
import { consoleState, isConsoleGateError } from "@/lib/agent-trace/console-gate";
import { PlanGraphView } from "./PlanGraphView";
import { StepDetail, TraceTimeline } from "./TraceSteps";

type Tab = "graph" | "steps" | "agents" | "swarm";

export interface AgentTracePanelProps {
  threadId: string;
  turnId?: string | null;
  /** Bump to refetch, e.g. after a new chat turn. */
  refreshKey?: unknown;
  compact?: boolean;
  className?: string;
  /** Injected in tests; defaults to the real backend client. */
  api?: Pick<typeof backendApi, "getAgentTrace" | "getPlanGraph" | "getCapabilities">;
}

export function AgentMetricsTable({ agents }: { agents: AgentMetric[] }) {
  if (!agents.length) return <p className="text-sm text-tertiary">No agent activity recorded yet.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm" aria-label="Agent metrics">
        <thead>
          <tr className="text-left text-xs text-tertiary">
            <th className="py-1 pr-3 font-medium">Agent</th>
            <th className="pr-3 font-medium">Calls</th>
            <th className="pr-3 font-medium">OK / failed</th>
            <th className="pr-3 font-medium">Retries</th>
            <th className="pr-3 font-medium">Latency p50 / max</th>
            <th className="pr-3 font-medium">LLM calls</th>
            <th className="pr-3 font-medium">Tokens in / out</th>
            <th className="pr-3 font-medium">Cost</th>
            <th className="font-medium">Models</th>
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a.node} className="border-t border-secondary">
              <td className="py-1.5 pr-3">
                <div className="font-medium text-primary">{a.agent}</div>
                <div className="font-mono text-[10px] text-tertiary">{a.node}</div>
              </td>
              <td className="pr-3">{a.activations}</td>
              <td className="pr-3">
                <span className="text-success-primary">{a.succeeded}</span> /{" "}
                <span className={a.failed ? "text-error-primary" : "text-tertiary"}>{a.failed}</span>
              </td>
              <td className="pr-3">{a.retries}</td>
              <td className="pr-3">
                {fmtMs(a.latency_ms.p50)} / {fmtMs(a.latency_ms.max)}
              </td>
              <td className="pr-3">
                {a.llm_calls}
                {a.llm_failures ? <span className="text-error-primary"> ({a.llm_failures} failed)</span> : null}
              </td>
              <td className="pr-3">
                {fmtTokens(a.tokens_in)} / {fmtTokens(a.tokens_out)}
              </td>
              <td className="pr-3">{fmtCost(a.cost_usd, a.unpriced_calls)}</td>
              <td className="font-mono text-[10px] text-tertiary">{a.models.join(", ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-secondary bg-primary px-3 py-2">
      <div className="text-[11px] text-tertiary">{label}</div>
      <div className="text-sm font-semibold text-primary">{value}</div>
    </div>
  );
}

function SwarmView({ turn, swarm, consoleOpen, threadId }: { turn: TraceTurn; swarm?: SwarmCapability | null; consoleOpen?: boolean; threadId: string }) {
  const clones = turn.swarm?.clones ?? [];
  return (
    <div className="space-y-3">
      {swarm?.message ? <SwarmLimitNote swarm={swarm} /> : null}
      {consoleOpen ? (
        <p className="text-sm text-secondary">
          <a
            href={`/swarm?thread=${encodeURIComponent(threadId)}`}
            className="font-semibold text-brand-secondary hover:text-brand-secondary_hover"
          >
            Run a live swarm on this dataset
          </a>{" "}
          — watch the clones attack it in parallel and converge.
        </p>
      ) : null}
      {turn.swarm?.message ? <p className="text-sm text-secondary">{turn.swarm.message}</p> : null}
      {clones.length ? (
        <ul className="grid gap-2 sm:grid-cols-2">
          {clones.map((c) => (
            <li key={c.clone} className="rounded-lg border border-secondary bg-primary p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-semibold text-primary">{c.name ?? c.clone}</span>
                <span
                  className={cx(
                    "rounded-md px-1.5 py-0.5 text-xs font-medium",
                    c.status === "done" && "bg-utility-green-50 text-utility-green-700",
                    c.status === "failed" && "bg-utility-red-50 text-utility-red-700",
                    c.status === "not_run" && "bg-utility-neutral-50 text-utility-neutral-500",
                  )}
                >
                  {c.status === "not_run" ? "not run this turn" : c.status}
                </span>
              </div>
              <p className="text-xs text-tertiary">{c.role}</p>
              {c.focus ? <p className="mt-1 text-xs text-secondary">{c.focus}</p> : null}
              <p className="mt-1 text-[11px] text-tertiary">
                {c.agent_steps.length} agent step{c.agent_steps.length === 1 ? "" : "s"} · {fmtMs(c.latency_ms)}
              </p>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-tertiary">No swarm was dispatched for this turn.</p>
      )}
    </div>
  );
}

export function SwarmLimitNote({ swarm }: { swarm: SwarmCapability }) {
  if (!swarm.message) return null;
  return (
    <p
      role="note"
      className={cx(
        "rounded-lg px-3 py-2 text-xs",
        swarm.limited ? "bg-utility-orange-50 text-utility-orange-700" : "bg-utility-neutral-50 text-utility-neutral-700",
      )}
    >
      {swarm.message}
    </p>
  );
}

export function AgentTracePanel({ threadId, turnId, refreshKey, compact, className, api = backendApi }: AgentTracePanelProps) {
  const [trace, setTrace] = useState<AgentTrace | null>(null);
  const [graph, setGraph] = useState<PlanGraph | null>(null);
  const [swarm, setSwarm] = useState<SwarmCapability | null>(null);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("graph");
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    Promise.all([
      api.getAgentTrace(threadId, turnId ?? undefined),
      api.getPlanGraph(threadId, turnId ?? undefined).catch((e: { status?: number }) => {
        if (e?.status === 404) return null; // no recorded turn: the trace says so too
        throw e;
      }),
      api.getCapabilities().catch(() => null),
    ])
      .then(([t, g, caps]) => {
        if (!live) return;
        setTrace(t);
        setGraph(g);
        setSwarm(caps?.features?.swarm ?? null);
        setConsoleOpen(consoleState(caps) === "open");
      })
      .catch((e: { status?: number; message?: string }) => {
        if (!live) return;
        setError(
          isConsoleGateError(e)
            ? "The agent trace is part of the swarm console, available in Avaloka Professional and Enterprise."
            : e?.status === 404
            ? "No agent trace for this analysis."
            : e?.status === 401
              ? "Your session has expired — please sign in again."
              : e?.message || "Could not load the agent trace.",
        );
      })
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [threadId, turnId, refreshKey, api]);

  const turn = useMemo(() => {
    const turns = trace?.turns ?? [];
    return (turnId && turns.find((t) => t.turn_id === turnId)) || turns[turns.length - 1] || null;
  }, [trace, turnId]);
  const selectedStep = turn?.steps.find((s) => s.step_id === selected) ?? null;

  if (loading) return <p className={cx("text-sm text-tertiary", className)}>Loading the agent trace…</p>;
  if (error) return <p className={cx("text-sm text-error-primary", className)}>{error}</p>;
  if (!trace || !turn) {
    return <p className={cx("text-sm text-tertiary", className)}>No agent steps were recorded for this analysis yet.</p>;
  }

  const tabs: Array<{ id: Tab; label: string }> = [
    { id: "graph", label: "Planner graph" },
    { id: "steps", label: `Steps (${turn.totals.agent_steps})` },
    { id: "agents", label: "Agents" },
    { id: "swarm", label: "Swarm" },
  ];

  return (
    <div className={cx("space-y-3", className)} data-testid="agent-trace-panel">
      {turn.question ? <p className="text-sm text-secondary">“{turn.question}”</p> : null}
      <div className={cx("grid gap-2", compact ? "grid-cols-2 sm:grid-cols-4" : "grid-cols-2 md:grid-cols-5")}>
        <Stat label="Wall time" value={fmtMs(turn.latency_ms)} />
        <Stat label="Agent steps / decisions" value={`${turn.totals.agent_steps} / ${turn.totals.decisions}`} />
        <Stat label="LLM calls" value={`${turn.totals.llm_calls}${turn.totals.llm_failures ? ` (${turn.totals.llm_failures} failed)` : ""}`} />
        <Stat label="Tokens in / out" value={`${fmtTokens(turn.totals.tokens_in)} / ${fmtTokens(turn.totals.tokens_out)}`} />
        {!compact ? <Stat label="Cost" value={fmtCost(turn.totals.cost_usd, turn.totals.unpriced_calls)} /> : null}
      </div>
      {swarm ? <SwarmLimitNote swarm={swarm} /> : null}

      <div role="tablist" aria-label="Agent trace views" className="inline-flex rounded-lg border border-secondary bg-primary p-1 shadow-xs">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            type="button"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={cx(
              "rounded-md px-3 py-1.5 text-sm font-semibold transition",
              tab === t.id ? "bg-secondary text-primary" : "text-tertiary",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className={cx("grid gap-4", !compact && selectedStep ? "lg:grid-cols-[minmax(0,1fr)_380px]" : "")}>
        <div className="min-w-0" role="tabpanel">
          {tab === "graph" ? (
            graph ? (
              <PlanGraphView graph={graph} selectedStepId={selected} onSelect={(id) => setSelected(id)} />
            ) : (
              <p className="text-sm text-tertiary">No planner graph for this turn.</p>
            )
          ) : null}
          {tab === "steps" ? <TraceTimeline turn={turn} selectedStepId={selected} onSelect={setSelected} /> : null}
          {tab === "agents" ? <AgentMetricsTable agents={trace.agents} /> : null}
          {tab === "swarm" ? <SwarmView turn={turn} swarm={swarm} consoleOpen={consoleOpen} threadId={threadId} /> : null}
        </div>
        {selectedStep && (tab === "graph" || tab === "steps") ? (
          <aside className="rounded-xl border border-secondary bg-primary p-4 shadow-xs" aria-label="Step detail">
            <div className="mb-2 flex justify-end">
              <button type="button" className="text-xs text-tertiary hover:text-secondary" onClick={() => setSelected(null)}>
                Close
              </button>
            </div>
            <StepDetail step={selectedStep} turn={turn} />
          </aside>
        ) : null}
      </div>
    </div>
  );
}
