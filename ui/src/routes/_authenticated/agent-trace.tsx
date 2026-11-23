import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { backendApi } from "@/lib/api/backendApi";
import type { AgentMetric } from "@/lib/agent-trace/types";
import { AgentMetricsTable, AgentTracePanel } from "@/components/agent-trace/AgentTracePanel";
import { SwarmConsoleNotice, useSwarmConsole } from "@/components/agent-trace/SwarmConsoleGate";

interface AgentTraceSearch {
  thread?: string;
  turn?: string;
  /** The analysis to return to (analyses are addressed by id, not thread). */
  aid?: string;
}

export const Route = createFileRoute("/_authenticated/agent-trace")({
  validateSearch: (search: Record<string, unknown>): AgentTraceSearch => ({
    thread: typeof search.thread === "string" && search.thread ? search.thread : undefined,
    turn: typeof search.turn === "string" && search.turn ? search.turn : undefined,
    aid: typeof search.aid === "string" && search.aid ? search.aid : undefined,
  }),
  head: () => ({
    meta: [
      { title: "Agent trace · Avaloka AI" },
      { name: "description", content: "Every agent step of an analysis: decisions, calculations and inference." },
    ],
  }),
  component: AgentTracePage,
});

type TraceApi = Pick<typeof backendApi, "getAgentTrace" | "getPlanGraph" | "getCapabilities" | "getAgentMetrics">;

/** Without a thread: how every agent has behaved across your analyses. */
function AllAgents({ api }: { api: TraceApi }) {
  const [data, setData] = useState<{ threads: number; turns: number; agents: AgentMetric[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .getAgentMetrics()
      .then(setData)
      .catch((e: { message?: string }) => setError(e?.message || "Could not load agent metrics."));
  }, [api]);
  if (error) return <p className="text-sm text-error-primary">{error}</p>;
  if (!data) return <p className="text-sm text-tertiary">Loading agent metrics…</p>;
  return (
    <div className="space-y-3">
      <p className="text-sm text-secondary">
        Across {data.turns} recorded turn{data.turns === 1 ? "" : "s"} in {data.threads} analys
        {data.threads === 1 ? "is" : "es"}. Open an analysis and choose <em>Agent trace</em> to see one run step by step.
      </p>
      <AgentMetricsTable agents={data.agents} />
    </div>
  );
}

function AgentTracePage() {
  const { thread, turn, aid } = Route.useSearch();
  // A thread's trace is open in every edition: it shows the user how their
  // prompt was acted upon. Only the page's swarm-console parts (the live swarm
  // link, the cross-analysis agent metrics) follow features.swarm_console.
  return <AgentTraceView thread={thread} turn={turn} aid={aid} />;
}

/** The page body, without the router, so it can be rendered on its own. */
export function AgentTraceView({ thread, turn, aid, api = backendApi }: AgentTraceSearch & { api?: TraceApi }) {
  const backClass =
    "inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover";
  const swarmConsole = useSwarmConsole(api);
  const consoleOpen = swarmConsole.state === "open";
  if (!thread && swarmConsole.state === "closed") {
    // Without a thread the page is the operator's agent metrics, which are console.
    return <SwarmConsoleNotice message={swarmConsole.message} aid={aid} title="Agent trace" />;
  }
  return (
    <main className="min-w-0 flex-1 overflow-y-auto bg-primary">
      <div className="mx-auto max-w-7xl space-y-4 px-6 py-6">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-primary">Agent trace</h1>
            <p className="text-sm text-tertiary">
              What each agent decided and calculated, the models it called, and how the agents handed work to each other.
            </p>
          </div>
          <div className="flex items-center gap-2">
          {consoleOpen ? (
            <Link to="/swarm" search={(thread ? { thread, aid } : {}) as never} className={backClass}>
              Swarm demo
            </Link>
          ) : null}
          {aid ? (
            <Link to="/analysis" search={{ aid } as never} className={backClass}>
              Back to analysis
            </Link>
          ) : thread ? (
            <button type="button" className={backClass} onClick={() => window.history.back()}>
              Back
            </button>
          ) : null}
          </div>
        </header>
        {thread ? <AgentTracePanel threadId={thread} turnId={turn} api={api} /> : consoleOpen ? <AllAgents api={api} /> : null}
      </div>
    </main>
  );
}
