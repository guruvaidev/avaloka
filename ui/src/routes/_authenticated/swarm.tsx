import { createFileRoute, Link } from "@tanstack/react-router";
import { backendApi } from "@/lib/api/backendApi";
import { SwarmDemoPanel, type SwarmDemoApi } from "@/components/agent-trace/SwarmDemoPanel";
import { SwarmConsoleGate } from "@/components/agent-trace/SwarmConsoleGate";

interface SwarmSearch {
  /** The analysis this was opened from: its dataset becomes a source option. */
  thread?: string;
  /** The analysis to return to (analyses are addressed by id, not thread). */
  aid?: string;
  dataset?: string;
  /** A mission to open at once. */
  mission?: string;
}

const text = (v: unknown) => (typeof v === "string" && v ? v : undefined);

export const Route = createFileRoute("/_authenticated/swarm")({
  validateSearch: (search: Record<string, unknown>): SwarmSearch => ({
    thread: text(search.thread),
    aid: text(search.aid),
    dataset: text(search.dataset),
    mission: text(search.mission),
  }),
  head: () => ({
    meta: [
      { title: "Swarm · Avaloka AI" },
      { name: "description", content: "Watch Avaloka clone herself, attack a dataset in parallel and converge one answer." },
    ],
  }),
  component: SwarmPage,
});

function SwarmPage() {
  const search = Route.useSearch();
  // The console is Professional and Enterprise (features.swarm_console); a
  // Free caller who types the URL sees the gate copy, never the page.
  return (
    <SwarmConsoleGate aid={search.aid} title="Swarm">
      <SwarmView {...search} />
    </SwarmConsoleGate>
  );
}

/** The page body, without the router, so it can be rendered on its own. */
export function SwarmView({ thread, aid, dataset, mission, api = backendApi, pollMs }: SwarmSearch & { api?: SwarmDemoApi; pollMs?: number }) {
  const linkClass =
    "inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover";
  return (
    <main className="min-w-0 flex-1 overflow-y-auto bg-primary">
      <div className="mx-auto max-w-7xl space-y-4 px-6 py-6">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-primary">Swarm</h1>
            <p className="text-sm text-tertiary">
              Avaloka clones herself: each copy takes a shard of the data and runs the same fireflies, and she converges what they bring
              back into one answer. Live, from the API.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Link to="/agent-trace" search={(thread ? { thread, aid } : {}) as never} className={linkClass}>
              Agent trace
            </Link>
            {aid ? (
              <Link to="/analysis" search={{ aid } as never} className={linkClass}>
                Back to analysis
              </Link>
            ) : null}
          </div>
        </header>
        <SwarmDemoPanel threadId={thread} datasetName={dataset} missionId={mission} api={api} pollMs={pollMs} />
      </div>
    </main>
  );
}
