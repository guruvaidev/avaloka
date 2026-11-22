import { useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { formatRelativeTime, formatVersion, shortRunId, type InferenceModel } from "@/api/inference";

interface Props {
  models: InferenceModel[];
  loading: boolean;
  error: boolean;
  selectedRunId: string;
  onRetry: () => void;
  onSelect: (runId: string) => void;
}

const PAGE = 50;

function ModelItem({ model, active, onSelect }: { model: InferenceModel; active: boolean; onSelect?: () => void }) {
  const body = (
    <>
      <span className="break-all font-mono text-xs font-semibold">{model.name ?? "Unnamed run"}</span>
      <span className="flex flex-wrap gap-1">
        {model.taskType && <span className="inf-badge inf-status-starting">{model.taskType}</span>}
        {model.version && <span className="inf-badge inf-status-stopped">{formatVersion(model.version)}</span>}
        {model.usable
          ? model.serviceStarting
            ? <span className="inf-badge inf-status-starting">Starting…</span>
            : <span className={`inf-badge ${model.serviceRunning ? "inf-status-running" : model.endpointStatus === "failed" ? "inf-status-warning" : "inf-status-stopped"}`}>{model.serviceRunning ? (model.serviceAsleep ? "Ready (sleeping)" : "Running") : model.endpointStatus === "failed" ? "Failed" : "Stopped"}</span>
          : <span className="inf-badge inf-status-warning">Not ready</span>}
      </span>
      <span className="text-xs text-muted-foreground">Trained {formatRelativeTime(model.createdAt)}</span>
      <code className="text-xs text-muted-foreground">{shortRunId(model.runId)}</code>
    </>
  );
  if (!onSelect) return <div className="inf-model cursor-not-allowed opacity-60" aria-disabled="true">{body}</div>;
  return (
    <button type="button" onClick={onSelect} aria-current={active ? "true" : undefined} className={`inf-model w-full text-left ${active ? "inf-model-active" : ""}`}>
      {body}
    </button>
  );
}

export function ModelsList({ models, loading, error, selectedRunId, onRetry, onSelect }: Props) {
  const [search, setSearch] = useState("");
  const [limit, setLimit] = useState(PAGE);
  const [showIncomplete, setShowIncomplete] = useState(false);
  const term = search.trim().toLowerCase();
  const { usable, incomplete } = useMemo(() => {
    const matches = models.filter((m) => !term || (m.name ?? "").toLowerCase().includes(term));
    return { usable: matches.filter((m) => m.usable), incomplete: matches.filter((m) => !m.usable) };
  }, [models, term]);

  return (
    <aside className="inf-card lg:sticky lg:top-5" aria-label="Models">
      <h2 className="mb-4 font-semibold">Models</h2>
      <label className="sr-only" htmlFor="model-search">Search models</label>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-3.5 size-4 text-muted-foreground" aria-hidden="true" />
        <Input id="model-search" className="min-h-11 bg-inf-input pl-9" placeholder="Search models" value={search} onChange={(e) => { setSearch(e.target.value); setLimit(PAGE); }} />
      </div>
      <div className="mt-3 space-y-2">
        {loading ? (
          [0, 1, 2, 3].map((n) => <Skeleton key={n} className="h-24 w-full" />)
        ) : error ? (
          <div className="py-5 text-sm" role="alert">
            <p>Models couldn't load.</p>
            <Button variant="outline" className="mt-3 min-h-11" onClick={onRetry}>Retry</Button>
          </div>
        ) : models.length === 0 ? (
          <div className="py-5 text-sm">
            <p className="text-muted-foreground">No trained models yet. Train a model in Chat to run predictions here.</p>
            <Button asChild className="mt-4 min-h-11"><Link to="/proanalysis">Go to Chat</Link></Button>
          </div>
        ) : (
          <>
            {usable.slice(0, limit).map((m) => (
              <ModelItem key={m.key} model={m} active={m.runId === selectedRunId} onSelect={() => m.runId && onSelect(m.runId)} />
            ))}
            {usable.length === 0 && <p className="py-4 text-sm text-muted-foreground">No matching models.</p>}
            {usable.length > limit && (
              <Button variant="outline" className="min-h-11 w-full" onClick={() => setLimit((l) => l + PAGE)}>Load more</Button>
            )}
            {incomplete.length > 0 && (
              <Button variant="ghost" className="min-h-11 w-full justify-start text-xs" aria-expanded={showIncomplete} onClick={() => setShowIncomplete((v) => !v)}>
                {showIncomplete ? "Hide" : "Show"} incomplete runs ({incomplete.length})
              </Button>
            )}
            {showIncomplete && incomplete.map((m) => <ModelItem key={m.key} model={m} active={false} />)}
          </>
        )}
      </div>
    </aside>
  );
}
