import { useMemo, useState } from "react";
import { XClose, SearchLg, Folder, BarChartSquare02, CheckCircle } from "@untitledui/icons";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export type PickableAnalysis = {
  id: string;
  name: string;
  project_id: string | null;
  project_name: string | null;
};

interface Props {
  analyses: PickableAnalysis[];
  projects?: { id: string; name: string }[];
  selected: string[];
  onClose: () => void;
  onConfirm: (ids: string[]) => void;
}

export function AnalysisPickerModal({ analyses, projects: allProjects = [], selected, onClose, onConfirm }: Props) {
  const [mode, setMode] = useState<"root" | "pro" | "project">("root");
  const [projectId, setProjectId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [picked, setPicked] = useState<string[]>(selected);

  const pro = useMemo(() => analyses.filter((a) => !a.project_id), [analyses]);
  const projects = useMemo(() => {
    const map = new Map<string, { id: string; name: string; count: number }>();
    for (const a of analyses) {
      if (!a.project_id) continue;
      const prev = map.get(a.project_id);
      if (prev) prev.count += 1;
      else map.set(a.project_id, { id: a.project_id, name: a.project_name ?? "Project", count: 1 });
    }
    for (const p of allProjects) {
      if (!map.has(p.id)) map.set(p.id, { id: p.id, name: p.name, count: 0 });
      else map.get(p.id)!.name = p.name;
    }
    return Array.from(map.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [analyses, allProjects]);

  const filter = (list: PickableAnalysis[]) => {
    const q = query.trim().toLowerCase();
    return q ? list.filter((a) => a.name.toLowerCase().includes(q)) : list;
  };

  const toggle = (id: string) =>
    setPicked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const AnalysisRow = ({ a }: { a: PickableAnalysis }) => {
    const on = picked.includes(a.id);
    return (
      <button
        type="button"
        onClick={() => toggle(a.id)}
        className={cn(
          "flex w-full items-center justify-between gap-2 rounded-lg border px-2.5 py-2 text-left text-xs transition-colors",
          on
            ? "border-brand-600 bg-brand-600/10 text-foreground"
            : "border-border bg-background text-muted-foreground hover:border-brand-600/40 hover:text-foreground",
        )}
      >
        <span className="flex min-w-0 items-center gap-2">
          <BarChartSquare02 className="size-3.5 shrink-0 text-brand-600" />
          <span className="truncate">{a.name}</span>
        </span>
        {on && <CheckCircle className="size-3.5 shrink-0 text-brand-600" />}
      </button>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-3" onClick={onClose}>
      <div
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-xl"
      >
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">Attach an analysis</h3>
            <p className="text-[10px] text-muted-foreground">
              {mode === "root" ? "Choose where the analysis lives." : "Select one or more analyses."}
            </p>
          </div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            className="rounded-lg p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <XClose className="size-4" />
          </button>
        </div>

        {mode === "root" ? (
          <div className="grid gap-3 p-4 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => setMode("pro")}
              className="flex flex-col gap-1 rounded-xl border border-border bg-background p-4 text-left transition-colors hover:border-brand-600/50 hover:bg-brand-600/5"
            >
              <BarChartSquare02 className="size-5 text-brand-600" />
              <span className="text-sm font-medium text-foreground">Pro analysis</span>
              <span className="text-[11px] text-muted-foreground">{pro.length} standalone analyses</span>
            </button>
            <button
              type="button"
              onClick={() => setMode("project")}
              className="flex flex-col gap-1 rounded-xl border border-border bg-background p-4 text-left transition-colors hover:border-brand-600/50 hover:bg-brand-600/5"
            >
              <Folder className="size-5 text-brand-600" />
              <span className="text-sm font-medium text-foreground">Projects</span>
              <span className="text-[11px] text-muted-foreground">{projects.length} projects</span>
            </button>
          </div>
        ) : (
          <>
            <div className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-2">
              <button
                type="button"
                onClick={() => {
                  setMode("root");
                  setProjectId(null);
                  setQuery("");
                }}
                className="rounded-lg border border-border px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
              >
                Back
              </button>
              <div className="flex min-w-0 flex-1 items-center gap-1.5 rounded-lg border border-border bg-background px-2">
                <SearchLg className="size-3.5 shrink-0 text-muted-foreground" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search analyses…"
                  className="h-8 w-full bg-transparent text-xs text-foreground outline-none placeholder:text-muted-foreground"
                />
              </div>
            </div>

            {mode === "pro" ? (
              <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto p-3">
                {filter(pro).length === 0 && (
                  <p className="p-3 text-xs text-muted-foreground">No pro analyses found.</p>
                )}
                {filter(pro).map((a) => (
                  <AnalysisRow key={a.id} a={a} />
                ))}
              </div>
            ) : (
              <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,180px)_minmax(0,1fr)] overflow-hidden">
                <div className="min-h-0 overflow-y-auto border-r border-border p-2">
                  {projects.length === 0 && (
                    <p className="p-2 text-xs text-muted-foreground">No projects.</p>
                  )}
                  {projects.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => setProjectId(p.id)}
                      className={cn(
                        "flex w-full items-center justify-between gap-1 rounded-lg px-2 py-1.5 text-left text-xs transition-colors",
                        projectId === p.id
                          ? "bg-brand-600/10 font-medium text-brand-600"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground",
                      )}
                    >
                      <span className="truncate">{p.name}</span>
                      <span className="text-[10px]">{p.count}</span>
                    </button>
                  ))}
                </div>
                <div className="flex min-h-0 flex-col gap-1 overflow-y-auto p-3">
                  {!projectId && <p className="p-2 text-xs text-muted-foreground">Select a project.</p>}
                  {projectId &&
                    filter(analyses.filter((a) => a.project_id === projectId)).map((a) => (
                      <AnalysisRow key={a.id} a={a} />
                    ))}
                  {projectId &&
                    filter(analyses.filter((a) => a.project_id === projectId)).length === 0 && (
                      <p className="p-2 text-xs text-muted-foreground">No analyses in this project.</p>
                    )}
                </div>
              </div>
            )}
          </>
        )}

        <div className="flex shrink-0 items-center justify-between gap-2 border-t border-border px-4 py-3">
          <span className="text-[11px] text-muted-foreground">{picked.length} selected</span>
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" className="h-8 rounded-lg text-xs" onClick={onClose}>
              Cancel
            </Button>
            <Button
              type="button"
              className="h-8 rounded-lg bg-brand-600 text-xs text-white hover:bg-brand-700"
              onClick={() => onConfirm(picked)}
            >
              Attach
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
