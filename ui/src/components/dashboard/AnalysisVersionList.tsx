import { useCallback, useEffect, useState } from "react";
import { RefreshCcw01, Loading01 } from "@untitledui/icons";
import { toast } from "sonner";
import { cx } from "@/lib/utils/cx";
import { Tooltip } from "@/components/base/tooltip/tooltip";
import {
  listAnalysisVersions,
  restoreAnalysisVersion,
  emitVersionRestored,
  versionDate,
  formatAbsolute,
  formatRelative,
  type AnalysisVersion,
} from "@/lib/api/analysis-versions";

const truncate = (s: string, n = 60) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex h-4 items-center rounded-full bg-secondary px-1.5 text-[10px] font-medium text-tertiary ring-1 ring-inset ring-secondary">
      {children}
    </span>
  );
}

export function AnalysisVersionList({ analysisId }: { analysisId: string }) {
  const [versions, setVersions] = useState<AnalysisVersion[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restoringTs, setRestoringTs] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setVersions(await listAnalysisVersions(analysisId));
    } catch (err: any) {
      setError(err?.message || "Failed to load versions");
    } finally {
      setLoading(false);
    }
  }, [analysisId]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleRestore = async (promptTs: string) => {
    setRestoringTs(promptTs);
    try {
      const result = await restoreAnalysisVersion(analysisId, promptTs);
      setVersions(
        (prev) => prev?.map((v) => ({ ...v, is_current: v.prompt_ts === (result.restored_prompt_ts ?? promptTs) })) ?? prev,
      );
      emitVersionRestored({ analysisId, promptTs, result });
      toast.success("Version restored");
      void load();
    } catch (err: any) {
      toast.error(err?.message || "Restore failed");
    } finally {
      setRestoringTs(null);
    }
  };

  return (
    <div className="ml-6 flex flex-col gap-0.5 border-l border-secondary pl-2">
      {loading && versions === null ? (
        <div className="flex flex-col gap-1 py-1">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-9 animate-pulse rounded-md bg-secondary" />
          ))}
        </div>
      ) : error ? (
        <div className="flex flex-col items-start gap-1 px-2 py-2">
          <p className="text-xs text-error-primary">{error}</p>
          <button
            type="button"
            onClick={() => void load()}
            className="inline-flex items-center gap-1 text-xs font-semibold text-brand-secondary hover:underline"
          >
            <RefreshCcw01 className="size-3" /> Retry
          </button>
        </div>
      ) : !versions || versions.length === 0 ? (
        <p className="px-2 py-2 text-xs text-tertiary">No saved versions yet.</p>
      ) : (
        versions.map((v) => {
          const d = versionDate(v);
          const label = v.prompt?.trim() ? truncate(v.prompt.trim()) : "Re-run";
          const isRestoring = restoringTs === v.prompt_ts;
          return (
            <div
              key={v.prompt_ts}
              className={cx(
                "flex min-h-9 flex-col gap-0.5 rounded-md px-2 py-1.5 hover:bg-primary_hover",
                v.is_current && "bg-brand-primary_alt",
              )}
            >
              <div className="flex items-center gap-1.5">
                <Tooltip title={v.prompt ?? "Re-run"} placement="top" delay={150}>
                  <span className="min-w-0 flex-1 truncate text-xs font-medium text-secondary">{label}</span>
                </Tooltip>
                {v.is_current ? (
                  <span className="shrink-0 rounded-full bg-brand-solid px-1.5 text-[10px] font-semibold text-white">
                    Current
                  </span>
                ) : (
                  <button
                    type="button"
                    disabled={isRestoring}
                    onClick={() => void handleRestore(v.prompt_ts)}
                    className="shrink-0 inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-semibold text-brand-secondary hover:bg-primary_hover disabled:opacity-60"
                  >
                    {isRestoring ? <Loading01 className="size-3 animate-spin" /> : null}
                    Restore
                  </button>
                )}
              </div>
              <div className="flex items-center gap-1">
                <Tooltip title={formatAbsolute(d, v.prompt_ts)} placement="top" delay={150}>
                  <span className="text-[10px] text-tertiary">{formatRelative(d) || v.prompt_ts}</span>
                </Tooltip>
                {v.has_code ? <Badge>Code</Badge> : null}
                {v.has_output ? <Badge>Output</Badge> : null}
                {v.has_viz ? <Badge>Chart</Badge> : null}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
