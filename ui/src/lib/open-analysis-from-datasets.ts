/**
 * Shared navigation helper used by the multi-file upload flow and by the
 * database "analyze selected tables" flow.
 *
 * It performs exactly what UploadModal does after a successful multi-file
 * upload: stash the session (session_id + thread_id + dataset chips) in
 * sessionStorage, persist analyses rows (one parent + one child per extra
 * dataset) and navigate to the Pro Analysis view with the multi-dataset
 * response in router state so every chat call carries the dataset_ids and the
 * X-Avaloka-Session header.
 */
import {
  createAnalysis as persistCreateAnalysis,
} from "@/lib/analysis-messages";

export type AnalysisDataset = {
  dataset_id: string;
  filename?: string | null;
  alias?: string | null;
  columns?: string[];
  rows?: unknown[];
  schema?: unknown;
  samples?: unknown;
  visualization_config?: unknown;
  visualization_status?: string;
  [k: string]: unknown;
};

const datasetLabel = (d: AnalysisDataset) =>
  d.alias || d.filename || d.dataset_id;

/**
 * Backends return per-dataset payloads either as `samples`/`schema` (upload
 * flow) or as `rows`/`columns` (database tables-to-analysis flow). Normalise to
 * the upload shape so each dataset keeps ITS OWN rows, columns and chart config
 * and no tab ever falls back to the primary dataset's data.
 */
function normalizeDataset(d: AnalysisDataset): AnalysisDataset {
  const samples = Array.isArray((d as any).samples)
    ? (d as any).samples
    : Array.isArray(d.rows)
      ? d.rows
      : undefined;
  const schema =
    d.schema ?? (Array.isArray(d.columns) ? d.columns : undefined) ?? null;
  return {
    ...d,
    samples,
    rows: samples,
    schema,
    columns: Array.isArray(d.columns) ? d.columns : undefined,
    rows_sampled: (d as any).rows_sampled ?? samples?.length,
  };
}

export async function openDatasetsInProAnalysis(opts: {
  navigate: (o: Record<string, unknown>) => void;
  sessionId: string | null;
  threadId: string | null;
  datasets: AnalysisDataset[];
  showAutoInsights?: boolean;
}): Promise<string | null> {
  const { navigate, sessionId, threadId } = opts;
  const datasets = (opts.datasets ?? []).map(normalizeDataset);
  const primary = datasets[0];
  if (!primary) return null;

  const primaryName = datasetLabel(primary);

  try {
    sessionStorage.setItem(
      "analysis:dataset",
      JSON.stringify({
        filename: primaryName,
        schema: primary.schema ?? primary.columns ?? null,
        samples: primary.samples ?? primary.rows ?? null,
        visualization_config: primary.visualization_config ?? null,
        visualization_status: primary.visualization_status ?? null,
      }),
    );
    sessionStorage.setItem(
      "analysis:session",
      JSON.stringify({
        threadId,
        sessionId,
        datasetChips: datasets.map((d) => ({
          id: d.dataset_id,
          name: datasetLabel(d),
          sessionId: sessionId ?? undefined,
          threadId: threadId ?? undefined,
          schema: d.schema ?? d.columns ?? null,
          samples: d.samples ?? d.rows ?? undefined,
          rowsSampled: (d as any).rows_sampled,
          visualizationConfig: d.visualization_config ?? null,
          visualizationStatus: d.visualization_status ?? null,
        })),
      }),
    );

  } catch {
    /* ignore */
  }

  let newAid: string | null = null;
  try {
    const created = await persistCreateAnalysis({
      project_id: null,
      name: primaryName.replace(/\.[^.]+$/, ""),
      dataset_id: primary.dataset_id,
      thread_id: threadId,
      session_id: sessionId,
      filename: primaryName,
      schema: (primary.schema ?? primary.columns ?? null) as unknown,
      samples: (primary.samples ?? null) as unknown,
      viz_config: (primary.visualization_config ?? null) as unknown,
    });
    newAid = created.id;

    const extras = datasets.filter(
      (d) => d.dataset_id && d.dataset_id !== primary.dataset_id,
    );
    if (newAid && extras.length) {
      await Promise.all(
        extras.map((d) =>
          persistCreateAnalysis({
            project_id: null,
            parent_analysis_id: newAid,
            name: datasetLabel(d).replace(/\.[^.]+$/, ""),
            dataset_id: d.dataset_id,
            thread_id: threadId,
            session_id: sessionId,
            filename: d.filename ?? d.alias ?? null,
            schema: (d.schema ?? d.columns ?? null) as unknown,
            samples: (d.samples ?? null) as unknown,
            viz_config: (d.visualization_config ?? null) as unknown,
          }).catch(() => null),
        ),
      );
    }
  } catch (err) {
    console.warn("Failed to persist analysis rows", err);
  }

  // Bind the workspace to THIS analysis so the Pro Analysis screen does not
  // reopen the previously visited project context.
  try {
    sessionStorage.setItem(
      "analysis:context",
      JSON.stringify({
        name: primaryName.replace(/\.[^.]+$/, ""),
        project: "",
        projectId: null,
        analysisId: newAid,
      }),
    );
  } catch {
    /* ignore */
  }



  navigate({
    to: "/analysis",
    search: (newAid ? { aid: newAid } : {}) as never,
    state: {
      uploadResponse: {
        ...primary,
        session_id: sessionId,
        thread_id: threadId,
        datasets,
      },
      filename: primaryName,
      newAnalysisId: newAid,
      standaloneAnalysis: true,
      showAutoInsights: opts.showAutoInsights ?? true,
    } as never,
  });

  return newAid;
}
