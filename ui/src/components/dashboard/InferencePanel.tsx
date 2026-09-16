import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";

import { backendApi } from "@/lib/api/backendApi";
import { useUploadedDatasets } from "@/lib/uploaded-datasets";

type FeatureSpec = { name: string; kind: "number" | "text"; options?: string[] };

interface Props {
  /** MLflow run id of the selected model. */
  runId: string;
  /** Raw model payload from GET /api/models/{run_id} — source of the feature schema. */
  raw?: Record<string, unknown>;
}

const NUMERIC_TYPE = /int|float|double|num|decimal|real|long/i;

const CATEGORICAL_TYPE = /str|obj|cat|bool|text|char/i;

/** Column names that are string-valued in practice even when no dtype is reported. */
const NAME_CATEGORICAL = /proximity|category|type|class|status|region|city|state|zone|label|gender|name/i;


function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function pickDatasetId(raw: Record<string, unknown> | undefined): string | null {
  if (!raw) return null;
  const sources = [
    raw,
    asRecord(raw.metadata),
    asRecord(raw.config),
    asRecord(raw.model_config),
    asRecord(raw.training_config),
    asRecord(raw.dataset),
    asRecord(asRecord(raw.config).metadata),
    asRecord(asRecord(raw.model_config).metadata),
  ];
  for (const source of sources) {
    const value = source.dataset_id ?? source.datasetId ?? source.training_dataset_id;
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

function toSpec(f: unknown): FeatureSpec | null {
  if (typeof f === "string") return { name: f, kind: "number" };
  if (f && typeof f === "object") {
    const r = f as Record<string, unknown>;
    const name = r.name ?? r.feature ?? r.column;
    if (!name) return null;
    const t = String(r.type ?? r.dtype ?? "");
    const rawOptions = r.categories ?? r.values ?? r.choices ?? r.enum ?? r.classes;
    const options = Array.isArray(rawOptions) ? rawOptions.map(String).filter(Boolean) : undefined;
    const categorical =
      !!options?.length || CATEGORICAL_TYPE.test(t) || (!!t && !NUMERIC_TYPE.test(t));
    return { name: String(name), kind: categorical ? "text" : "number", options };
  }
  return null;
}

/** Read the model's real feature columns out of the model metadata payload. */
function extractFeatures(raw: Record<string, unknown> | undefined): FeatureSpec[] {
  if (!raw) return [];
  const config = (raw.config ?? raw.model_config ?? {}) as Record<string, unknown>;
  const metrics = (raw.metrics ?? {}) as Record<string, unknown>;
  const schema = asRecord(raw.feature_schema ?? raw.input_schema ?? config.feature_schema ?? config.input_schema);
  const preprocessing = asRecord(raw.preprocessing ?? config.preprocessing);
  const typeMaps = [raw.feature_types, raw.dtypes, config.feature_types, config.dtypes, schema.properties];
  const candidates = [
    raw.feature_columns,
    raw.feature_names,
    raw.features,
    raw.feature_schema,
    raw.input_schema,
    config.feature_columns,
    config.feature_names,
    config.features,
    metrics.feature_columns,
  ];
  const categoricalHints = new Map<string, string[] | true>();
  for (const src of [
    raw.categorical_columns,
    config.categorical_columns,
    raw.categorical_features,
    config.categorical_features,
    raw.categories,
    config.categories,
    raw.category_mappings,
    config.category_mappings,
    preprocessing.categorical_columns,
    preprocessing.categorical_features,
    preprocessing.categories,
  ]) {
    if (Array.isArray(src)) src.forEach((n) => categoricalHints.set(String(n), true));
    else if (src && typeof src === "object")
      Object.entries(src as Record<string, unknown>).forEach(([k, v]) =>
        categoricalHints.set(k, Array.isArray(v) ? v.map(String) : true),
      );
  }
  const applyHints = (specs: FeatureSpec[]) =>
    specs.map((s) => {
      const hint = categoricalHints.get(s.name);
      const mappedType = typeMaps
        .map(asRecord)
        .map((types) => types[s.name])
        .find((value) => value !== undefined);
      const mappedSpec = mappedType === undefined ? null : toSpec({ name: s.name, ...(typeof mappedType === "object" ? asRecord(mappedType) : { type: mappedType }) });
      if (!hint && !mappedSpec) return NAME_CATEGORICAL.test(s.name) ? { ...s, kind: "text" as const } : s;
      return {
        ...s,
        kind: hint || mappedSpec?.kind === "text" ? ("text" as const) : s.kind,
        options: Array.isArray(hint) ? hint : mappedSpec?.options ?? s.options,
      };
    });

  for (const c of candidates) {
    if (Array.isArray(c) && c.length > 0) {
      const specs = c.map(toSpec).filter(Boolean) as FeatureSpec[];
      if (specs.length) return applyHints(specs);
    }
    if (c && typeof c === "object" && !Array.isArray(c)) {
      const specs = Object.entries(c as Record<string, unknown>)
        .map(([name, type]) => toSpec({ name, type }))
        .filter(Boolean) as FeatureSpec[];
      if (specs.length) return applyHints(specs);
    }
  }
  return [];
}

function formatPrediction(p: unknown): string {
  if (p === null || p === undefined) return "—";
  if (typeof p === "number") return Number.isInteger(p) ? String(p) : p.toFixed(4);
  if (Array.isArray(p)) return p.map(formatPrediction).join(", ");
  if (typeof p === "object") return JSON.stringify(p);
  return String(p);
}

export function InferencePanel({ runId, raw }: Props) {
  const schema = useMemo(() => {
    const specs = extractFeatures(raw);
    const seen = new Set<string>();
    return specs.filter((f) => {
      const name = (f.name ?? "").trim();
      if (!name || name.toLowerCase() === "undefined" || name.toLowerCase() === "null") return false;
      if (seen.has(name)) return false;
      seen.add(name);
      return true;
    });
  }, [raw]);

  const [fields, setFields] = useState<Record<string, string>>({});
  const [json, setJson] = useState("{\n  \n}");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ prediction: unknown; reply: string } | null>(null);

  const { data: datasets = [], isLoading: datasetsLoading } = useUploadedDatasets();
  const modelDatasetId = useMemo(() => pickDatasetId(raw), [raw]);
  const [datasetId, setDatasetId] = useState<string>("");

  useEffect(() => {
    if (datasetId && (datasetId === modelDatasetId || datasets.some((d) => d.dataset_id === datasetId))) return;
    const preferred =
      (modelDatasetId && datasets.some((d) => d.dataset_id === modelDatasetId)
        ? modelDatasetId
        : null) ??
      datasets[0]?.dataset_id ??
      "";
    setDatasetId(preferred);
  }, [datasets, modelDatasetId, datasetId]);

  const selectedDataset = datasets.find((d) => d.dataset_id === datasetId);

  useEffect(() => {
    setFields(Object.fromEntries(schema.map((f) => [f.name, ""])));
    setResult(null);
    setError(null);
  }, [schema]);

  const run = async () => {
    setError(null);
    if (!datasetId) {
      setError("Select a dataset to run inference under.");
      return;
    }
    let features: Record<string, unknown>;

    if (schema.length > 0) {
        const missing = schema.find((f) => (fields[f.name] ?? "").trim() === "");
        if (missing) {
          setError(`Enter a value for "${missing.name}".`);
          return;
        }
        features = Object.fromEntries(
          schema.map((f) => {
            const v = (fields[f.name] ?? "").trim();
            return [f.name, v];
          }),
        );
      }

    else {
      try {
        const parsed = JSON.parse(json);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          setError("Input must be a JSON object of feature values.");
          return;
        }
        features = parsed as Record<string, unknown>;
      } catch {
        setError("Invalid JSON — check for missing commas or quotes.");
        return;
      }
    }

    setLoading(true);
    setResult(null);
    try {
      const res = await backendApi.runInferenceViaThread(
        runId,
        features,
        datasetId || null,
        selectedDataset?.sessionId ?? null,
      );
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inference request failed.");
    } finally {
      setLoading(false);
    }
  };

  const modelName = raw ? (raw.name ?? raw.model_name ?? raw.run_name) : undefined;
  const modelVersion = raw ? (raw.version ?? raw.model_version ?? raw.latest_version) : undefined;
  const modelType = raw ? (raw.model_type ?? raw.task_type ?? raw.type) : undefined;
  const targetName = useMemo(() => {
    if (!raw) return "";
    const config = asRecord(raw.config ?? raw.model_config);
    const value =
      raw.target_column ?? raw.target ?? raw.label_column ?? config.target_column ?? config.target;
    return typeof value === "string" ? value : "";
  }, [raw]);


  return (
    <div className="rounded-2xl border border-border bg-white p-6 shadow-sm">
      <h2 className="text-base font-semibold text-foreground">Inference</h2>
      <p className="mt-1 text-xs text-muted-foreground">
        Runs against the live model service. The first request can take several seconds while the
        model is downloaded.
      </p>

      <div className="mt-4">
        <label className="text-xs text-muted-foreground" htmlFor="inf-dataset">
          Run under dataset / session
        </label>
        <select
          id="inf-dataset"
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
          className="mt-1.5 w-full max-w-sm rounded-xl border border-border bg-muted/40 px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
        >
          <option value="">{datasetsLoading ? "Loading datasets…" : "Select a dataset"}</option>
          {datasets.map((d) => (
            <option key={d.dataset_id} value={d.dataset_id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>

      {schema.length > 0 ? (
        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {schema.map((f) => (
            <div key={f.name}>
              <label className="text-xs text-muted-foreground" htmlFor={`inf-${f.name}`}>
                {f.name}
              </label>
              {f.options && f.options.length > 0 ? (
                <select
                  id={`inf-${f.name}`}
                  value={fields[f.name] ?? ""}
                  onChange={(e) => setFields((p) => ({ ...p, [f.name]: e.target.value }))}
                  className="mt-1.5 w-full rounded-xl border border-border bg-muted/40 px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
                >
                  <option value="">Select…</option>
                  {f.options.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              ) : (
                
                <input
                  id={`inf-${f.name}`}
                  type="text"
                  value={fields[f.name] ?? ""}
                  onChange={(e) => setFields((p) => ({ ...p, [f.name]: e.target.value }))}
                  className="mt-1.5 w-full rounded-xl border border-border bg-muted/40 px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
                />

              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-5">
          <label className="text-xs text-muted-foreground" htmlFor="inf-json">
            Input payload (JSON) — no feature schema was returned for this model
          </label>
          <textarea
            id="inf-json"
            value={json}
            onChange={(e) => setJson(e.target.value)}
            spellCheck={false}
            rows={10}
            className="mt-1.5 w-full rounded-xl border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
          />
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium text-red-600">
          {error}
        </div>
      )}

      <div className="mt-5 flex items-center gap-3">
        <button
          type="button"
          onClick={run}
          disabled={loading}
          style={{ backgroundColor: "#1565ef", color: "#ffffff" }}
          className="inline-flex h-10 min-w-44 shrink-0 items-center justify-center gap-2 rounded-lg px-5 text-sm font-semibold shadow-sm transition hover:opacity-90 disabled:opacity-60"
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          {loading ? "Running inference…" : "Run Inference"}
        </button>
      </div>


      {result && !loading && (
        <div className="mt-6 rounded-xl border border-border bg-muted/40 p-5">
          <div className="text-xs text-muted-foreground">Prediction</div>
          {result.prediction === null || result.prediction === undefined ? (
            <div className="mt-1 text-sm text-muted-foreground">
              The model responded, but no numeric prediction could be parsed. See the full response
              below.
            </div>
          ) : (
            <div className="mt-1 text-3xl font-bold text-foreground">
              {targetName ? `${targetName} = ` : ""}
              {formatPrediction(result.prediction)}
            </div>
          )}

          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
            {[
              { label: "Model", value: modelName },
              { label: "Version", value: modelVersion },
              { label: "Type", value: modelType },
            ]
              .filter((t) => t.value !== undefined && t.value !== null && t.value !== "")
              .map((t) => (
                <div key={t.label} className="rounded-lg border border-border bg-white p-3">
                  <div className="text-xs text-muted-foreground">{t.label}</div>
                  <div className="mt-0.5 text-sm font-semibold text-foreground">{String(t.value)}</div>
                </div>
              ))}
          </div>

          {result.reply && (
            <div className="mt-5">
              <div className="text-xs text-muted-foreground">Assistant response</div>
              <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-white p-3 text-xs text-foreground">
                {result.reply}
              </pre>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
