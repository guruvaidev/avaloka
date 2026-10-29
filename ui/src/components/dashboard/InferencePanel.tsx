import { useEffect, useMemo, useState } from "react";
import { Loader2, Play } from "lucide-react";

import { Button } from "@/components/ui/button";
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

function parseInputs(value: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Input must be a JSON object of feature values.");
  }
  return parsed as Record<string, unknown>;
}

function probabilityEntries(value: unknown): Array<[string, unknown]> {
  if (Array.isArray(value)) return value.map((v, i) => [String(i), v]);
  if (value && typeof value === "object") return Object.entries(value);
  return [];
}

const SAMPLES = [
  { label: "Sample 1 – 1st class female, 29", values: { Pclass: 1, Sex: "female", Age: 29, SibSp: 0, Parch: 0, Fare: 211.34, Embarked: "S" } },
  { label: "Sample 2 – 3rd class girl, 12", values: { Pclass: 3, Sex: "female", Age: 12, SibSp: 0, Parch: 0, Fare: 7.85, Embarked: "S" } },
] as const;

function formPayload(fields: Record<string, string>, schema: FeatureSpec[]): Record<string, unknown> {
  return Object.fromEntries(schema.flatMap((feature) => {
    const value = fields[feature.name]?.trim();
    if (!value) return [];
    const numeric = feature.kind === "number" ? Number(value) : NaN;
    return [[feature.name, Number.isFinite(numeric) ? numeric : value]];
  }));
}

function predictedClass(prediction: unknown, reply: string): unknown {
  if (prediction !== null && prediction !== undefined) return prediction;
  const match = reply.match(/(?:predicted\s+class|prediction|predicted\s+label)\s*(?:is|:|=)\s*["']?([^\n.,"']+)/i);
  return match?.[1]?.trim() || null;
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
  const [json, setJson] = useState("");
  const [runHovered, setRunHovered] = useState(false);
  const [mode, setMode] = useState<"form" | "json">("form");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [result, setResult] = useState<Awaited<ReturnType<typeof backendApi.runInferenceViaThread>> | null>(null);
  const activeMode = schema.length === 0 ? "json" : mode;

  useEffect(() => {
    try {
      if (window.sessionStorage.getItem("inference-input-mode") === "json") setMode("json");
    } catch { /* session storage may be unavailable */ }
  }, []);

  useEffect(() => {
    if (!loading) {
      setStarting(false);
      return;
    }
    const timer = window.setTimeout(() => setStarting(true), 3000);
    return () => window.clearTimeout(timer);
  }, [loading]);

  const { data: datasets = [] } = useUploadedDatasets();
  const modelDatasetId = useMemo(() => pickDatasetId(raw), [raw]);
  const trainingDataset = useMemo(
    () => datasets.find((d) => d.dataset_id === modelDatasetId),
    [datasets, modelDatasetId],
  );
  const targetName = useMemo(() => {
    if (!raw) return "";
    const config = asRecord(raw.config ?? raw.model_config);
    const value = raw.target_column ?? raw.target ?? raw.label_column ?? config.target_column ?? config.target;
    return typeof value === "string" ? value : "";
  }, [raw]);
  useEffect(() => {
    setFields(Object.fromEntries(schema.map((f) => [f.name, ""])));
    setResult(null);
    setError(null);
    setRequestError(null);
    setJsonError(null);
    setJson("");
  }, [schema]);

  useEffect(() => {
    if (activeMode === "json") {
      setJson((current) => current || JSON.stringify(SAMPLES[0].values, null, 2));
    }
  }, [activeMode]);

  const switchMode = (next: "form" | "json") => {
    if (next === activeMode) return;
    if (next === "json") {
      const values = formPayload(fields, schema);
      setJson(JSON.stringify(Object.keys(values).length ? values : SAMPLES[0].values, null, 2));
      setJsonError(null);
    } else {
      try {
        const parsed = parseInputs(json);
        setFields(Object.fromEntries(schema.map((f) => [f.name, parsed[f.name] == null ? "" : String(parsed[f.name])])));
        setJsonError(null);
      } catch (e) {
        setJsonError(`Invalid JSON: ${e instanceof Error ? e.message : "Check your input."}`);
        return;
      }
    }
    setError(null);
    setMode(next);
    try { window.sessionStorage.setItem("inference-input-mode", next); } catch { /* unavailable */ }
  };

  const run = async () => {
    setError(null);
    setJsonError(null);
    setRequestError(null);
    let features: Record<string, unknown>;
    if (activeMode === "json") {
      try {
        features = parseInputs(json);
      } catch (e) {
        setJsonError(`Invalid JSON: ${e instanceof Error ? e.message : "Check your input."}`);
        return;
      }
    } else {
      const missing = schema.find((f) => (fields[f.name] ?? "").trim() === "");
      if (missing) {
        setError(`Enter a value for "${missing.name}".`);
        return;
      }
      features = Object.fromEntries(schema.map((f) => [f.name, (fields[f.name] ?? "").trim()]));
    }
    if (!modelDatasetId) {
      setError("This model has no linked training dataset, so inference can't run.");
      return;
    }
    console.debug("[Inference] resolved dataset/session", {
      datasetId: modelDatasetId,
      sessionId: trainingDataset?.sessionId ?? null,
    });
    setLoading(true);
    setResult(null);
    try {
      const res = await backendApi.runInferenceViaThread(
        runId,
        features,
        modelDatasetId || null,
        trainingDataset?.sessionId ?? null,
      );
      setResult(res);
    } catch (e) {
      setRequestError(e instanceof Error ? e.message : "Inference request failed.");
    } finally {
      setLoading(false);
    }
  };

  const displayedPrediction = result ? predictedClass(result.prediction, result.reply) : null;
  const modelName = raw ? (raw.name ?? raw.model_name ?? raw.run_name) : undefined;
  const modelVersion = raw ? (raw.version ?? raw.model_version ?? raw.latest_version) : undefined;
  const modelType = raw ? (raw.model_type ?? raw.task_type ?? raw.type) : undefined;
  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-foreground">Inference</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            The model service starts automatically on your first run. The first request may take several seconds while the model loads.
          </p>
        </div>
        <div className="inline-flex shrink-0 rounded-lg border border-border bg-muted/40 p-0.5" role="group" aria-label="Inference input mode">
          <Button type="button" size="sm" variant="ghost" onClick={() => switchMode("form")} aria-pressed={activeMode === "form"} disabled={schema.length === 0} className="min-w-20 shadow-none disabled:opacity-100" style={{ backgroundColor: activeMode === "form" ? "var(--primary)" : "transparent", color: activeMode === "form" ? "var(--primary-foreground)" : "var(--foreground)" }}>Form</Button>
          <Button type="button" size="sm" variant="ghost" onClick={() => switchMode("json")} aria-pressed={activeMode === "json"} className="min-w-20 shadow-none" style={{ backgroundColor: activeMode === "json" ? "var(--primary)" : "transparent", color: activeMode === "json" ? "var(--primary-foreground)" : "var(--foreground)" }}>JSON</Button>
        </div>
      </div>


      {activeMode === "form" ? (
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
          <div className="flex flex-wrap items-center justify-between gap-2">
            <label className="text-xs text-muted-foreground" htmlFor="inf-json">
              Input payload (JSON){schema.length === 0 ? " — no feature schema was returned for this model" : ""}
            </label>
            <select aria-label="Load sample" value="" onChange={(e) => {
              const sample = SAMPLES[Number(e.target.value)];
              if (sample) { setJson(JSON.stringify(sample.values, null, 2)); setJsonError(null); }
            }} className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground">
              <option value="" disabled>Load sample</option>
              {SAMPLES.map((sample, index) => <option key={index} value={index}>{sample.label}</option>)}
            </select>
          </div>
          <textarea
            id="inf-json"
            value={json}
            onChange={(e) => { setJson(e.target.value); setJsonError(null); }}
            spellCheck={false}
            rows={12}
            className="mt-1.5 w-full rounded-xl border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
          />
          {jsonError && <p className="mt-2 text-xs text-destructive" role="alert">{jsonError}</p>}
        </div>
      )}

      {error && (
        <p className="mt-4 text-xs font-medium text-destructive" role="alert">{error}</p>
      )}

      <div className="mt-5 flex items-center gap-3">
        <Button
          type="button"
          variant="ghost"
          onClick={run}
          onMouseEnter={() => setRunHovered(true)}
          onMouseLeave={() => setRunHovered(false)}
          disabled={loading}
          className="h-10 min-w-44"
          style={{ backgroundColor: runHovered ? "color-mix(in oklch, var(--primary) 85%, black)" : "var(--primary)", color: "var(--primary-foreground)" }}
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {loading ? "Running…" : "Run Inference"}
        </Button>
      </div>
      {loading && starting && <p className="mt-2 text-xs text-muted-foreground" role="status">Starting model service…</p>}


      {(result || requestError) && !loading && (
        <div className="mt-6 rounded-xl border border-border bg-muted/40 p-5">
          {requestError && <div role="alert" className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">{requestError}</div>}
          {result && <>
          <div className="text-xs text-muted-foreground">Prediction</div>
          {displayedPrediction === null ? (
            <div className="mt-1 text-sm text-muted-foreground">
               The model responded, but no prediction could be parsed. See the full response
              below.
            </div>
          ) : (
            <div className="mt-1 text-3xl font-bold text-foreground">
              {targetName ? `${targetName} = ` : ""}
               {formatPrediction(displayedPrediction)}
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
                <div key={t.label} className="rounded-lg border border-border bg-muted p-3">
                  <div className="text-xs text-muted-foreground">{t.label}</div>
                  <div className="mt-0.5 text-sm font-semibold text-foreground">{String(t.value)}</div>
                </div>
              ))}
          </div>

           {probabilityEntries(result.probabilities).length > 0 && (
             <div className="mt-5">
               <div className="text-xs text-muted-foreground">Class probabilities</div>
               <div className="mt-2 grid gap-2 sm:grid-cols-2">
                 {probabilityEntries(result.probabilities).map(([label, value]) => (
                   <div key={label} className="flex justify-between gap-3 rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground">
                     <span>{label}</span><span className="font-semibold">{typeof value === "number" ? `${(value * 100).toFixed(2)}%` : String(value)}</span>
                   </div>
                 ))}
               </div>
             </div>
           )}

           {result.reply && (
            <div className="mt-5">
              <div className="text-xs text-muted-foreground">Assistant response</div>
              <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-card p-3 text-xs text-foreground">
                {result.reply}
              </pre>
            </div>
          )}
           <details className="mt-5 border-t border-border pt-4">
             <summary className="cursor-pointer text-sm font-medium text-foreground">Raw response</summary>
             <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-background p-3 font-mono text-xs text-foreground">{JSON.stringify(result.rawResponse, null, 2)}</pre>
           </details>
           </>}
        </div>
      )}

    </div>
  );
}
