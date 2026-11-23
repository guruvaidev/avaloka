import { useEffect, useMemo, useRef, useState } from "react";
import { Download, Loader2, Play, Plus, Upload, X } from "lucide-react";
import Papa from "papaparse";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { HOUSING_EXAMPLE_ROW, type Feature, type InputRow } from "@/api/inference";

const fmtRange = (n: number, name: string) => n.toLocaleString("en-US", { minimumFractionDigits: name === "median_income" ? 1 : 0, maximumFractionDigits: 6 });
import { download, parseCsv, parseJson, rowWarnings } from "@/lib/inference";

type Mode = "Form" | "JSON" | "CSV upload";
export interface InputState { mode: Mode; fields: Record<string, string>; json: string; csvRows: InputRow[]; filename: string; batch: InputRow[] }
export const emptyInput = (): InputState => ({ mode: "Form", fields: {}, json: "{}", csvRows: [], filename: "", batch: [] });

interface Props {
  features: Feature[];
  state: InputState;
  onChange: (value: InputState) => void;
  running: boolean;
  progress: { done: number; total: number };
  stepIndicator?: string;
  onRun: (rows: InputRow[], single: boolean, chunked: boolean) => void;
}

export function InferenceInput({ features, state, onChange, running, progress, stepIndicator, onRun }: Props) {
  const [error, setError] = useState("");
  const [attempted, setAttempted] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const noSchema = features.length === 0;
  const canUseExample = features.length > 0 && features.every((f) => f.name in HOUSING_EXAMPLE_ROW);
  // Models without a saved feature schema open on JSON input.
  useEffect(() => { if (noSchema && state.mode === "Form") onChange({ ...state, mode: "JSON" }); }, [noSchema]); // eslint-disable-line react-hooks/exhaustive-deps

  const row: InputRow = useMemo(() => Object.fromEntries(features.map((f) => {
    const raw = state.fields[f.name] ?? "";
    return [f.name, f.numeric && raw.trim() !== "" && Number.isFinite(Number(raw)) ? Number(raw) : raw];
  })), [features, state.fields]);
  const validation = rowWarnings(row, features);

  const json = useMemo(() => {
    try {
      const rows = parseJson(state.json, features);
      return { rows, single: !state.json.trim().startsWith("["), message: `Valid JSON · ${rows.length.toLocaleString("en-US")} ${rows.length === 1 ? "row" : "rows"}${features.length ? " · all features present" : ""}` };
    } catch (e) { return { rows: [] as InputRow[], single: true, message: e instanceof Error ? e.message : "Invalid JSON" }; }
  }, [state.json, features]);

  async function upload(file?: File) {
    if (!file) return;
    if (!/\.csv$/i.test(file.name)) { setError("Choose a CSV file"); return; }
    try { const rows = parseCsv(await file.text(), features); onChange({ ...state, csvRows: rows, filename: file.name }); setError(""); }
    catch (e) { onChange({ ...state, csvRows: [], filename: "" }); setError(e instanceof Error ? e.message : "The CSV couldn't be read"); }
  }

  function submit(add = false) {
    setAttempted(true); setError("");
    if (state.mode === "Form") {
      if (!features.length) { setError("This model has no saved feature schema. Enter rows as JSON."); return; }
      if (validation.required.length) return;
      if (add) {
        if (state.batch.length >= 10000) { setError("For more than 10,000 rows, split the file."); return; }
        onChange({ ...state, batch: [...state.batch, row] });
        return;
      }
      if (state.batch.length) onRun(state.batch, false, true); else onRun([row], true, false);
    } else if (state.mode === "JSON") {
      if (!json.rows.length) { setError(json.message); return; }
      onRun(json.rows, json.single, true);
    } else {
      if (!state.csvRows.length) { setError("Add at least one row"); return; }
      onRun(state.csvRows, false, true);
    }
  }

  const modes: Mode[] = ["Form", "JSON", "CSV upload"];
  return (
    <section className="inf-card" aria-label="Input">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="font-semibold">Input</h2>
        <div className="flex rounded-lg bg-inf-tile p-1" role="group" aria-label="Input mode">
          {modes.map((mode) => (
            <Button key={mode} variant={state.mode === mode ? "default" : "ghost"} className="min-h-11 px-3" aria-pressed={state.mode === mode} disabled={running} onClick={() => { onChange({ ...state, mode }); setError(""); }}>{mode}</Button>
          ))}
        </div>
      </div>

      <fieldset disabled={running} className="mt-5 min-w-0">
        <legend className="sr-only">{state.mode} input</legend>
        {state.mode === "Form" && (features.length ? (
          <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2">
            {features.map((f) => {
              const id = `feature-${f.name}`;
              const messages = (validation.fields[f.name] ?? []).filter((m) => m !== "Required" || attempted);
              const rangeText = f.min !== undefined && f.max !== undefined ? `${fmtRange(f.min, f.name)} to ${fmtRange(f.max, f.name)}` : undefined;
              return (
                <div key={f.name}>
                  <label htmlFor={id} className="mb-1.5 block break-all text-xs font-medium">{f.name}</label>
                  {f.categories?.length ? (
                    <Select value={state.fields[f.name] ?? ""} onValueChange={(v) => onChange({ ...state, fields: { ...state.fields, [f.name]: v } })}>
                      <SelectTrigger id={id} className="min-h-11 bg-inf-input" aria-describedby={`help-${f.name}`}><SelectValue placeholder="Select a value" /></SelectTrigger>
                      <SelectContent>{f.categories.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                    </Select>
                  ) : (
                    <Input id={id} type={f.numeric ? "number" : "text"} step="any" inputMode={f.numeric ? "decimal" : undefined} className="min-h-11 bg-inf-input" value={state.fields[f.name] ?? ""}
                      aria-invalid={attempted && validation.required.includes(f.name)} aria-describedby={`help-${f.name}`}
                      placeholder={rangeText ?? "Enter a value"}
                      onChange={(e) => onChange({ ...state, fields: { ...state.fields, [f.name]: e.target.value } })} />
                  )}
                  <div id={`help-${f.name}`} className="mt-1.5 text-xs">
                    {rangeText && <p className="text-muted-foreground">Training range {rangeText}</p>}
                    {messages.map((m) => <p key={m} className="inf-warning-text mt-1">{m}</p>)}
                  </div>
                </div>
              );
            })}
          </div>
        ) : <p className="text-sm text-muted-foreground">This model has no saved feature schema. Enter rows as JSON.</p>)}

        {state.mode === "JSON" && (
          <div>
            {noSchema && <p className="mb-3 text-sm text-muted-foreground">This model has no saved feature schema. Enter rows as JSON.</p>}
            <label htmlFor="inference-json" className="mb-2 block text-xs font-medium">Input rows (one object or an array)</label>
            <Textarea id="inference-json" value={state.json} spellCheck={false} className="min-h-64 bg-inf-input font-mono text-sm" aria-describedby="json-status" onChange={(e) => onChange({ ...state, json: e.target.value })} />
            <p id="json-status" className={`mt-2 text-xs ${json.rows.length ? "inf-success-text" : "inf-warning-text"}`}>{json.message}</p>
          </div>
        )}

        {state.mode === "CSV upload" && (
          <div>
            <div className={`inf-drop ${dragging ? "border-inf-blue bg-inf-active" : "border-input"}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); if (!running) void upload(e.dataTransfer.files[0]); }}>
              <Upload className="mx-auto size-7 text-muted-foreground" aria-hidden="true" />
              <label htmlFor="inference-csv" className="mt-3 block text-sm font-medium">{state.filename || "Drop a CSV file here"}</label>
              <input ref={fileInput} id="inference-csv" type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = ""; }} />
              <Button variant="outline" className="mt-3 min-h-11" onClick={() => fileInput.current?.click()}><Upload />Choose CSV</Button>
              {state.csvRows.length > 0 && <p className="mt-2 text-xs text-muted-foreground">{state.csvRows.length.toLocaleString("en-US")} rows ready</p>}
            </div>
            <Button variant="link" className="mt-2 min-h-11 px-0" disabled={!features.length} onClick={() => download("inference-template.csv", Papa.unparse([features.map((f) => f.name)]), "text/csv")}><Download />Download CSV template</Button>
          </div>
        )}
      </fieldset>

      {error && <p className="inf-warning mt-3" role="alert">{error}</p>}
      {running && progress.total > 500 && (
        <div className="mt-4">
          <p className="mb-2 text-xs">Predicting {progress.done.toLocaleString("en-US")} of {progress.total.toLocaleString("en-US")} rows</p>
          <Progress value={(progress.done / progress.total) * 100} aria-label="Prediction progress" />
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-2">
        <Button disabled={running} className="min-h-11" onClick={() => submit()}>{running ? <Loader2 className="animate-spin motion-reduce:animate-none" /> : <Play />}{running ? "Running…" : "Run inference"}</Button>
        {state.mode === "Form" && <Button variant="outline" className="min-h-11" disabled={running} onClick={() => submit(true)}><Plus />Add to batch</Button>}
        {state.mode === "Form" && state.batch.length > 0 && (
          <span className="inf-badge inf-status-starting">Batch: {state.batch.length.toLocaleString("en-US")} {state.batch.length === 1 ? "row" : "rows"}
            <button type="button" className="ml-1 inline-flex size-5 items-center justify-center" aria-label="Clear batch" onClick={() => onChange({ ...state, batch: [] })}><X className="size-3" /></button>
          </span>
        )}
        <Button variant="ghost" className="min-h-11" disabled={running} onClick={() => { onChange({ ...emptyInput(), mode: state.mode }); setAttempted(false); setError(""); }}>Clear</Button>
        {state.mode === "Form" && canUseExample && (
          <Button variant="ghost" className="min-h-11" disabled={running} onClick={() => { onChange({ ...state, fields: { ...HOUSING_EXAMPLE_ROW } }); setError(""); }}>Use example row</Button>
        )}
      </div>
      {stepIndicator && <p role="status" className="mt-3 flex items-center gap-2 text-xs text-muted-foreground"><Loader2 className="size-4 animate-spin motion-reduce:animate-none" />{stepIndicator}</p>}
    </section>
  );
}
