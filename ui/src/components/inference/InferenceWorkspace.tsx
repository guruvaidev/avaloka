import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Code2, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { asRecord, inferenceApi, InferenceApiError, modelQuery, modelsQuery, normalizeResult, type InferenceModel, type InputRow } from "@/api/inference";
import { chunksOf, keepRecent, recentSummary, rowWarnings, safeErrorDetails, type InferenceRun } from "@/lib/inference";
import { BACKEND_API_BASE } from "@/lib/api/backend-config";
import { ModelSummary, ServiceBadge, copyText } from "./ModelSummary";
import { InferenceInput, emptyInput, type InputState } from "./InferenceInput";
import { InferenceResults } from "./InferenceResults";
import { ModelsList } from "./ModelsList";
import { SectionBoundary } from "./SectionBoundary";
import { PLAN_CHECK_UNAVAILABLE_RETRY } from "@/lib/plan-check";

// Session memory only: inputs and results are never written to browser storage.
let sessionRuns: InferenceRun[] = [];

type Upgrade = { message: string; plans: string[] };

export function InferenceWorkspace({ runId }: { runId: string }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const list = useQuery(modelsQuery());
  const detail = useQuery(modelQuery(runId));
  const models = list.data ?? [];
  const listEntry = useMemo(() => models.find((m) => m.runId === runId), [models, runId]);
  const model: InferenceModel | undefined = detail.data ?? listEntry;
  const features = detail.data?.features.length ? detail.data.features : listEntry?.features ?? [];

  const [input, setInput] = useState<InputState>(emptyInput);
  const [run, setRun] = useState<InferenceRun | null>(null);
  const [recent, setRecent] = useState(sessionRuns);
  const [selected, setSelected] = useState(0);
  const [apiOpen, setApiOpen] = useState(false);
  const [stopOpen, setStopOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [errorDetails, setErrorDetails] = useState<ReturnType<typeof safeErrorDetails> | null>(null);
  const [step, setStep] = useState<{ label: string; started: number } | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const operationStarted = useRef(0);
  const [upgrade, setUpgrade] = useState<Upgrade | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [autoStarting, setAutoStarting] = useState(false);
  const activeRunId = useRef(runId);
  activeRunId.current = runId;
  const pendingOpen = useRef<InferenceRun | null>(null);

  useEffect(() => {
    setError(""); setErrorDetails(null); setStep(null); setUpgrade(null); setNotice(""); setSelected(0); setAutoStarting(false); setProgress({ done: 0, total: 0 });
    const opened = pendingOpen.current;
    if (opened && opened.model.runId === runId) {
      setInput({ ...emptyInput(), mode: "JSON", json: JSON.stringify(opened.rows, null, 2) });
      setRun(opened);
      pendingOpen.current = null;
    } else { setInput(emptyInput()); setRun(null); }
  }, [runId]);
  useEffect(() => {
    if (!step) { setElapsed(0); return; }
    const update = () => setElapsed(Math.floor((Date.now() - step.started) / 1000));
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [step]);
  function beginStep(label: string, id: string) {
    if (activeRunId.current === id) { setElapsed(0); setStep({ label, started: Date.now() }); }
  }

  // Auto-select the newest usable run when no valid ?run_id= is present.
  useEffect(() => {
    if (!list.isSuccess) return;
    const usable = models.filter((m) => m.usable);
    if (runId && usable.some((m) => m.runId === runId)) return;
    if (runId && detail.isSuccess && detail.data?.runId) return;
    const newest = usable[0]?.runId;
    if (newest && newest !== runId) void navigate({ to: "/inference", search: { run_id: newest }, replace: true });
  }, [list.isSuccess, models, runId, detail.isSuccess, detail.data, navigate]);

  function handleError(e: unknown, id: string) {
    if (activeRunId.current !== id) return;
    setNotice(""); setAutoStarting(false);
    setStep(null); setErrorDetails(safeErrorDetails(e));
    const apiError = e instanceof InferenceApiError ? e : null;
    const d = asRecord(apiError?.detail);
    const status = apiError?.status;
    if (status === 400) {
      const raw = apiError?.detail;
      setError(typeof raw === "string" ? raw : typeof d.message === "string" ? d.message : Array.isArray(raw) ? raw.map((item) => String(asRecord(item).msg ?? "Invalid input")).join("; ") : "Check the input values and try again.");
    } else if (status === 403 && d.reason === "plan_upgrade_required") {
      setUpgrade({ message: typeof d.message === "string" ? d.message : "Inference requires an upgraded plan.", plans: Array.isArray(d.allowed_plans) ? d.allowed_plans.map(String) : [] });
    } else if (status === 503 && d.reason === "plan_check_unavailable") {
      // The plan could not be checked: say so, and offer no upgrade.
      setError(typeof d.message === "string" ? d.message : PLAN_CHECK_UNAVAILABLE_RETRY);
    } else if (status === 403) setError("You don't have access to this model.");
    else if (status === 404) { setError("This model no longer exists."); void queryClient.invalidateQueries({ queryKey: modelsQuery().queryKey }); }
    else if (status === 401) void navigate({ to: "/" });
    else setError("The prediction couldn't be completed. Try again, or check the Ray dashboard.");
  }

  async function refresh(id: string) {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: modelQuery(id).queryKey }),
      queryClient.invalidateQueries({ queryKey: modelsQuery().queryKey }),
    ]);
  }

  const service = useMutation({
    mutationFn: async ({ id, start }: { id: string; start: boolean }) => { operationStarted.current = Date.now(); if (start) { beginStep("Starting model service…", id); await inferenceApi.startService(id); } else await inferenceApi.stopService(id); return id; },
    onSuccess: (id) => { void refresh(id); if (activeRunId.current === id) { setError(""); setErrorDetails(null); setStep(null); setNotice(""); } },
    onError: (e, vars) => handleError(e, vars.id),
  });

  const prediction = useMutation({
    mutationFn: async ({ rows, single, chunked, snapshot, featureSet }: { rows: InputRow[]; single: boolean; chunked: boolean; snapshot: InferenceModel; featureSet: typeof features }) => {
      const id = snapshot.runId ?? "";
      const start = performance.now();
      operationStarted.current = Date.now();
      beginStep("Running prediction…", id);
      const predictions: InferenceRun["predictions"] = [];
      let configured = false;
      for (const chunk of chunked ? chunksOf(rows) : [rows]) {
        const body = single ? chunk[0] : chunk;
        let response: { result?: unknown };
        try { response = await inferenceApi.predict(id, body); }
        catch (e) {
          if (!(e instanceof InferenceApiError) || e.status !== 409 || configured) throw e;
          configured = true;
          if (activeRunId.current === id) { setNotice("Starting the model service. The first run can take a few seconds."); setAutoStarting(true); }
          beginStep("Starting model service…", id);
          await inferenceApi.startService(id);
          void refresh(id);
          beginStep("Running prediction…", id);
          response = await inferenceApi.predict(id, body);
          if (activeRunId.current === id) { setAutoStarting(false); setNotice(""); }
        }
        predictions.push(...normalizeResult(asRecord(response).result ?? response, chunk.length, snapshot.target));
        if (activeRunId.current === id) setProgress({ done: predictions.length, total: rows.length });
      }
      const completed: InferenceRun = { id: crypto.randomUUID(), time: Date.now(), duration: performance.now() - start, model: snapshot, rows, predictions, warnings: rows.map((r) => rowWarnings(r, featureSet).warnings) };
      return completed;
    },
    onSuccess: (completed) => {
      sessionRuns = keepRecent(sessionRuns, completed);
      setRecent(sessionRuns);
      if (completed.model.runId) void refresh(completed.model.runId);
      if (activeRunId.current === completed.model.runId) { setRun(completed); setSelected(0); setNotice(""); setAutoStarting(false); setStep(null); }
    },
    onError: (e, vars) => handleError(e, vars.snapshot.runId ?? ""),
  });

  function selectModel(id: string) { void navigate({ to: "/inference", search: { run_id: id } }); }
  function openRecent(item: InferenceRun) {
    if (item.model.runId === runId) {
      setRun(item); setSelected(0); setError(""); setUpgrade(null);
      setInput({ ...emptyInput(), mode: "JSON", json: JSON.stringify(item.rows, null, 2) });
    } else if (item.model.runId) { pendingOpen.current = item; selectModel(item.model.runId); }
  }

  const starting = autoStarting || (service.isPending && service.variables?.start === true);
  const endpoint = `${BACKEND_API_BASE}/api/models/${encodeURIComponent(runId || "<run_id>")}/inference`;
  const sample = Object.fromEntries(features.map((f) => [f.name, f.numeric ? f.min ?? 0 : f.categories?.[0] ?? "value"]));
  const curl = `curl -X POST '${endpoint}' \\\n  -H 'Authorization: Bearer <YOUR_TOKEN>' \\\n  -H 'Content-Type: application/json' \\\n  --data '${JSON.stringify(sample).replace(/'/g, "'\\''")}'`;
  const python = `import requests\n\nresponse = requests.post(\n    ${JSON.stringify(endpoint)},\n    headers={"Authorization": "Bearer <YOUR_TOKEN>"},\n    json=${JSON.stringify(sample)},\n)\nresponse.raise_for_status()\nprint(response.json()["result"])`;

  return (
    <main className="inference-page min-w-0 flex-1 overflow-y-auto bg-inf-canvas text-foreground">
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-border bg-card px-5 py-5 sm:px-7">
        <div>
          <h1 className="text-xl font-semibold">Inference</h1>
          <p className="mt-1 text-sm text-muted-foreground">Run real-time and batch predictions on your trained models.</p>
        </div>
        <div className="flex items-center gap-3">
          {model && <ServiceBadge running={model.serviceRunning} starting={starting || model.serviceStarting} failed={model.endpointStatus === "failed"} asleep={model.serviceAsleep} />}
          <Button variant="outline" className="min-h-11" disabled={!runId} onClick={() => setApiOpen(true)}><Code2 />API</Button>
        </div>
      </header>

      <div className="grid items-start gap-5 p-5 sm:p-7 lg:grid-cols-[280px_minmax(0,1fr)]">
        <SectionBoundary label="Models list" resetKey={list.dataUpdatedAt}>
          <ModelsList models={models} loading={list.isPending} error={list.isError} selectedRunId={runId} onRetry={() => void list.refetch()} onSelect={selectModel} />
        </SectionBoundary>

        <div className="min-w-0 space-y-5">
          <SectionBoundary label="Model summary" resetKey={runId}>
            {model ? (
              <ModelSummary model={model} starting={starting} busy={service.isPending || prediction.isPending}
                onStart={() => { setError(""); setErrorDetails(null); setUpgrade(null); service.mutate({ id: runId, start: true }); }} onStop={() => setStopOpen(true)} />
            ) : runId && (detail.isPending || list.isPending) ? (
              <Skeleton className="h-48 w-full rounded-2xl" />
            ) : detail.isError ? (
              <section className="inf-card" role="alert">
                <p className="text-sm">{detail.error instanceof InferenceApiError && detail.error.status === 404 ? "This model no longer exists." : detail.error instanceof InferenceApiError && detail.error.status === 403 ? "You don't have access to this model." : "Model details couldn't load."}</p>
                <Button variant="outline" className="mt-3 min-h-11" onClick={() => { void detail.refetch(); void list.refetch(); }}>Retry</Button>
              </section>
            ) : !list.isPending && (
              <section className="inf-card text-sm text-muted-foreground">Select a model to run inference.</section>
            )}
          </SectionBoundary>

          {model?.runId && (
            <SectionBoundary label="Input" resetKey={runId}>
              <InferenceInput key={runId} features={features} state={input} onChange={setInput} running={prediction.isPending || service.isPending} progress={progress}
                stepIndicator={step && Date.now() - operationStarted.current >= 5000 ? `${step.label} ${elapsed} s` : undefined}
                onRun={(rows, single, chunked) => { setError(""); setErrorDetails(null); setUpgrade(null); setNotice(""); setProgress({ done: 0, total: rows.length }); prediction.mutate({ rows, single, chunked, snapshot: model, featureSet: features }); }} />
            </SectionBoundary>
          )}

          {notice && <div role="status" className="rounded-lg border border-inf-active-border bg-inf-active p-4 text-sm text-inf-blue">{notice}</div>}
          {error && <div role="alert" className="inf-warning">{error}</div>}
          {upgrade && (
            <div role="alert" className="inf-warning">
              <p className="font-semibold">{upgrade.message}</p>
              {upgrade.plans.length > 0 && <p className="mt-1">Available plans: {upgrade.plans.join(", ")}</p>}
              <Button asChild className="mt-3 min-h-11"><Link to="/settings" search={{ tab: "billing" }}>Upgrade plan</Link></Button>
            </div>
          )}
          {errorDetails && (error || upgrade) && <details className="text-xs text-muted-foreground"><summary className="inline-flex min-h-11 cursor-pointer items-center">Details</summary><dl className="space-y-1"><div>HTTP status: {errorDetails.status}</div><div>Call: {errorDetails.call}</div>{errorDetails.detail && <div className="break-words">Server detail: {errorDetails.detail}</div>}</dl></details>}

          <SectionBoundary label="Results" resetKey={run?.id}>
            <InferenceResults run={run} selected={selected} onSelect={setSelected} />
          </SectionBoundary>

          <section className="inf-card" aria-label="Recent runs (this session)">
            <h2 className="font-semibold">Recent runs (this session)</h2>
            {!recent.length ? <p className="py-6 text-sm text-muted-foreground">No runs in this session.</p> : (
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-muted-foreground"><tr><th className="px-3 py-2 font-medium">Time</th><th className="px-3 py-2 font-medium">Model</th><th className="px-3 py-2 font-medium">Rows</th><th className="px-3 py-2 font-medium">Summary</th><th className="px-3 py-2"><span className="sr-only">Actions</span></th></tr></thead>
                  <tbody>
                    {recent.map((item) => {
                      const summary = recentSummary(item);
                      return (
                        <tr key={item.id} className="border-t border-border">
                          <td className="whitespace-nowrap px-3">{new Date(item.time).toLocaleTimeString("en-US")}</td>
                          <td className="max-w-64 break-all px-3 font-mono">{item.model.name ?? "—"}</td>
                          <td className="px-3">{item.rows.length.toLocaleString("en-US")}</td>
                          <td className="max-w-64 px-3">{summary}</td>
                          <td><Button variant="link" className="min-h-11" disabled={prediction.isPending} onClick={() => openRecent(item)}>Open</Button></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      </div>

      <Sheet open={apiOpen} onOpenChange={setApiOpen}>
        <SheetContent className="inference-page w-full overflow-y-auto sm:max-w-xl">
          <SheetHeader><SheetTitle>Inference API</SheetTitle><SheetDescription>POST /api/models/{runId || "<run_id>"}/inference</SheetDescription></SheetHeader>
          {([["cURL", curl], ["Python", python]] as const).map(([label, code]) => (
            <div key={label} className="mt-6">
              <div className="mb-2 flex items-center justify-between"><h3 className="font-semibold">{label}</h3><Button variant="outline" className="min-h-11" onClick={() => copyText(code, `${label} copied`)}><Copy />Copy</Button></div>
              <pre className="overflow-x-auto rounded-lg bg-inf-input p-4 text-xs"><code>{code}</code></pre>
            </div>
          ))}
        </SheetContent>
      </Sheet>

      <AlertDialog open={stopOpen} onOpenChange={setStopOpen}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>Stop the inference service?</AlertDialogTitle><AlertDialogDescription>The next run will start it again and may take a few seconds.</AlertDialogDescription></AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="min-h-11">Cancel</AlertDialogCancel>
            <AlertDialogAction className="min-h-11" onClick={() => { setError(""); setUpgrade(null); service.mutate({ id: runId, start: false }); }}>Stop service</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
}
