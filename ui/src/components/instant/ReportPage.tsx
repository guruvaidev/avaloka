// The /instant/:runId page body: header (recipe title, verdict, question),
// the report in its frame, Evidence, Limitations, downloads, and — by
// verdict — the paywall card, the question back, or the refusal. The route
// wraps it in the app shell; tests render it alone with a mocked fetch.
// Nothing here is an agent, and no number is written by this page.
import { useEffect, useState } from "react";
import { instantApi, type InstantApi, type InstantRun } from "@/lib/api/instant";
import { VERDICT_LABELS, recipeTitle } from "@/lib/instant-labels";
import { ReportFrame } from "./ReportFrame";
import { EvidencePanel } from "./EvidencePanel";
import { LimitationsPanel } from "./LimitationsPanel";
import { DownloadRow } from "./DownloadRow";
import { PaywallCard } from "./PaywallCard";
import { AskBack } from "./AskBack";

/** Where the chat lane lives; a refused question is pointed there. */
export const DEEP_INVESTIGATION_PATH = "/analysis";

const VERDICT_CLASS: Record<InstantRun["verdict"], string> = {
  pass: "bg-success-secondary text-success-primary",
  qualified: "bg-warning-secondary text-warning-primary",
  ask: "bg-secondary text-secondary",
  refuse: "bg-error-secondary text-error-primary",
};

export function ReportPage({
  runId,
  api = instantApi,
  onReady,
}: {
  runId: string;
  api?: InstantApi;
  /** A question-back answered with `ready`: take the user to the new run. */
  onReady?: (runId: string) => void;
}) {
  const [run, setRun] = useState<InstantRun | null>(null);
  const [html, setHtml] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setRun(null);
    setHtml(null);
    setError(null);
    api
      .fetchInstantRun(runId)
      .then(async (r) => {
        if (!alive) return;
        setRun(r);
        if (r.export_url) {
          try {
            const text = await api.fetchInstantExport(r.export_url);
            if (alive) setHtml(text);
          } catch {
            if (alive) setHtml(null);
          }
        }
      })
      .catch(
        (e) => alive && setError(e instanceof Error ? e.message : "This run could not be read."),
      );
    return () => {
      alive = false;
    };
  }, [api, runId]);

  const clarify = async (option: string) => {
    if (!run?.ask) return;
    setAsking(true);
    setAskError(null);
    try {
      const res = await api.askInstant({
        dataset_id: run.dataset_id,
        question: run.question ?? run.ask.text,
        clarify: { parameter: run.ask.parameter, option },
      });
      if (res.status === "ready") onReady?.(res.run_id);
      else setAskError("Verified reports are on their way; the answer was noted.");
    } catch (e) {
      setAskError(e instanceof Error ? e.message : "Could not ask");
    } finally {
      setAsking(false);
    }
  };

  if (error) {
    return (
      <main className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-8 sm:px-6">
        <p role="alert" className="text-sm text-error-primary">
          {error}
        </p>
        <a href="/instant" className="text-sm text-tertiary underline-offset-2 hover:underline">
          Ask another question
        </a>
      </main>
    );
  }
  if (!run) {
    return (
      <main className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-8 sm:px-6">
        <p role="status" className="text-sm text-tertiary">
          Loading the report…
        </p>
      </main>
    );
  }

  const report = run.report;
  const checks = report?.data_checks ?? [];
  const caveats = report?.caveats ?? [];
  const hasDownloads = Boolean(run.export_url || run.notebook_url || run.markdown_url);

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8 sm:px-6">
      <header className="flex flex-col gap-2" data-testid="instant-report-header">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-2xl font-semibold text-primary" data-testid="instant-recipe-title">
            {recipeTitle(run.recipe.id)}
          </h1>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${VERDICT_CLASS[run.verdict]}`}
            data-testid="instant-verdict"
            data-verdict={run.verdict}
          >
            {VERDICT_LABELS[run.verdict]}
          </span>
        </div>
        {run.question ? (
          <p className="text-sm text-tertiary">
            <q data-testid="instant-question">{run.question}</q>
          </p>
        ) : null}
      </header>

      {run.locked ? <PaywallCard /> : null}

      {run.verdict === "ask" && run.ask ? (
        <AskBack
          text={run.ask.text}
          parameter={run.ask.parameter}
          options={run.ask.options}
          busy={asking}
          error={askError}
          onChoose={clarify}
        />
      ) : null}

      {run.verdict === "refuse" ? (
        <section
          aria-label="Not answered"
          className="flex flex-col gap-3 rounded-xl border border-secondary p-4"
          data-testid="instant-refusal"
        >
          <p className="text-sm text-primary">
            {run.refusal?.text ?? "This question is outside what the recipes can answer."}
          </p>
          <p className="text-sm text-secondary">
            <a
              href={DEEP_INVESTIGATION_PATH}
              className="font-medium text-brand-secondary underline-offset-2 hover:underline"
              data-testid="instant-deep-link"
            >
              Try Deep Investigation
            </a>{" "}
            — the lane where Avaloka writes and runs code for the question; every sentence a model
            writes there is marked as commentary.
          </p>
        </section>
      ) : null}

      {html != null ? <ReportFrame html={html} title={recipeTitle(run.recipe.id)} /> : null}

      {report ? (
        <>
          <EvidencePanel checks={checks} caveats={caveats} />
          <LimitationsPanel
            caveats={caveats}
            completion={report.completion}
            method={report.method}
          />
        </>
      ) : null}

      {hasDownloads || report ? (
        <DownloadRow
          exportUrl={run.export_url}
          notebookUrl={run.notebook_url}
          markdownUrl={run.markdown_url}
        />
      ) : null}

      <a
        href="/instant"
        className="self-start text-sm text-tertiary underline-offset-2 hover:underline"
      >
        Ask another question
      </a>
    </main>
  );
}
