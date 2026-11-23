// The Ask state of /instant: one question, three example chips, and the
// answer. Until PR-6 lands the recipe endpoint the answer is "coming soon",
// with the question echoed so the user sees what was taken.
import { useState, type FormEvent } from "react";
import type { InstantAskResponse } from "@/lib/api/instant";

export function AskBox({
  examples,
  busy,
  error,
  result,
  onAsk,
}: {
  examples: string[];
  busy: boolean;
  error: string | null;
  result: InstantAskResponse | null;
  onAsk: (question: string) => void;
}) {
  const [question, setQuestion] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (q && !busy) onAsk(q);
  };

  return (
    <section aria-label="Ask" className="flex flex-col gap-3" data-testid="instant-ask">
      <h2 className="text-lg font-semibold text-primary">Ask one question</h2>
      <form onSubmit={submit} className="flex flex-col gap-2 sm:flex-row">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What do you want to know about this data?"
          aria-label="Your question"
          disabled={busy}
          className="min-w-0 flex-1 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary placeholder:text-placeholder focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-lg bg-brand-solid px-4 py-2 text-sm font-semibold text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Asking…" : "Ask"}
        </button>
      </form>

      <div className="flex flex-wrap gap-2" aria-label="Example questions">
        {examples.map((q) => (
          <button
            key={q}
            type="button"
            disabled={busy}
            onClick={() => setQuestion(q)}
            className="rounded-full border border-secondary bg-secondary px-3 py-1 text-xs text-secondary transition-colors hover:bg-primary_hover disabled:opacity-60"
          >
            {q}
          </button>
        ))}
      </div>

      {error ? (
        <p role="alert" className="text-sm text-error-primary">
          {error}
        </p>
      ) : null}

      {result?.status === "coming_soon" ? (
        <div
          role="status"
          data-testid="instant-coming-soon"
          className="rounded-xl border border-secondary bg-secondary p-4 text-sm text-secondary"
        >
          <p className="font-semibold text-primary">Coming soon</p>
          <p className="mt-1">
            Verified reports are on their way. Your question is noted:{" "}
            <q className="text-primary" data-testid="instant-echo">
              {result.question}
            </q>
          </p>
        </div>
      ) : null}
    </section>
  );
}
