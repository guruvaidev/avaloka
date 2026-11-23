// Limitations: the report's caveats, what did not verify (completion.missing)
// and how the answer was computed (method.summary). All three are the
// report's own text, written by code, shown plain — told, never apologised
// for (2.0-OUTCOMES.md §3.5).
import type { Caveat, Completion, Method } from "@/lib/api/instant";
import { COMPLETION_WORDS } from "@/lib/instant-labels";

export function LimitationsPanel({
  caveats,
  completion,
  method,
}: {
  caveats: Caveat[];
  completion?: Completion;
  method?: Method;
}) {
  const missing = completion?.missing ?? [];
  return (
    <section
      aria-label="Limitations"
      className="flex flex-col gap-4 rounded-xl border border-secondary p-4"
      data-testid="instant-limitations"
    >
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-primary">Limitations</h2>
        {completion ? (
          <span
            className="shrink-0 rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-secondary"
            data-testid="instant-completion"
          >
            {COMPLETION_WORDS[completion.state]}
          </span>
        ) : null}
      </div>

      {caveats.length ? (
        <ul
          className="list-disc space-y-1 pl-5 text-sm text-secondary"
          data-testid="instant-caveats"
        >
          {caveats.map((c, i) => (
            <li key={`${c.kind}-${i}`} data-kind={c.kind}>
              {c.text}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-tertiary">No caveats were raised.</p>
      )}

      {missing.length ? (
        <div className="flex flex-col gap-1" data-testid="instant-missing">
          <h3 className="text-sm font-semibold text-primary">Did not verify</h3>
          <ul className="list-disc space-y-1 pl-5 text-sm text-secondary">
            {missing.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {method ? (
        <div className="flex flex-col gap-1" data-testid="instant-method">
          <h3 className="text-sm font-semibold text-primary">How it was computed</h3>
          <p className="text-sm text-secondary">{method.summary}</p>
        </div>
      ) : null}
    </section>
  );
}
