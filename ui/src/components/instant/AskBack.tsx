// Verdict "ask": the recipe computed nothing and asks one question back. The
// options are the values the recipe accepts for that parameter; choosing one
// asks again with `clarify` set.
export function AskBack({
  text,
  parameter,
  options,
  busy,
  error,
  onChoose,
}: {
  text: string;
  parameter: string;
  options: string[];
  busy: boolean;
  error: string | null;
  onChoose: (option: string) => void;
}) {
  return (
    <section
      aria-label="One question back"
      className="flex flex-col gap-3 rounded-xl border border-secondary p-4"
      data-testid="instant-ask-back"
    >
      <p className="text-sm text-primary">{text}</p>
      <div className="flex flex-wrap gap-2" aria-label={`Options for ${parameter}`}>
        {options.map((o) => (
          <button
            key={o}
            type="button"
            disabled={busy}
            onClick={() => onChoose(o)}
            className="rounded-full border border-secondary bg-secondary px-3 py-1 text-sm text-secondary transition-colors hover:bg-primary_hover disabled:opacity-60"
          >
            {o}
          </button>
        ))}
      </div>
      {busy ? <p className="text-sm text-tertiary">Asking…</p> : null}
      {error ? (
        <p role="alert" className="text-sm text-error-primary">
          {error}
        </p>
      ) : null}
    </section>
  );
}
