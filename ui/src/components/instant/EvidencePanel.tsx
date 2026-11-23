// Evidence: the source-data checks (report.data_checks, the GateReport) as
// one row per gate — its label, its status word and, when it raised a
// caveat, the report's own sentence for that caveat kind. No number is shown
// here: the figures a gate measured live inside the frame, where the backend
// verified them. tests/component/instant-report.test.tsx fails if a digit
// reaches this panel.
import type { Caveat, DataCheck } from "@/lib/api/instant";
import { STATUS_WORDS, gateLabel } from "@/lib/instant-labels";

const STATUS_CLASS: Record<DataCheck["status"], string> = {
  passed: "bg-success-secondary text-success-primary",
  qualified: "bg-warning-secondary text-warning-primary",
  failed: "bg-error-secondary text-error-primary",
};

export function EvidencePanel({ checks, caveats }: { checks: DataCheck[]; caveats: Caveat[] }) {
  return (
    <section
      aria-label="Evidence"
      className="flex flex-col gap-3 rounded-xl border border-secondary p-4"
      data-testid="instant-evidence"
    >
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-primary">Evidence</h2>
        <p className="text-sm text-tertiary">
          Source-data checks, run by code before anything was computed.
        </p>
      </div>
      {checks.length ? (
        <ul className="flex flex-col divide-y divide-secondary">
          {checks.map((c) => {
            const caveat = c.caveat_kind
              ? caveats.find((k) => k.kind === c.caveat_kind)?.text
              : undefined;
            return (
              <li
                key={c.gate}
                data-gate={c.gate}
                data-status={c.status}
                className="flex flex-col gap-1 py-2"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm text-primary">{gateLabel(c.gate)}</span>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_CLASS[c.status]}`}
                  >
                    {STATUS_WORDS[c.status]}
                  </span>
                </div>
                {caveat ? <p className="text-sm text-secondary">{caveat}</p> : null}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="text-sm text-tertiary">No source-data checks were recorded for this run.</p>
      )}
    </section>
  );
}
