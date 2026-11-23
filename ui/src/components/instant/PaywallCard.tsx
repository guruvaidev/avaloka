// The paywall placeholder for a locked run. Copy comes from one config object
// (lib/instant-paywall.ts) and carries no price; the button does nothing until
// Checkout lands (PR-8).
import { PAYWALL_COPY, type PaywallCopy } from "@/lib/instant-paywall";

export function PaywallCard({ copy = PAYWALL_COPY }: { copy?: PaywallCopy }) {
  return (
    <section
      aria-label={copy.title}
      className="flex flex-col gap-3 rounded-xl border border-brand bg-brand-primary p-4"
      data-testid="instant-paywall"
    >
      <h2 className="text-lg font-semibold text-primary">{copy.title}</h2>
      <p className="text-sm text-secondary">{copy.body}</p>
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
        <button
          type="button"
          disabled
          title={copy.note}
          className="rounded-lg bg-brand-solid px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {copy.cta}
        </button>
        <p className="text-xs text-tertiary">{copy.note}</p>
      </div>
    </section>
  );
}
