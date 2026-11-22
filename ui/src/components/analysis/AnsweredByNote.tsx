import { Shuffle } from "lucide-react";
import { ANSWERED_BY_TITLE, answeredBySentence, type AnsweredBy } from "@/lib/answered-by";

/**
 * Under a reply that a fallback model answered: which model answered and why.
 * Code-written only: fixed labels plus the backend's recorded reason.
 */
export function AnsweredByNote({ answeredBy }: { answeredBy: AnsweredBy }) {
  return (
    <section
      role="note"
      aria-label="Model provenance"
      data-testid="answered-by"
      data-source="model-tiers"
      className="mt-2 rounded-lg border border-[#d0d5dd] bg-[#f9fafb] px-3 py-2 text-xs text-[#344054]"
    >
      <p className="flex items-center gap-1.5 font-semibold">
        <Shuffle className="size-3.5 shrink-0" aria-hidden />
        {ANSWERED_BY_TITLE}
      </p>
      <p className="mt-1" data-testid="answered-by-sentence">
        {answeredBySentence(answeredBy)}
      </p>
    </section>
  );
}
