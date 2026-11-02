import { AlertTriangle } from "lucide-react";
import {
  VERIFIER_NOTICE_TITLE,
  verifierNoticeSummary,
  type VerifierNotice,
} from "@/lib/verifier-notice";

/**
 * The claim verifier's findings under a reply it did not clear. Code-written
 * only: fixed labels plus the verifier's own detail sentences.
 */
export function VerifierNoticeBlock({ notice }: { notice: VerifierNotice }) {
  return (
    <section
      role="note"
      aria-label="Claim check"
      data-testid="verifier-notice"
      data-source="claim-verifier"
      className="mt-2 rounded-lg border border-[#fedf89] bg-[#fffaeb] px-3 py-2 text-xs text-[#93370d]"
    >
      <p className="flex items-center gap-1.5 font-semibold">
        <AlertTriangle className="size-3.5 shrink-0" aria-hidden />
        {VERIFIER_NOTICE_TITLE}
      </p>
      <p className="mt-1" data-testid="verifier-notice-summary">
        {verifierNoticeSummary(notice)}
      </p>
      {notice.lines.length > 0 && (
        <ul className="mt-1.5 flex flex-col gap-1">
          {notice.lines.map((line, i) => (
            <li key={i}>
              <span className="font-semibold">
                {line.check} · {line.verdict}
              </span>
              {line.detail ? `: ${line.detail}` : ""}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
