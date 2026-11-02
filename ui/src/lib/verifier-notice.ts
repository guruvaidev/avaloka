// The claim verifier's result, shown under a reply it did not clear.
//
// verify_claims (app/agents/claim_verifier.py) runs as the graph's last node
// and returns verification_safe_to_present plus verification_report on every
// chat turn. It only reports; it never blocks or edits the reply. This turns
// that report into a notice. Every word is code-written: fixed labels here and
// the verifier's own `detail` sentences, which are f-strings in
// claim_verifier.py. The finding's `claim` (the model's sentence) is never
// shown, and nothing here says a reply was checked or is correct.

export type VerifierFinding = {
  claim?: unknown;
  verdict?: unknown;
  check?: unknown;
  detail?: unknown;
  evidence_ref?: unknown;
};

export type VerifierNoticeLine = { check: string; verdict: string; detail: string };

export type VerifierNotice = {
  claimsChecked: number | null;
  problems: number;
  lines: VerifierNoticeLine[];
};

export const VERIFIER_NOTICE_TITLE = "Some numbers or claims in this answer did not check out";

const VERDICT_LABEL: Record<string, string> = {
  contradicted: "Contradicted by the computed results",
  overreach: "Goes beyond the evidence",
  unsupported: "Not found in the computed results",
};

const CHECK_LABEL: Record<string, string> = {
  causal_overreach: "Cause claimed from a correlation",
  invented_number: "Figure not in the results",
  contradicted_by_metrics: "Contradicts the model's metrics",
  ordering_contradicted: "Contradicts the result's ordering",
  absolute_language: "Absolute or predictive wording",
  unsupported_significance: "Significance without a test",
};

const MAX_DETAIL = 280;
const MAX_LINES = 5;

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * The notice for one chat response, or null when there is nothing to show:
 * the verifier cleared the reply, did not run, or sent no usable findings.
 */
export function verifierNotice(response: unknown): VerifierNotice | null {
  if (!response || typeof response !== "object") return null;
  const res = response as Record<string, unknown>;
  // Only an explicit false: absent or null means the verifier did not run.
  if (res.verification_safe_to_present !== false) return null;
  const report = (res.verification_report ?? {}) as Record<string, unknown>;
  const findings: VerifierFinding[] = Array.isArray(report.findings) ? report.findings : [];
  const problems = findings.filter(
    (f) => typeof f?.verdict === "string" && f.verdict !== "supported",
  );
  const lines = problems.slice(0, MAX_LINES).map((f) => {
    const verdict = String(f.verdict);
    const check = typeof f.check === "string" ? f.check : "";
    const detail = typeof f.detail === "string" ? f.detail.trim() : "";
    return {
      check: CHECK_LABEL[check] ?? "Claim check",
      verdict: VERDICT_LABEL[verdict] ?? "Did not check out",
      detail: detail.length > MAX_DETAIL ? `${detail.slice(0, MAX_DETAIL - 1)}…` : detail,
    };
  });
  return {
    claimsChecked: count(report.claims_checked),
    problems: count(report.n_problems) ?? problems.length,
    lines,
  };
}

/** One code-written sentence for the notice header. */
export function verifierNoticeSummary(notice: VerifierNotice): string {
  const n = notice.problems;
  const of = notice.claimsChecked != null ? ` of ${notice.claimsChecked}` : "";
  return `${n}${of} claim${(notice.claimsChecked ?? n) === 1 ? "" : "s"} in this answer did not match the computed results. Treat this answer as unchecked.`;
}
