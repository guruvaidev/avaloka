// @vitest-environment happy-dom
// The fixture is the real 1.6 claim verifier's output: verify_claims() from
// app/agents/claim_verifier.py run on a narrative with one invented figure,
// one causal claim and one absolute claim, against a two-row result.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { VerifierNoticeBlock } from "../../src/components/analysis/VerifierNoticeBlock";
import { VERIFIER_NOTICE_TITLE, verifierNotice } from "../../src/lib/verifier-notice";

const source = (rel: string) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf8");
const FIXTURE = JSON.parse(source("../fixtures/claim-verifier-reports.json"));
const UNSAFE = FIXTURE.unsafe;
const SAFE = FIXTURE.safe;

function renderNotice(response: unknown) {
  const notice = verifierNotice(response);
  if (!notice) return null;
  render(<VerifierNoticeBlock notice={notice} />);
  return screen.getByTestId("verifier-notice");
}

describe("verifier notice", () => {
  it("an unsafe report renders the block with the report's figures", () => {
    const block = renderNotice(UNSAFE);
    expect(block).not.toBeNull();
    const text = block!.textContent ?? "";
    expect(text).toContain(VERIFIER_NOTICE_TITLE);
    expect(text).toContain("3 of 4 claims in this answer did not match the computed results.");
    expect(text).toContain("Treat this answer as unchecked.");
    expect(text).toContain(
      "cites 950, which does not appear in the computed evidence within tolerance",
    );
    expect(text).toContain("Figure not in the results · Not found in the computed results");
    expect(text).toContain("Cause claimed from a correlation · Goes beyond the evidence");
    expect(text).toContain("Absolute or predictive wording · Goes beyond the evidence");
    expect(block!.querySelectorAll("li")).toHaveLength(3);
    expect(block!.getAttribute("data-source")).toBe("claim-verifier");
  });

  it("a safe report shows no block", () => {
    expect(SAFE.verification_safe_to_present).toBe(true);
    expect(verifierNotice(SAFE)).toBeNull();
    expect(renderNotice(SAFE)).toBeNull();
  });

  it("no verifier result (absent or null) shows no block either", () => {
    expect(verifierNotice({ messages: [] })).toBeNull();
    expect(
      verifierNotice({
        verification_safe_to_present: null,
        verification_report: UNSAFE.verification_report,
      }),
    ).toBeNull();
    expect(verifierNotice(null)).toBeNull();
  });

  it("the block is code-written: report fields and fixed labels, never the model's sentences", () => {
    const block = renderNotice(UNSAFE)!;
    const text = block.textContent ?? "";
    for (const f of UNSAFE.verification_report.findings) {
      expect(text).not.toContain(f.claim);
    }
    // Each line is exactly a fixed label pair plus the verifier's own detail.
    const lines = [...block.querySelectorAll("li")].map((li) => li.textContent);
    const details = UNSAFE.verification_report.findings.map((f: { detail: string }) => f.detail);
    lines.forEach((line, i) => expect(line?.endsWith(`: ${details[i]}`)).toBe(true));
  });

  it("an unknown check or verdict gets a neutral label, not the raw field", () => {
    const notice = verifierNotice({
      verification_safe_to_present: false,
      verification_report: {
        claims_checked: 1,
        n_problems: 1,
        findings: [
          {
            claim: "MODEL PROSE 4242",
            verdict: "weird",
            check: "new_check",
            detail: "code detail",
          },
        ],
      },
    })!;
    expect(notice.lines).toEqual([
      { check: "Claim check", verdict: "Did not check out", detail: "code detail" },
    ]);
    render(<VerifierNoticeBlock notice={notice} />);
    expect(screen.getByTestId("verifier-notice").textContent).not.toContain("MODEL PROSE");
  });
});

describe("deferred turns", () => {
  // GET /threads/{id}/pending-turn answers {status: "done", kind, result}, where
  // result is the same ChatResponse dump (server.py _pending_turn_done(...,
  // resp.model_dump())), so it carries the verifier fields. training-jobs
  // resolves with that envelope and analysis.tsx reads pending.result.
  it("reads the verifier fields from the polled result, not the envelope", () => {
    const envelope = { status: "done", kind: "analysis", result: UNSAFE };
    expect(verifierNotice(envelope)).toBeNull();
    expect(verifierNotice(envelope.result)?.problems).toBe(3);
  });

  it("analysis.tsx renders the polled result through the same reply path", () => {
    const page = source("../../src/routes/analysis.tsx");
    expect(page).toContain("res = pending.result;");
    expect(page).toContain("applyBackgroundTrainingResult(claimed.result");
  });
});

describe("analysis page wiring", () => {
  const page = source("../../src/routes/analysis.tsx");
  it("attaches the verifier result to the reply and renders it under the bubble", () => {
    expect(page).toContain("const verification = verifierNotice(res);");
    // both shapes of the turn's reply (text, or the "analysis complete" line) carry it
    expect(page).toContain("content: lastAssistantText, time: nowTime(), verification }");
    expect(page).toMatch(
      /content: ANALYSIS_COMPLETE_MESSAGE,\s*time: nowTime\(\),\s*verification,/,
    );
    expect(page).toContain("verification: verifierNotice(res),");
    expect(page).toContain(
      "{message.verification && <VerifierNoticeBlock notice={message.verification} />}",
    );
  });
});
