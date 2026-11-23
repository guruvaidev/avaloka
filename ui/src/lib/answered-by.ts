// Which model answered a reply, when a fallback tier stood in for the configured
// model (owner ruling 2026-11-02: a model swap is never silent).
//
// The backend stamps the record where the swap happens (app/core/model_tiers.py)
// and returns it as ChatResponse.answered_by. Every word shown is code-written:
// fixed labels here plus the backend's `reason`, which is built from the recorded
// HTTP status and provider message, never from a model's text.

export type AnsweredBy = {
  configuredModel: string;
  answeredModel: string;
  provider: string;
  reason: string;
};

export const ANSWERED_BY_TITLE = "Answered by a fallback model";

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

/** The record for one chat response, or null when the configured model answered. */
export function answeredBy(response: unknown): AnsweredBy | null {
  if (!response || typeof response !== "object") return null;
  const raw = (response as Record<string, unknown>).answered_by;
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const configuredModel = text(r.configured_model);
  const answeredModel = text(r.answered_model);
  // The record exists only when a fallback tier answered. The model id may be the
  // same (gpt-oss-120b on OpenRouter, then on Groq): the provider is what changed.
  if (!configuredModel || !answeredModel) return null;
  return {
    configuredModel,
    answeredModel,
    provider: text(r.provider),
    reason: text(r.reason),
  };
}

/** "openai/gpt-oss-120b could not answer (reason); openai/gpt-6-luna via openrouter answered." */
export function answeredBySentence(a: AnsweredBy): string {
  const why = a.reason ? ` (${a.reason})` : "";
  const via = a.provider ? ` via ${a.provider}` : "";
  return `${a.configuredModel} did not answer${why}; ${a.answeredModel}${via} answered instead.`;
}
