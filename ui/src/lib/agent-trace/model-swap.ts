// A model call that a fallback tier answered instead of the configured model
// (owner ruling 2026-11-02: a model swap is never silent). The backend records it
// on the trace's inference call as `answered_by` (app/core/model_tiers.answered_record
// via app/trace/recorder.py). Code-written only: fixed words plus the recorded
// models, provider and reason, which come from the HTTP status, never from a model.
import type { InferenceCall } from "./types";

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

/** "configured openai/gpt-4o → answered by openai/gpt-6-luna via openrouter (reason)", or null. */
export function modelSwapLine(call: InferenceCall): string | null {
  const rec = call.answered_by;
  if (!rec || typeof rec !== "object") return null;
  const configured = text(rec.configured_model);
  const answered = text(rec.answered_model);
  if (!configured || !answered) return null;
  const via = text(rec.provider) ? ` via ${text(rec.provider)}` : "";
  const why = text(rec.reason) ? ` (${text(rec.reason)})` : "";
  return `configured ${configured} → answered by ${answered}${via}${why}`;
}
