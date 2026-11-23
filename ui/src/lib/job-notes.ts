// Background-job notes in the chat (app/core/job_notes.py on the API).
//
// A job the user started from a chat thread leaves one short, code-written
// assistant note in that thread when it ends. The API delivers notes only at a
// turn boundary and writes the transcript row itself; the UI's job is to show
// a delivered batch once, after the reply it followed, and to ask for pending
// notes when the bell says a job of this thread finished.

export type JobNoteBatch = {
  batch_id: string;
  thread_id: string;
  text: string;
  job_ids: string[];
  count: number;
  analysis_id?: string | null;
  transcript_written?: boolean;
  created_at?: string;
};

export type ThreadJobNotesResponse = {
  delivered: JobNoteBatch | null;
  deferred: boolean;
  pending: number;
};

/** The chat message a delivered batch becomes. `jobNote` keeps it out of
 *  feedback and out of anything that learns from assistant turns. */
export type JobNoteChatMessage = {
  id: string;
  role: "ai";
  content: string;
  time: string;
  jobNote: true;
  batchId: string;
};

/** The frame the API puts in front of the note for the model is not for people. */
const MODEL_FRAME_RE = /^\[Background job update[^\]]*\]\s*/;

export function jobNoteChatMessages(
  batches: JobNoteBatch[] | null | undefined,
  time: string,
  seen: Set<string>,
): JobNoteChatMessage[] {
  const out: JobNoteChatMessage[] = [];
  for (const b of batches ?? []) {
    if (!b || !b.batch_id || seen.has(b.batch_id)) continue;
    seen.add(b.batch_id);
    out.push({
      id: `job-note-${b.batch_id}`,
      role: "ai",
      content: String(b.text ?? "").replace(MODEL_FRAME_RE, ""),
      time,
      jobNote: true,
      batchId: b.batch_id,
    });
  }
  return out;
}

type MetaLike = { thread_id?: unknown; job_id?: unknown } | null | undefined;

/** A bell item that says a background job of `threadId` finished. */
export function isThreadJobNotification(n: { metadata?: unknown }, threadId: string | null | undefined): boolean {
  if (!threadId) return false;
  const meta = (n?.metadata && typeof n.metadata === "object" ? n.metadata : null) as MetaLike;
  return !!meta && meta.thread_id === threadId && typeof meta.job_id === "string" && meta.job_id.length > 0;
}

/** A persisted transcript row written by the API for a note batch. */
export function batchIdOfTranscriptRow(output: unknown): string | null {
  const jn = (output && typeof output === "object" ? (output as { job_note?: unknown }).job_note : null) as
    | { batch_id?: unknown }
    | null;
  return jn && typeof jn.batch_id === "string" ? jn.batch_id : null;
}
