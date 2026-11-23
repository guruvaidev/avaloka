/**
 * One-time, in-memory hand-off of a chat prompt (e.g. from Auto Insights
 * "Suggested next steps") to the analysis chat. Never persisted, so a reload
 * or Back/Forward cannot resend it.
 */
export type PendingChatPrompt = { id: string; text: string; aid: string | null };

let current: PendingChatPrompt | null = null;
let counter = 0;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export function setPendingChatPrompt(text: string, aid: string | null): PendingChatPrompt | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  counter += 1;
  current = { id: `pp-${Date.now()}-${counter}`, text: trimmed, aid };
  emit();
  return current;
}

export function getPendingChatPrompt(): PendingChatPrompt | null {
  return current;
}

/** Clears the prompt only if it is still the one with this id. */
export function clearPendingChatPrompt(id?: string): void {
  if (!current || (id && current.id !== id)) return;
  current = null;
  emit();
}

export function subscribePendingChatPrompt(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Sends the pending prompt once through the caller's existing send function.
 * Waits (returns "waiting") while the chat is busy; on failure calls onFailed
 * with the text and never retries.
 */
export async function deliverPendingChatPrompt(opts: {
  aid: string | null;
  isBusy: boolean;
  sentIds: Set<string>;
  send: (text: string) => Promise<boolean>;
  onBeforeSend?: () => void;
  onFailed: (text: string) => void;
}): Promise<"none" | "waiting" | "sent" | "failed"> {
  const p = current;
  if (!p || p.aid !== opts.aid) return "none";
  if (opts.sentIds.has(p.id)) {
    clearPendingChatPrompt(p.id);
    return "none";
  }
  if (opts.isBusy) return "waiting";
  opts.sentIds.add(p.id);
  clearPendingChatPrompt(p.id);
  opts.onBeforeSend?.();
  let ok = false;
  try {
    ok = await opts.send(p.text);
  } catch {
    ok = false;
  }
  if (!ok) {
    opts.onFailed(p.text);
    return "failed";
  }
  return "sent";
}
