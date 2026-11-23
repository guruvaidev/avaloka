// Avaloka talking the user through each step (ChatResponse / UploadResponse
// `assistant_prompts`, built by app/api/chat_voice.py and chat_swarm.py).
//
// Each prompt is one short message in her voice plus one concrete question and
// quick replies; the UI renders each as its own chat bubble with buttons, and a
// button press sends `send` as the user's next message. Everything is read
// defensively: an older API without the field yields no bubbles, and a prompt
// missing a part is rendered without it, never dropped as a whole.

import { parseReplySegments, type ReplySegment } from "@/lib/reply-segments";

export interface QuickReply {
  label: string;
  send: string;
}

export interface AssistantPrompt {
  step: string;
  message: string;
  /** Which words of `message` a model wrote (2.0 commentary marking). */
  message_segments?: ReplySegment[];
  question?: string;
  quick_replies?: QuickReply[];
  link?: { label: string; href: string };
}

export interface PromptMessage {
  id: string;
  role: "ai";
  content: string;
  time: string;
  step: string;
  quickReplies?: QuickReply[];
  link?: { label: string; href: string };
  segments?: ReplySegment[];
}

/** The prompts a response carries, or none. Never throws on a strange shape. */
export function readAssistantPrompts(res: unknown): AssistantPrompt[] {
  const raw = (res as { assistant_prompts?: unknown } | null | undefined)?.assistant_prompts;
  if (!Array.isArray(raw)) return [];
  const out: AssistantPrompt[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const p = item as Record<string, unknown>;
    const message = typeof p.message === "string" ? p.message.trim() : "";
    if (!message) continue;
    const replies = Array.isArray(p.quick_replies)
      ? (p.quick_replies as unknown[])
          .map((r) => {
            if (typeof r === "string") return { label: r, send: r };
            const o = (r ?? {}) as Record<string, unknown>;
            const label = typeof o.label === "string" ? o.label : "";
            const send = typeof o.send === "string" && o.send ? o.send : label;
            return label ? { label, send } : null;
          })
          .filter((r): r is QuickReply => r !== null)
      : [];
    const link = p.link && typeof p.link === "object" ? (p.link as Record<string, unknown>) : null;
    out.push({
      step: typeof p.step === "string" ? p.step : "result",
      message,
      message_segments: parseReplySegments(p.message_segments) ?? undefined,
      question: typeof p.question === "string" && p.question.trim() ? p.question.trim() : undefined,
      quick_replies: replies,
      link:
        link && typeof link.href === "string" && typeof link.label === "string"
          ? { label: link.label, href: link.href }
          : undefined,
    });
  }
  return out;
}

/**
 * The chat bubbles for a response's prompts: one per step, in the API's order.
 * `alreadyShown` is the assistant text the thread just rendered: a prompt whose
 * message is that same text (the swarm turn's story) keeps only its question
 * and buttons, so the story is not read twice.
 */
export function promptsToMessages(res: unknown, idPrefix: string, time: string, alreadyShown?: string): PromptMessage[] {
  const shown = (alreadyShown ?? "").trim();
  return readAssistantPrompts(res).map((p, i) => ({
    id: `${idPrefix}-prompt-${i}-${p.step}`,
    role: "ai" as const,
    content:
      shown && p.message === shown
        ? p.question ?? p.message
        : p.question
          ? `${p.message}\n\n${p.question}`
          : p.message,
    time,
    step: p.step,
    quickReplies: p.quick_replies && p.quick_replies.length ? p.quick_replies : undefined,
    link: p.link,
    segments: promptSegments(p, !!shown && p.message === shown),
  }));
}

/** The bubble's segments: the message's, then the question (a template's) as plain. */
function promptSegments(p: AssistantPrompt, messageAlreadyShown: boolean): ReplySegment[] | undefined {
  if (!p.message_segments) return undefined;
  if (messageAlreadyShown) return p.question ? undefined : p.message_segments;
  return p.question ? [...p.message_segments, { kind: "plain", text: `\n\n${p.question}` }] : p.message_segments;
}
