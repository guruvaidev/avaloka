// Chatbot intro: the first thing the chat says when a thread opens.
//
// Every word here is code-written, so it must be TRUE of the release line it
// ships on (docs/design/1.6-CHATBOT-INTRO.md holds the per-claim evidence).
// Later lines add sentences only once the feature behind them is wired into
// the product; tests/unit/chatbot-intro.test.ts pins the text per line.

export type ReleaseLine = "1.6";

export const CHATBOT_RELEASE_LINE: ReleaseLine = "1.6";

export const CHATBOT_INTRO: Record<ReleaseLine, string> = {
  "1.6":
    "Hi, I'm Avaloka, a data scientist you can talk to. Tell me your goal in plain English: " +
    "analyse this data, train a model on it, or look at a trend over time. I'll find the data " +
    "sources you've connected, suggest how your datasets join, plan the steps, move and analyse " +
    "the data, and test my code on a sample of your data before I run it. What would you like " +
    "to achieve?",
};

export function chatbotIntro(line: ReleaseLine = CHATBOT_RELEASE_LINE): string {
  return CHATBOT_INTRO[line];
}

export const CHAT_INTRO_STORAGE_PREFIX = "avaloka.chatIntro.dismissed";

type StorageLike = Pick<Storage, "getItem" | "setItem">;

function defaultStorage(): StorageLike | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

/** One key per user and thread; a chat with no thread yet is the "draft" thread. */
export function chatIntroKey(
  userId: string | null | undefined,
  threadId: string | null | undefined,
): string {
  return `${CHAT_INTRO_STORAGE_PREFIX}:${userId || "anon"}:${threadId || "draft"}`;
}

/** True once the intro was dismissed (or the first goal was sent) in this thread. */
export function readIntroDismissed(
  userId: string | null | undefined,
  threadId: string | null | undefined,
  storage?: StorageLike | null,
): boolean {
  try {
    return (storage ?? defaultStorage())?.getItem(chatIntroKey(userId, threadId)) === "true";
  } catch {
    return false;
  }
}

/** Persists the dismissal; storage failures keep the in-memory choice only. */
export function writeIntroDismissed(
  userId: string | null | undefined,
  threadId: string | null | undefined,
  storage?: StorageLike | null,
): void {
  try {
    (storage ?? defaultStorage())?.setItem(chatIntroKey(userId, threadId), "true");
  } catch {
    /* storage unavailable: the intro stays hidden for this page load */
  }
}

/** A thread is new until its first user message; page-posted summaries do not count. */
export function isNewChatThread(messages: ReadonlyArray<{ role: string }>): boolean {
  return !messages.some((m) => m.role === "user");
}
