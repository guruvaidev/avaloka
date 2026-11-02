import { useCallback, useEffect, useRef, useState } from "react";
import { X } from "@untitledui/icons";
import { chatbotIntro, readIntroDismissed, writeIntroDismissed } from "@/lib/chatbot-intro";

/**
 * Whether the intro was dismissed for this user and thread. A draft chat (no
 * thread yet) or a profile that has not loaded yet carries its dismissal over
 * to the key it resolves to, so the intro never comes back mid-session.
 */
export function useChatIntro(
  userId: string | null | undefined,
  threadId: string | null | undefined,
) {
  const [dismissed, setDismissed] = useState(() => readIntroDismissed(userId, threadId));
  const prev = useRef({ userId, threadId, dismissed });
  prev.current.dismissed = dismissed;

  useEffect(() => {
    const was = prev.current;
    if (was.userId === userId && was.threadId === threadId) return;
    const carried = was.dismissed && (!was.threadId || !was.userId);
    prev.current = { userId, threadId, dismissed: was.dismissed };
    if (carried) {
      writeIntroDismissed(userId, threadId);
      return;
    }
    setDismissed(readIntroDismissed(userId, threadId));
  }, [userId, threadId]);

  const dismiss = useCallback(() => {
    setDismissed(true);
    writeIntroDismissed(userId, threadId);
  }, [userId, threadId]);

  return { dismissed, dismiss };
}

/** The code-written intro shown once at the top of a new chat. */
export function ChatIntroCard({ onDismiss }: { onDismiss: () => void }) {
  return (
    <section
      aria-label="About Avaloka"
      data-testid="chat-intro"
      className="mb-5 flex items-start gap-2 rounded-lg border border-[#b2ddff] bg-[#f5faff] px-3 py-2.5"
    >
      <p className="flex-1 text-sm text-primary">{chatbotIntro()}</p>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss intro"
        title="Dismiss intro"
        className="grid size-6 shrink-0 cursor-pointer place-items-center rounded-md text-fg-quaternary hover:bg-secondary hover:text-fg-secondary"
      >
        <X className="size-4" />
      </button>
    </section>
  );
}
