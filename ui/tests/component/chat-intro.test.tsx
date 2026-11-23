import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { useState } from "react";
import { ChatIntroCard, useChatIntro } from "../../src/components/analysis/ChatIntroCard";
import { chatbotIntro, isNewChatThread } from "../../src/lib/chatbot-intro";

beforeEach(() => localStorage.clear());
afterEach(() => localStorage.clear());

// Mirrors the analysis page: the intro shows while it is not dismissed and the
// thread has no user message; sending the first message dismisses it.
function Chat({ userId, threadId }: { userId: string | null; threadId: string | null }) {
  const intro = useChatIntro(userId, threadId);
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const show = !intro.dismissed && isNewChatThread(messages);
  return (
    <div>
      {show && <ChatIntroCard onDismiss={intro.dismiss} />}
      <button
        type="button"
        onClick={() => {
          if (!intro.dismissed) intro.dismiss();
          setMessages((m) => [
            ...m,
            { role: "user", content: "hi" },
            { role: "ai", content: "plan" },
          ]);
        }}
      >
        send
      </button>
      <span data-testid="count">{messages.length}</span>
    </div>
  );
}

describe("chat intro", () => {
  it("renders the intro once on a new chat", () => {
    render(<Chat userId="u1" threadId="t1" />);
    expect(screen.getAllByTestId("chat-intro")).toHaveLength(1);
    expect(screen.getByTestId("chat-intro").textContent).toContain(chatbotIntro());
  });

  it("is dismissible and stays dismissed for that user and thread", () => {
    const { unmount } = render(<Chat userId="u1" threadId="t1" />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss intro" }));
    expect(screen.queryByTestId("chat-intro")).toBeNull();
    unmount();
    render(<Chat userId="u1" threadId="t1" />);
    expect(screen.queryByTestId("chat-intro")).toBeNull();
  });

  it("a different thread gets its own intro", () => {
    const { unmount } = render(<Chat userId="u1" threadId="t1" />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss intro" }));
    unmount();
    render(<Chat userId="u1" threadId="t2" />);
    expect(screen.getAllByTestId("chat-intro")).toHaveLength(1);
  });

  it("is not re-shown on new messages, nor after a reload of the thread", () => {
    const { unmount } = render(<Chat userId="u1" threadId="t1" />);
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    expect(screen.getByTestId("count").textContent).toBe("4");
    expect(screen.queryByTestId("chat-intro")).toBeNull();
    unmount();
    render(<Chat userId="u1" threadId="t1" />);
    expect(screen.queryByTestId("chat-intro")).toBeNull();
  });

  it("a draft chat that gets its thread keeps the dismissal", () => {
    const { result, rerender } = renderHook(({ t }) => useChatIntro("u1", t), {
      initialProps: { t: null as string | null },
    });
    act(() => result.current.dismiss());
    rerender({ t: "t9" });
    expect(result.current.dismissed).toBe(true);
    expect(localStorage.getItem("avaloka.chatIntro.dismissed:u1:t9")).toBe("true");
  });
});
