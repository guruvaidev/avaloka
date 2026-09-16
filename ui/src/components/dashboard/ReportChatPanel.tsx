import { useState } from "react";
import { Plus, Lightning01, MessageChatCircle, Edit01, RefreshCw01, ThumbsUp, ThumbsDown, X } from "@untitledui/icons";

const RecordingIcon = ({ className }: { className?: string }) => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    <path d="M2.5 8.33333L2.5 11.6667M6.25 5L6.25 15M10 2.5V17.5M13.75 5V15M17.5 8.33333V11.6667" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);
import { cx } from "@/lib/utils/cx";

type ChatMessage =
  | { id: string; role: "user"; time: string; text?: string; tag?: string; sub?: string; read?: boolean }
  | { id: string; role: "assistant"; time: string; name: string; text: string };

const SEED: ChatMessage[] = [
  { id: "m1", role: "user", time: "Thursday 11:41am", text: "Awesome! Thanks.", read: true },
  {
    id: "m2",
    role: "assistant",
    time: "Friday 2:20pm",
    name: "Avaloka AI",
    text: "AI Insights generated for the given data set",
  },
  {
    id: "m3",
    role: "user",
    time: "Friday 2:20pm",
    tag: "Insights",
    sub: "Trend report for the given data set",
    read: true,
  },
];

export function ReportChatPanel() {
  const [messages] = useState<ChatMessage[]>(SEED);
  const [input, setInput] = useState("");
  const [chips, setChips] = useState<string[]>(["Data Set 1"]);

  return (
    <aside className="flex h-full w-[360px] shrink-0 flex-col overflow-hidden border-l border-secondary bg-primary">
      <div className="flex items-center justify-between px-5 py-4">
        <h2 className="text-md font-semibold text-primary">Chat</h2>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-2">
        <div className="my-3 text-center text-xs text-tertiary">Today</div>
        <ul className="flex flex-col gap-5">
          {messages.map((m) =>
            m.role === "user" ? (
              <li key={m.id} className="flex flex-col items-end gap-1">
                <div className="flex items-center gap-1.5 text-[11px] text-tertiary">
                  <span>You</span>
                  <span>{m.time}</span>
                  {m.read && <span className="text-[#1565EF]">✓</span>}
                </div>
                {m.text && (
                  <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-[#1565EF] px-3.5 py-2 text-sm text-white">
                    {m.text}
                  </div>
                )}
                {m.tag && (
                  <div className="flex max-w-[85%] flex-col gap-1.5 rounded-2xl rounded-tr-sm border border-secondary bg-secondary/40 px-3 py-2">
                    <span className="inline-flex w-fit items-center gap-1 rounded-md bg-primary px-2 py-0.5 text-xs font-medium text-[#1565EF] ring-1 ring-inset ring-secondary">
                      <MessageChatCircle className="size-3" />
                      {m.tag}
                    </span>
                    {m.sub && <p className="text-sm text-primary">{m.sub}</p>}
                  </div>
                )}
              </li>
            ) : (
              <li key={m.id} className="flex flex-col gap-1.5">
                <div className="flex items-center gap-2">
                  <span className="grid size-7 place-items-center rounded-full bg-[#1565EF]/10 text-[10px] font-semibold text-[#1565EF]">
                    AI
                  </span>
                  <span className="text-sm font-semibold text-primary">{m.name}</span>
                  <span className="ml-auto text-[11px] text-tertiary">{m.time}</span>
                </div>
                <div className="ml-9 max-w-[85%] rounded-2xl rounded-tl-sm border border-secondary bg-secondary/40 px-3.5 py-2 text-sm text-primary">
                  {m.text}
                </div>
                <div className="ml-9 mt-1 flex items-center gap-3 text-fg-quaternary">
                  <button aria-label="Edit" className="hover:text-secondary"><Edit01 className="size-4" /></button>
                  <button aria-label="Refresh" className="hover:text-secondary"><RefreshCw01 className="size-4" /></button>
                  <button aria-label="Thumbs up" className="hover:text-secondary"><ThumbsUp className="size-4" /></button>
                  <button aria-label="Thumbs down" className="hover:text-secondary"><ThumbsDown className="size-4" /></button>
                </div>
              </li>
            ),
          )}
        </ul>
      </div>

      <div className="border-t border-secondary p-3">
        <div className="rounded-xl border border-secondary bg-primary p-3 shadow-xs">
          <div className="flex items-start gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me anything..."
              rows={2}
              className="flex-1 resize-none bg-transparent text-sm text-primary placeholder:text-tertiary outline-none"
            />
            <button aria-label="Voice" className="text-fg-quaternary hover:text-secondary">
              <RecordingIcon className="size-5" />
            </button>
          </div>
          <div className="mt-2 flex justify-end">
            <button
              type="button"
              disabled={!input.trim()}
              className={cx(
                "text-sm font-semibold",
                input.trim() ? "text-[#1565EF]" : "text-tertiary",
              )}
            >
              Send
            </button>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          {chips.map((c) => (
            <span
              key={c}
              className="inline-flex items-center gap-1.5 rounded-md bg-[#1565EF]/10 px-2 py-1 font-medium text-[#1565EF]"
            >
              <span className="size-1.5 rounded-full bg-[#1565EF]" />
              {c}
              <button
                aria-label={`Remove ${c}`}
                onClick={() => setChips((prev) => prev.filter((x) => x !== c))}
                className="ml-0.5 text-[#1565EF]/70 hover:text-[#1565EF]"
              >
                <X className="size-3" />
              </button>
            </span>
          ))}
          <button className="inline-flex items-center gap-1 rounded-md px-2 py-1 font-medium text-secondary hover:bg-secondary/40">
            <Lightning01 className="size-3.5 text-[#1565EF]" />
            Connect Cloud
          </button>
          <button className="inline-flex items-center gap-1 rounded-md px-2 py-1 font-medium text-secondary hover:bg-secondary/40">
            <Plus className="size-3.5" />
            Add
          </button>
        </div>
      </div>
    </aside>
  );
}
