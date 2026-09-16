import { useState } from "react";
import { X, Paperclip, FaceSmile } from "@untitledui/icons";

const RecordingIcon = ({ className }: { className?: string }) => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className={className}>
    <path d="M2.5 8.33333L2.5 11.6667M6.25 5L6.25 15M10 2.5V17.5M13.75 5V15M17.5 8.33333V11.6667" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

export function CommentPopover({ onClose }: { onClose: () => void }) {
  const [text, setText] = useState("");
  return (
    <div
      role="dialog"
      aria-label="Comment"
      className="absolute bottom-10 left-0 z-30 w-[320px] rounded-2xl border border-secondary bg-primary p-4 shadow-lg"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-md font-semibold text-primary">Comment</h3>
        <button onClick={onClose} aria-label="Close" className="rounded p-1 text-fg-quaternary hover:bg-primary_hover">
          <X className="size-4" />
        </button>
      </div>

      <div className="mt-3 flex gap-2">
        <div className="relative shrink-0">
          <span className="grid size-8 place-items-center rounded-full bg-[#1565EF]/10 text-[10px] font-semibold text-[#1565EF]">AI</span>
          <span className="absolute -bottom-0 -right-0 size-2 rounded-full border border-white bg-[#17b26a]" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-semibold text-primary">Avaloka AI</span>
            <span className="text-[11px] text-tertiary">Friday 2:20pm</span>
          </div>
          <div className="mt-1.5 rounded-lg border border-secondary bg-secondary/40 px-3 py-2 text-sm text-primary">
            AI Insights generated for the given data set
          </div>
        </div>
      </div>

      <div className="mt-3 rounded-xl border border-secondary bg-primary p-3 shadow-xs">
        <div className="flex items-start gap-2">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Message"
            rows={2}
            className="flex-1 resize-none bg-transparent text-sm text-primary placeholder:text-tertiary outline-none"
          />
          <button aria-label="Voice" className="text-fg-quaternary hover:text-secondary">
            <RecordingIcon className="size-4" />
          </button>
        </div>
        <div className="mt-2 flex items-center justify-end gap-3 text-fg-quaternary">
          <button aria-label="Attach" className="hover:text-secondary"><Paperclip className="size-4" /></button>
          <button aria-label="Emoji" className="hover:text-secondary"><FaceSmile className="size-4" /></button>
          <button
            type="button"
            disabled={!text.trim()}
            className={text.trim() ? "text-sm font-semibold text-[#1565EF]" : "text-sm font-semibold text-tertiary"}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
