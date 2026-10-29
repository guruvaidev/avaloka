import { useState, type ReactNode } from "react";
import { Edit01 as Edit01Icon } from "@untitledui/icons";

/** Inline editor for a user's own message: Save re-sends the edited text through the existing send function. */
export function UserMessageEditor({
  content,
  onResend,
  children,
}: {
  content: string;
  onResend?: (text: string) => void;
  children: (controls: { editButton: ReactNode; editing: boolean }) => ReactNode;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(content);
  const save = () => {
    const text = draft.trim();
    if (!text) return;
    setEditing(false);
    onResend?.(text);
  };
  const editButton = onResend ? (
    <button
      type="button"
      aria-label="Edit"
      onClick={() => {
        setDraft(content);
        setEditing(true);
      }}
      className="hover:text-fg-secondary"
    >
      <Edit01Icon className="size-3.5" />
    </button>
  ) : null;
  if (editing) {
    return (
      <div className="mt-1 flex w-full max-w-[85%] flex-col gap-2 self-end">
        <textarea
          aria-label="Edit message"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              save();
            }
            if (e.key === "Escape") setEditing(false);
          }}
          rows={3}
          autoFocus
          className="w-full resize-y rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-primary"
        />
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="rounded-md border border-border bg-card px-3 py-1 text-xs font-semibold text-foreground hover:bg-muted"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={save}
            disabled={!draft.trim()}
            className="rounded-md bg-primary px-3 py-1 text-xs font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
    );
  }
  return <>{children({ editButton, editing })}</>;
}
