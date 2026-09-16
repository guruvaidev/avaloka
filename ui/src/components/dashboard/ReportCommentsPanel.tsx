import { X, Attachment01 } from "@untitledui/icons";
import { useCallback, useEffect, useRef, useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { getCurrentProfileId } from "@/lib/current-profile";
import {
  formatInsightCommentRelative,
  initialsFromName,
  type InsightCommentAttachment,
} from "@/lib/analysis-insight-comments";
import {
  addReportComment,
  loadReportComments,
  uploadReportCommentAttachment,
  type ReportComment,
} from "@/lib/report-comments";
import {
  insightCommentActionText,
  InsightCommentAttachmentList,
} from "@/components/dashboard/InsightCommentAttachments";
import { toast } from "sonner";

export function ReportCommentsPanel({
  onClose,
  reportId,
  open = true,
}: {
  onClose: () => void;
  reportId?: string | null;
  open?: boolean;
}) {
  const [comments, setComments] = useState<ReportComment[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentProfileId, setCurrentProfileId] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const reload = useCallback(() => {
    if (!reportId) return;
    setLoading(true);
    setError(null);
    Promise.all([loadReportComments(reportId), getCurrentProfileId()])
      .then(([rows, pid]) => {
        setComments(rows);
        setCurrentProfileId(pid);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, [reportId]);

  useEffect(() => {
    if (!reportId || !open) return;
    reload();
  }, [reportId, open, reload]);

  const handleSend = async () => {
    const text = message.trim();
    if ((!text && pendingFiles.length === 0) || sending || !reportId) return;
    setSending(true);
    try {
      const uploaded: InsightCommentAttachment[] = [];
      for (const f of pendingFiles) uploaded.push(await uploadReportCommentAttachment(reportId, f));
      const content = text || (uploaded.length ? "Shared an attachment." : "");
      const created = await addReportComment(reportId, content, uploaded);
      setComments((prev) => [created, ...prev]);
      setMessage("");
      setPendingFiles([]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to send comment");
    } finally {
      setSending(false);
    }
  };

  const flat = comments.flatMap((c) => [
    {
      id: c.id,
      name: c.author?.id === currentProfileId ? "You" : c.author?.full_name ?? "Unknown user",
      initials: initialsFromName(c.author?.full_name) ?? "?",
      avatarUrl: c.author?.avatar_url ?? null,
      time: formatInsightCommentRelative(c.created_at),
      text: c.text,
      attachments: c.attachments,
      isReply: false,
    },
    ...c.replies.map((r) => ({
      id: r.id,
      name: r.author?.id === currentProfileId ? "You" : r.author?.full_name ?? "Unknown user",
      initials: initialsFromName(r.author?.full_name) ?? "?",
      avatarUrl: r.author?.avatar_url ?? null,
      time: formatInsightCommentRelative(r.created_at),
      text: r.text,
      attachments: r.attachments,
      isReply: true,
    })),
  ]);

  return (
    <aside className="flex w-[380px] flex-col overflow-hidden rounded-xl border border-secondary bg-primary shadow-lg">
      <div className="flex items-center justify-between border-b border-secondary px-5 py-3.5">
        <h2 className="text-md font-semibold text-primary">Comments</h2>
        <button
          onClick={onClose}
          aria-label="Close comments"
          className="rounded-md p-1 text-fg-quaternary hover:bg-primary_hover"
        >
          <X className="size-5" />
        </button>
      </div>

      <div className="flex max-h-[360px] flex-col overflow-y-auto px-4 py-3">
        {!reportId ? (
          <p className="text-sm text-tertiary">No report selected.</p>
        ) : loading ? (
          <p className="text-sm text-tertiary">Loading comments…</p>
        ) : error ? (
          <p className="text-sm text-[#d92d20]">{error}</p>
        ) : flat.length === 0 ? (
          <p className="text-sm text-tertiary">No comments yet for this report.</p>
        ) : (
          <ol className="relative">
            {flat.map((item, i) => {
              const actionText = insightCommentActionText(item.text, item.attachments.length);
              const showBubble = item.attachments.length === 0 || actionText !== "Added a file";
              return (
                <li key={item.id} className={"relative flex gap-3 pb-5 " + (item.isReply ? "pl-6" : "")}>
                  {i < flat.length - 1 && (
                    <span
                      className="absolute left-[18px] top-9 bottom-0 border-l border-dashed border-border-secondary"
                      aria-hidden
                    />
                  )}
                  <Avatar className="size-9">
                    <AvatarImage src={item.avatarUrl ?? undefined} alt={item.name} className="object-cover" />
                    <AvatarFallback className="bg-[#1565ef]/10 text-xs font-semibold text-[#1565ef]">
                      {item.initials}
                    </AvatarFallback>
                  </Avatar>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-sm font-semibold text-primary truncate">{item.name}</span>
                      <span className="shrink-0 text-[11px] text-tertiary">{item.time}</span>
                    </div>
                    {showBubble ? (
                      <p className="mt-0.5 text-sm text-secondary whitespace-pre-wrap break-words">
                        {item.isReply ? "↳ " : ""}
                        {actionText}
                      </p>
                    ) : (
                      <p className="mt-0.5 text-sm text-secondary">{actionText}</p>
                    )}
                    <InsightCommentAttachmentList attachments={item.attachments} />
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </div>

      <div className="border-t border-secondary p-3">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => {
            const list = e.target.files;
            if (list?.length) setPendingFiles((p) => [...p, ...Array.from(list)].slice(0, 5));
            e.target.value = "";
          }}
        />
        <div className="rounded-lg border border-secondary bg-primary">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void handleSend();
              }
            }}
            placeholder="Write a comment… (Enter to send)"
            rows={2}
            className="min-h-[44px] w-full resize-none bg-transparent px-3 pt-2 text-sm text-primary placeholder:text-tertiary focus:outline-none"
          />
          {pendingFiles.length > 0 && (
            <div className="flex flex-wrap gap-1.5 px-3 pb-1.5">
              {pendingFiles.map((f, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 rounded border border-secondary bg-secondary/30 px-2 py-0.5 text-xs text-primary"
                >
                  <Attachment01 className="size-3" />
                  <span className="max-w-[140px] truncate">{f.name}</span>
                  <button
                    onClick={() => setPendingFiles((p) => p.filter((_, j) => j !== i))}
                    className="text-fg-quaternary hover:text-fg-secondary"
                    aria-label="Remove"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="flex items-center justify-between px-3 pb-2">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="text-fg-quaternary hover:text-fg-secondary"
              aria-label="Attach"
            >
              <Attachment01 className="size-4" />
            </button>
            {sending ? <span className="text-xs text-tertiary">Sending…</span> : null}
          </div>
        </div>
      </div>
    </aside>
  );
}
