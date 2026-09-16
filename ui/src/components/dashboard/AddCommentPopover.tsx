import { useCallback, useEffect, useRef, useState } from "react";
import { Attachment01, FaceSmile, XClose } from "@untitledui/icons";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { getCurrentProfileId } from "@/lib/current-profile";
import {
  InsightCommentAttachmentList,
  insightCommentActionText,
} from "@/components/dashboard/InsightCommentAttachments";
import {
  addAnalysisInsightComment,
  addAnalysisInsightCommentReply,
  formatAttachmentSize,
  formatInsightCommentTime,
  initialsFromName,
  loadAnalysisInsightComments,
  MAX_INSIGHT_COMMENT_ATTACHMENTS,
  MAX_INSIGHT_COMMENT_ATTACHMENT_BYTES,
  uploadInsightCommentAttachment,
  type AnalysisInsightComment,
  type InsightCommentAttachment,
} from "@/lib/analysis-insight-comments";
import {
  loadReportComments,
  addReportComment,
  addReportCommentReply,
  uploadReportCommentAttachment,
} from "@/lib/report-comments";
import { toast } from "sonner";
import {
  MentionTextInput,
  useMentionCandidates,
  extractMentionedProfileIds,
} from "@/components/dashboard/MentionTextInput";
import { cx } from "@/lib/utils/cx";

type CardProps = {
  analysisId: string | null;
  /** When provided, comments are stored in report_comments instead of analysis_comments. */
  reportId?: string | null;
  /** Bump to reload comments after the sidebar panel adds one. */
  refreshKey?: number;
};


function CommentBubble({
  name,
  initials,
  avatarUrl,
  time,
  text,
  attachments = [],
  nested,
  footer,
}: {
  name: string;
  initials: string;
  avatarUrl: string | null;
  time: string;
  text: string;
  attachments?: InsightCommentAttachment[];
  nested?: boolean;
  footer?: React.ReactNode;
}) {
  const actionText = insightCommentActionText(text, attachments.length);
  const showTextBubble = attachments.length === 0 || actionText !== "Added a file";

  return (
    <div className={cx("flex items-start gap-2.5", nested && "ml-9 mt-2")}>
      <Avatar className="size-7">
        <AvatarImage src={avatarUrl ?? undefined} alt={name} className="object-cover" />
        <AvatarFallback className="bg-[#1565ef]/10 text-[10px] font-semibold text-[#1565ef]">
          {initials}
        </AvatarFallback>
      </Avatar>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-xs font-semibold text-primary">{name}</span>
          <span className="shrink-0 text-[11px] text-tertiary">{time}</span>
        </div>
        {showTextBubble ? (
          <div className="mt-1.5 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary">
            {actionText}
          </div>
        ) : (
          <p className="mt-1 text-sm text-secondary">{actionText}</p>
        )}
        <InsightCommentAttachmentList attachments={attachments} />
        {footer}
      </div>
    </div>
  );
}

function PendingAttachments({ files, onRemove }: { files: File[]; onRemove: (index: number) => void }) {
  if (!files.length) return null;

  return (
    <div className="flex flex-wrap gap-2 px-3 pt-2">
      {files.map((file, index) => (
        <span
          key={`${file.name}-${index}`}
          className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-secondary bg-secondary/30 px-2 py-1 text-xs text-primary"
        >
          <Attachment01 className="size-3.5 shrink-0 text-fg-quaternary" />
          <span className="truncate">{file.name}</span>
          <span className="shrink-0 text-tertiary">{formatAttachmentSize(file.size)}</span>
          <button
            type="button"
            aria-label={`Remove ${file.name}`}
            onClick={() => onRemove(index)}
            className="shrink-0 text-fg-quaternary hover:text-fg-secondary"
          >
            <XClose className="size-3.5" />
          </button>
        </span>
      ))}
    </div>
  );
}

/** Full-width insight comment thread + composer. */
export function InsightCommentCard({ analysisId, reportId = null, refreshKey = 0, hideThread = false }: CardProps & { hideThread?: boolean }) {
  const useReport = !!reportId;
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [comments, setComments] = useState<AnalysisInsightComment[]>([]);
  const [currentProfileId, setCurrentProfileId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [replyingToId, setReplyingToId] = useState<string | null>(null);
  const [replyDraft, setReplyDraft] = useState("");
  const [replySending, setReplySending] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [replyPendingFiles, setReplyPendingFiles] = useState<File[]>([]);
  const mentionPeople = useMentionCandidates();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const replyFileInputRef = useRef<HTMLInputElement>(null);

  const reload = useCallback(() => {
    if (useReport) {
      if (!reportId) { setComments([]); return; }
      setLoading(true);
      Promise.all([loadReportComments(reportId), getCurrentProfileId()])
        .then(([rows, profileId]) => { setComments(rows); setCurrentProfileId(profileId); })
        .catch((err) => console.error("[InsightCommentCard] load failed", err))
        .finally(() => setLoading(false));
      return;
    }
    if (!analysisId) { setComments([]); return; }
    setLoading(true);
    Promise.all([loadAnalysisInsightComments(analysisId), getCurrentProfileId()])
      .then(([rows, profileId]) => { setComments(rows); setCurrentProfileId(profileId); })
      .catch((err) => console.error("[InsightCommentCard] load failed", err))
      .finally(() => setLoading(false));
  }, [analysisId, reportId, useReport]);

  useEffect(() => {
    reload();
  }, [reload, refreshKey]);

  const pickFiles = (fileList: FileList | null, target: "main" | "reply") => {
    if (!fileList?.length) return;
    const next = Array.from(fileList);
    const invalid = next.find((f) => f.size > MAX_INSIGHT_COMMENT_ATTACHMENT_BYTES);
    if (invalid) {
      toast.error(`${invalid.name} is too large. Maximum size is 25MB.`);
      return;
    }
    const setter = target === "main" ? setPendingFiles : setReplyPendingFiles;
    setter((prev) => {
      const merged = [...prev, ...next].slice(0, MAX_INSIGHT_COMMENT_ATTACHMENTS);
      if (merged.length >= MAX_INSIGHT_COMMENT_ATTACHMENTS) {
        toast.message(`Maximum ${MAX_INSIGHT_COMMENT_ATTACHMENTS} attachments per comment.`);
      }
      return merged;
    });
  };

  const uploadPending = async (files: File[]) => {
    if (!files.length) return [];
    const uploaded: InsightCommentAttachment[] = [];
    for (const file of files) {
      if (useReport) {
        if (!reportId) return [];
        uploaded.push(await uploadReportCommentAttachment(reportId, file));
      } else {
        if (!analysisId) return [];
        uploaded.push(await uploadInsightCommentAttachment(analysisId, file));
      }
    }
    return uploaded;
  };

  const handleSend = async () => {
    const text = message.trim();
    if ((!text && pendingFiles.length === 0) || sending) return;
    if (useReport ? !reportId : !analysisId) {
      toast.error("Save this first to add comments.");
      return;
    }
    setSending(true);
    try {
      const attachments = await uploadPending(pendingFiles);
      const content = text || (attachments.length ? "Shared an attachment." : "");
      const mentionedProfileIds = extractMentionedProfileIds(content, mentionPeople);
      const created = useReport
        ? await addReportComment(reportId!, content, attachments, mentionedProfileIds)
        : await addAnalysisInsightComment(analysisId!, content, attachments, mentionedProfileIds);
      setComments((prev) => [created, ...prev]);
      setMessage("");
      setPendingFiles([]);
    } catch (err) {
      console.error("[InsightCommentCard] send failed", err);
      toast.error(err instanceof Error ? err.message : "Failed to send comment");
    } finally {
      setSending(false);
    }
  };

  const handleReply = async (parentId: string) => {
    const text = replyDraft.trim();
    const activeId = useReport ? reportId : analysisId;
    if ((!text && replyPendingFiles.length === 0) || replySending || !activeId) return;
    setReplySending(true);
    try {
      const attachments = await uploadPending(replyPendingFiles);
      const content = text || (attachments.length ? "Shared an attachment." : "");
      const mentionedProfileIds = extractMentionedProfileIds(content, mentionPeople);
      const reply = useReport
        ? await addReportCommentReply(reportId!, parentId, content, attachments, mentionedProfileIds)
        : await addAnalysisInsightCommentReply(analysisId!, parentId, content, attachments, mentionedProfileIds);
      setComments((prev) => prev.map((c) => (c.id === parentId ? { ...c, replies: [...c.replies, reply] } : c)));
      setReplyDraft("");
      setReplyPendingFiles([]);
      setReplyingToId(null);
    } catch (err) {
      console.error("[InsightCommentCard] reply failed", err);
      toast.error(err instanceof Error ? err.message : "Failed to send reply");
    } finally {
      setReplySending(false);
    }
  };


  const canSend = Boolean(message.trim() || pendingFiles.length) && !sending;

  return (
    <div className="w-full rounded-xl border border-secondary bg-primary">
      {!hideThread && loading && comments.length === 0 ? <p className="px-4 py-3 text-sm text-tertiary">Loading comments…</p> : null}

      {!hideThread && comments.length > 0 ? (
        <div className="flex flex-col gap-4 px-4 pt-4">
          {comments.map((c) => (
            <div key={c.id}>
              <CommentBubble
                name={c.author?.id === currentProfileId ? "You" : c.author?.full_name ?? "Unknown user"}
                initials={initialsFromName(c.author?.full_name) ?? "?"}
                avatarUrl={c.author?.avatar_url ?? null}
                time={formatInsightCommentTime(c.created_at)}
                text={c.text}
                attachments={c.attachments}
                footer={
                  <button
                    type="button"
                    onClick={() => {
                      setReplyingToId((id) => (id === c.id ? null : c.id));
                      setReplyDraft("");
                      setReplyPendingFiles([]);
                    }}
                    className="mt-1 text-xs text-tertiary hover:text-secondary"
                  >
                    {replyingToId === c.id ? "Cancel" : "Reply"}
                  </button>
                }
              />

              {c.replies.map((r) => (
                <CommentBubble
                  key={r.id}
                  nested
                  name={r.author?.id === currentProfileId ? "You" : r.author?.full_name ?? "Unknown user"}
                  initials={initialsFromName(r.author?.full_name) ?? "?"}
                  avatarUrl={r.author?.avatar_url ?? null}
                  time={formatInsightCommentTime(r.created_at)}
                  text={r.text}
                  attachments={r.attachments}
                />
              ))}

              {replyingToId === c.id && (
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void handleReply(c.id);
                  }}
                  className="ml-9 mt-2 rounded-lg border border-secondary bg-primary"
                >
                  <div className="flex items-center gap-2 px-3 py-2">
                    <MentionTextInput
                      value={replyDraft}
                      onChange={setReplyDraft}
                      people={mentionPeople}
                      multiline={false}
                      onEnterSend={() => void handleReply(c.id)}
                      placeholder="Write a reply… (@ to mention)"
                      className="w-full bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
                    />
                    <button
                      type="button"
                      aria-label="Attach file to reply"
                      onClick={() => replyFileInputRef.current?.click()}
                      className="cursor-pointer text-fg-quaternary hover:text-fg-secondary"
                    >
                      <Attachment01 className="size-4" />
                    </button>
                  </div>
                  <PendingAttachments
                    files={replyPendingFiles}
                    onRemove={(index) => setReplyPendingFiles((prev) => prev.filter((_, i) => i !== index))}
                  />
                  <div className="flex justify-end px-3 pb-2">
                    <button
                      type="submit"
                      disabled={(!replyDraft.trim() && replyPendingFiles.length === 0) || replySending}
                      className="text-sm font-semibold text-[#1565ef] disabled:opacity-50"
                    >
                      {replySending ? "Sending…" : "Send"}
                    </button>
                  </div>
                </form>
              )}
            </div>
          ))}
        </div>
      ) : null}

      <div className={cx("p-4", !hideThread && comments.length > 0 && "border-t border-secondary")}>
        <div className="w-full rounded-lg border border-secondary bg-primary">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              pickFiles(e.target.files, "main");
              e.target.value = "";
            }}
          />
          <input
            ref={replyFileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              pickFiles(e.target.files, "reply");
              e.target.value = "";
            }}
          />
          <div className="flex items-start gap-2 px-3 pt-2.5">
            <MentionTextInput
              value={message}
              onChange={setMessage}
              people={mentionPeople}
              onEnterSend={() => {
                if (canSend) void handleSend();
              }}
              placeholder="Write a comment… (@ to mention, Enter to send)"
              rows={2}
              className="min-h-[44px] w-full resize-none border-0 bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
            />
          </div>
          <PendingAttachments
            files={pendingFiles}
            onRemove={(index) => setPendingFiles((prev) => prev.filter((_, i) => i !== index))}
          />
          <div className="flex items-center justify-between px-3 pb-2.5">
            <div className="flex items-center gap-3 text-fg-quaternary">
              <button
                type="button"
                aria-label="Attach"
                onClick={() => fileInputRef.current?.click()}
                className="cursor-pointer hover:text-fg-secondary"
              >
                <Attachment01 className="size-4" />
              </button>
              <button type="button" aria-label="Emoji" className="cursor-pointer hover:text-fg-secondary">
                <FaceSmile className="size-4" />
              </button>
            </div>
            {sending ? <span className="text-xs text-tertiary">Sending…</span> : null}
          </div>
        </div>
      </div>
    </div>
  );
}

type PopoverProps = {
  trigger: React.ReactNode;
  analysisId: string | null;
  onOpenChange?: (open: boolean) => void;
};

/** Popover wrapper — used when a trigger button opens the card. */
export function AddCommentPopover({ trigger, analysisId, onOpenChange }: PopoverProps) {
  const [open, setOpen] = useState(false);

  const handleOpenChange = useCallback(
    (next: boolean) => {
      setOpen(next);
      onOpenChange?.(next);
    },
    [onOpenChange],
  );

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        sideOffset={12}
        className="w-[min(100vw-2rem,640px)] border-0 bg-transparent p-0 shadow-none"
      >
        <InsightCommentCard analysisId={analysisId} hideThread />
      </PopoverContent>
    </Popover>
  );
}
