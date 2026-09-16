import { X } from "@untitledui/icons";
import { useEffect, useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { getCurrentProfileId } from "@/lib/current-profile";
import {
  formatInsightCommentRelative,
  initialsFromName,
  loadAnalysisInsightComments,
  type AnalysisInsightComment,
  type InsightCommentAttachment,
} from "@/lib/analysis-insight-comments";
import {
  insightCommentActionText,
  InsightCommentAttachmentList,
} from "@/components/dashboard/InsightCommentAttachments";

type FlatComment = {
  id: string;
  name: string;
  initials: string;
  time: string;
  text: string;
  attachments: InsightCommentAttachment[];
  avatarUrl: string | null;
  isReply: boolean;
};

export function CommentsPanel({
  onClose,
  analysisId,
  open = true,
}: {
  onClose: () => void;
  analysisId?: string | null;
  /** When false, skip loading until the panel is shown. */
  open?: boolean;
}) {
  const [comments, setComments] = useState<AnalysisInsightComment[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentProfileId, setCurrentProfileId] = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId || !open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([loadAnalysisInsightComments(analysisId), getCurrentProfileId()])
      .then(([rows, profileId]) => {
        if (!cancelled) {
          setComments(rows);
          setCurrentProfileId(profileId);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [analysisId, open]);

  const flat: FlatComment[] = comments.flatMap((c) => [
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
    <aside className="flex w-[360px] flex-col overflow-hidden rounded-xl border border-secondary bg-primary shadow-lg">
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

      <div className="flex max-h-[320px] flex-col overflow-y-auto px-4 py-3">
        {!analysisId ? (
          <p className="text-sm text-tertiary">No analysis selected.</p>
        ) : loading ? (
          <p className="text-sm text-tertiary">Loading comments…</p>
        ) : error ? (
          <p className="text-sm text-[#d92d20]">{error}</p>
        ) : flat.length === 0 ? (
          <p className="text-sm text-tertiary">No comments yet for this analysis.</p>
        ) : (
          <ol className="relative">
            {flat.map((item, i) => {
              const actionText = insightCommentActionText(item.text, item.attachments.length);
              const showTextBubble = item.attachments.length === 0 || actionText !== "Added a file";

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
                    {showTextBubble ? (
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
    </aside>
  );
}
