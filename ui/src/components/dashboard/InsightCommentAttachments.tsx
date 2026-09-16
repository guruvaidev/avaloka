import { useEffect, useState } from "react";
import {
  formatAttachmentSize,
  getInsightCommentAttachmentUrl,
  type InsightCommentAttachment,
} from "@/lib/analysis-insight-comments";
import { cx } from "@/lib/utils/cx";

function fileExtBadge(name: string) {
  const ext = (name.split(".").pop() ?? "").toLowerCase();
  if (["xls", "xlsx"].includes(ext)) return { label: "XLS", color: "bg-[#079455]" };
  if (ext === "csv") return { label: "CSV", color: "bg-[#079455]" };
  if (ext === "json") return { label: "JSON", color: "bg-[#1565ef]" };
  if (ext === "pdf") return { label: "PDF", color: "bg-[#d92d20]" };
  if (["png", "jpg", "jpeg", "gif", "webp"].includes(ext)) return { label: "IMG", color: "bg-[#1565ef]" };
  if (["doc", "docx"].includes(ext)) return { label: "DOC", color: "bg-[#1565ef]" };
  return { label: (ext || "file").slice(0, 4).toUpperCase(), color: "bg-[#3f3f46]" };
}

function FileTypeIcon({ filename, className }: { filename: string; className?: string }) {
  const { label, color } = fileExtBadge(filename);
  return (
    <div className={cx("relative h-10 w-8 shrink-0", className)}>
      <svg viewBox="0 0 32 40" fill="none" className="h-10 w-8" aria-hidden>
        <path
          d="M4 2h16l8 8v28a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Z"
          fill="#F9FAFB"
          stroke="#E4E7EC"
          strokeWidth="1.5"
        />
        <path d="M20 2v8h8" fill="#F2F4F7" stroke="#E4E7EC" strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
      <span
        className={cx(
          "absolute bottom-0.5 left-0 rounded-[2px] px-1 py-px text-[8px] font-bold leading-none text-white",
          color,
        )}
      >
        {label}
      </span>
    </div>
  );
}

function AttachmentRow({
  attachment,
  href,
}: {
  attachment: InsightCommentAttachment;
  href?: string | null;
}) {
  const inner = (
    <>
      <FileTypeIcon filename={attachment.name} />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-primary">{attachment.name}</p>
        <p className="text-xs text-tertiary">{formatAttachmentSize(attachment.size)}</p>
      </div>
    </>
  );

  if (href) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-3 rounded-lg py-1 transition hover:opacity-80"
      >
        {inner}
      </a>
    );
  }

  return <div className="flex items-center gap-3 py-1 opacity-70">{inner}</div>;
}

type Props = {
  attachments: InsightCommentAttachment[];
  className?: string;
};

/** Figma-style file rows: document icon + filename + size. */
export function InsightCommentAttachmentList({ attachments, className }: Props) {
  const [urls, setUrls] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;
    attachments.forEach((attachment) => {
      void getInsightCommentAttachmentUrl(attachment.path).then((url) => {
        if (!cancelled && url) {
          setUrls((prev) => ({ ...prev, [attachment.path]: url }));
        }
      });
    });
    return () => {
      cancelled = true;
    };
  }, [attachments]);

  if (!attachments.length) return null;

  return (
    <div className={cx("mt-2 flex flex-col gap-1", className)}>
      {attachments.map((attachment) => (
        <AttachmentRow
          key={attachment.path}
          attachment={attachment}
          href={urls[attachment.path]}
        />
      ))}
    </div>
  );
}

export function insightCommentActionText(text: string, attachmentCount: number): string {
  if (attachmentCount <= 0) return text;
  const normalized = text.trim();
  if (!normalized || normalized === "Shared an attachment.") {
    return "Added a file";
  }
  return normalized;
}
