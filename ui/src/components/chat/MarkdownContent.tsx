import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  parseReplySegments,
  prepareSegments,
  remarkCommentary,
  segmentsToMarkdown,
  type ReplySegment,
} from "@/lib/reply-segments";

// Commentary (model-written text) is muted and italic with a left rule where
// it starts, and closes with a small "commentary — not checked" label. The
// muted colour is text-tertiary: #51525c on white (7.6:1) and neutral-400 on
// neutral-950 in dark mode, both above WCAG AA's 4.5:1 for body text.
const COMMENTARY_CLASSES =
  "[&_.reply-commentary]:italic [&_.reply-commentary]:text-tertiary " +
  "[&_.reply-commentary-start]:ml-0.5 [&_.reply-commentary-start]:border-l-2 [&_.reply-commentary-start]:border-current [&_.reply-commentary-start]:pl-1.5 " +
  "[&_.reply-commentary-label]:ml-1 [&_.reply-commentary-label]:rounded [&_.reply-commentary-label]:bg-secondary [&_.reply-commentary-label]:px-1 [&_.reply-commentary-label]:text-[11px] [&_.reply-commentary-label]:not-italic [&_.reply-commentary-label]:text-tertiary [&_.reply-commentary-label]:cursor-help [&_.reply-commentary-label]:whitespace-nowrap";

const MARKDOWN_CLASSES =
  "prose-sm max-w-none break-words text-sm leading-relaxed [&_a]:text-[#1565ef] [&_a]:underline [&_code]:rounded [&_code]:bg-secondary/60 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[12px] [&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-[#0f172a] [&_pre]:p-3 [&_pre]:text-[12px] [&_pre]:text-white [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-white [&_h1]:mt-2 [&_h1]:mb-1 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:mt-2 [&_h2]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h3]:mt-2 [&_h3]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold [&_p]:my-1 [&_ul]:my-1 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5 [&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-secondary [&_blockquote]:pl-3 [&_blockquote]:text-secondary [&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-secondary [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_td]:border [&_td]:border-secondary [&_td]:px-2 [&_td]:py-1 [&_hr]:my-2 [&_hr]:border-secondary [&_strong]:font-semibold";

/**
 * A chat reply. With `segments` (2.0 replies) the commentary is marked from
 * them and the suffix in the string is not shown; without (threads saved
 * before 2.0) `content` is rendered as-is, suffix and all.
 * `prepare` post-processes each segment's text (the chat strips tables,
 * which belong in the Data table panel). `renderLink` replaces the anchor
 * (the analysis page opens Ray job ids in the dashboard); without it the
 * markdown renders as before.
 */
export function MarkdownContent({
  content,
  segments,
  prepare,
  renderLink,
}: {
  content: string;
  segments?: ReplySegment[] | null;
  prepare?: (markdown: string) => string;
  renderLink?: (link: { href?: string; children?: ReactNode }) => ReactNode;
}) {
  const components = renderLink ? { a: ({ href, children }: { href?: string; children?: ReactNode }) => renderLink({ href, children }) } : undefined;
  const valid = parseReplySegments(segments);
  const marked = valid ? segmentsToMarkdown(prepare ? prepareSegments(valid, prepare) : valid) : null;
  if (marked != null && marked.trim()) {
    return (
      <div className={`${MARKDOWN_CLASSES} ${COMMENTARY_CLASSES}`} data-reply-segments="">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkCommentary]} components={components}>{marked}</ReactMarkdown>
      </div>
    );
  }
  return (
    <div className={MARKDOWN_CLASSES}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{content}</ReactMarkdown>
    </div>
  );
}
