import type { ReactNode } from "react";
import {
  COMMENTARY_LABEL,
  COMMENTARY_TOOLTIP,
  parseReplySegments,
  segmentsFromSuffixed,
  stripCommentarySuffix,
  type ReplySegment,
} from "@/lib/reply-segments";

// Same marking as a chat reply (MarkdownContent): muted italic text with a
// left rule, then a "commentary — not checked" label.
// "theme" follows light/dark mode (text-tertiary, AA in both; see
// MarkdownContent). "light" is for surfaces that stay white in dark mode
// (the Auto Insights cards): #475467 on white is 7.4:1.
type Tone = "theme" | "light";
const MUTED: Record<Tone, string> = { theme: "text-tertiary", light: "text-[#475467]" };
const LABEL_BG: Record<Tone, string> = { theme: "bg-secondary", light: "bg-[#f2f4f7]" };

export const commentaryTextClass = (tone: Tone = "theme") =>
  `reply-commentary reply-commentary-start ml-0.5 border-l-2 border-current pl-1.5 italic ${MUTED[tone]}`;
const labelClass = (tone: Tone) =>
  `reply-commentary-label ml-1 cursor-help whitespace-nowrap rounded px-1 text-[11px] font-normal not-italic ${LABEL_BG[tone]} ${MUTED[tone]}`;

export function CommentaryLabel({ tone = "theme" }: { tone?: Tone }) {
  return (
    <>
      <span className={labelClass(tone)} title={COMMENTARY_TOOLTIP} data-commentary-label="">
        {COMMENTARY_LABEL}
      </span>
      <span className="sr-only">: {COMMENTARY_TOOLTIP}</span>
    </>
  );
}

/**
 * One upload-card field (a lead, a finding, a title). With `segments` the
 * model's sentences are marked and the suffix in `text` is not shown; without
 * them `text` is shown as-is, so an older payload keeps its suffix.
 * `truncate` keeps the label visible when the text is cut to one line.
 * `fromSuffix` recovers the segments from the suffix when none travelled
 * with the text (insight lists collected from cards); `renderText` renders
 * each piece of text (e.g. as inline markdown).
 */
export function CommentaryText({
  text,
  segments,
  truncate = false,
  tone = "theme",
  fromSuffix = false,
  renderText = (t: string) => t,
}: {
  text: string;
  segments?: unknown;
  truncate?: boolean;
  tone?: Tone;
  fromSuffix?: boolean;
  renderText?: (text: string) => ReactNode;
}): ReactNode {
  const valid = parseReplySegments(segments) ?? (fromSuffix ? segmentsFromSuffixed(text) : null);
  if (!valid) return truncate ? <span className="truncate">{text}</span> : renderText(text);
  const commentary = valid.some((s) => s.kind === "commentary");
  if (truncate) {
    const shown = valid.map((s) => stripCommentarySuffix(s.text)).join("").trim();
    return (
      <span className="flex min-w-0 items-baseline" data-reply-segments="">
        <span className={`truncate ${commentary ? `italic ${MUTED[tone]}` : ""}`} title={commentary ? COMMENTARY_TOOLTIP : undefined}>
          {shown}
        </span>
        {commentary && <span className="shrink-0"><CommentaryLabel tone={tone} /></span>}
      </span>
    );
  }
  return (
    <span data-reply-segments="">
      {valid.map((s: ReplySegment, i) => {
        const clean = stripCommentarySuffix(s.text);
        if (s.kind !== "commentary") return <span key={i}>{renderText(clean)}</span>;
        const core = clean.trim();
        if (!core) return <span key={i}>{clean}</span>;
        const lead = clean.slice(0, clean.indexOf(core));
        return (
          <span key={i}>
            {lead}
            <span className={commentaryTextClass(tone)} title={COMMENTARY_TOOLTIP} data-commentary="">
              {renderText(core)}
            </span>
            <CommentaryLabel tone={tone} />
            {clean.slice(lead.length + core.length)}
          </span>
        );
      })}
    </span>
  );
}

/** The field's text without the suffix, for reuse as input (e.g. prefilling the chat). */
export function commentaryPlainText(text: string, segments?: unknown): string {
  const valid = parseReplySegments(segments);
  return valid ? valid.map((s) => stripCommentarySuffix(s.text)).join("").trim() : stripCommentarySuffix(text);
}
