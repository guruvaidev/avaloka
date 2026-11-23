import type { ReactNode } from "react";
import { CommentaryText } from "@/components/chat/CommentaryText";
import type { Slide } from "@/components/dashboard/dynamicChart";
import type { ReplySegment } from "@/lib/reply-segments";

/**
 * The insight lists collected from the charts of a tab, a section or a pasted
 * analysis: summaries and insights, deduplicated, each with its commentary
 * segments (2.0) when its card sent them.
 */
export function collectSlidesInsights(slides: Slide[]) {
  const summaries: string[] = [];
  const insights: string[] = [];
  const segments: Record<string, ReplySegment[]> = {};
  const seen = new Set<string>();
  const add = (list: string[], text?: string | null, segs?: ReplySegment[] | null) => {
    const t = text?.trim();
    if (!t || seen.has(t)) return;
    seen.add(t);
    list.push(t);
    if (segs) segments[t] = segs;
  };
  for (const slide of slides) {
    // A card's subtitle is often its insight again (the upload sends the same
    // sentence as reason and insight): it then carries the insight's segments.
    const only = slide.insights.length === 1 ? slide.insightSegments : undefined;
    add(summaries, slide.subtitle, slide.subtitle?.trim() === slide.insights[0]?.trim() ? only : undefined);
    slide.insights.forEach((item) => add(insights, item, only));
  }
  return { summaries, insights, segments };
}

/**
 * One collected insight, marked like every other commentary: from its
 * segments, or from its suffix when it travelled without them (copied or
 * pasted analyses keep only the string).
 */
export function CollectedInsight({
  text,
  segments,
  renderText,
}: {
  text: string;
  segments?: ReplySegment[];
  renderText?: (text: string) => ReactNode;
}) {
  return <CommentaryText text={text} segments={segments} fromSuffix renderText={renderText} />;
}
