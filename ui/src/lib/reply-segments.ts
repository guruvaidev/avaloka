/**
 * "Commentary — not checked" marking for chat replies and upload cards (2.0, c-strict).
 *
 * The API sends `reply_segments: [{text, kind: "plain" | "commentary"}]` next
 * to the plain reply string. Commentary is text a model wrote; plain is text
 * code or a template wrote. The plain string already carries
 * COMMENTARY_SUFFIX after every commentary sentence, so a thread saved before
 * segments existed still shows the marking when rendered as-is.
 *
 * When segments are present the suffix is stripped from each commentary
 * segment and the marking is rendered instead (see ReplyContent), so it is
 * never shown twice. No check marks: 2.0 marks what is not checked and makes
 * no claim that anything else was.
 */
import type { Root, RootContent, Parent, Text } from "mdast";

/** Mirrors COMMENTARY_SUFFIX in app/api/reply_segments.py (pinned by a test). */
export const COMMENTARY_SUFFIX = " (commentary — not checked)";
/** Mirrors TABLE_SUFFIX: after a table a model wrote, the label on its own line so the table still parses. */
export const TABLE_SUFFIX = "\n\n" + COMMENTARY_SUFFIX.trim();
export const COMMENTARY_LABEL = "commentary — not checked";
export const COMMENTARY_TOOLTIP = "Avaloka wrote this; it is not checked against your data.";

export type ReplySegmentKind = "plain" | "commentary";
export type ReplySegment = { text: string; kind: ReplySegmentKind };

/** The segments when `raw` is a usable list; null otherwise (old threads, bad shapes). */
export function parseReplySegments(raw: unknown): ReplySegment[] | null {
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const out: ReplySegment[] = [];
  for (const s of raw) {
    if (!s || typeof s !== "object") return null;
    const { text, kind } = s as { text?: unknown; kind?: unknown };
    if (typeof text !== "string" || (kind !== "plain" && kind !== "commentary")) return null;
    out.push({ text, kind });
  }
  return out;
}

/** Mirrors is_table: a table is the only commentary segment that spans lines. */
export const isTableSegment = (text: string) => text.includes("\n");

/** Mirrors suffix_for: the suffix the reply string carries after commentary `text`. */
export function suffixFor(text: string): string {
  return isTableSegment(text) ? TABLE_SUFFIX : COMMENTARY_SUFFIX;
}

const ANY_SUFFIX = /(\n\n\(commentary — not checked\)| \(commentary — not checked\))/;
const TABLE_LINE = /^[ \t]*\|.*\|[ \t]*$/;

/**
 * Segments recovered from a suffixed string, for text that travelled without
 * its segments (an insight list copied or pasted from a card). Before
 * COMMENTARY_SUFFIX the commentary is the sentence or item on that line;
 * before TABLE_SUFFIX it is the table above it. Earlier lines are plain, and
 * anything after the last suffix is plain. Within a line it errs toward
 * commentary, as the contract does for untagged text. Null when the string
 * carries no suffix (older data, or nothing a model wrote).
 */
export function segmentsFromSuffixed(text: string): ReplySegment[] | null {
  if (!ANY_SUFFIX.test(text)) return null;
  const parts = text.split(ANY_SUFFIX); // [piece, suffix, piece, suffix, ..., tail]
  const out: ReplySegment[] = [];
  const plain = (t: string) => t && out.push({ kind: "plain", text: t });
  for (let i = 0; i < parts.length; i += 2) {
    const piece = parts[i] ?? "";
    const suffix = parts[i + 1];
    if (suffix === undefined) {
      plain(piece);
      continue;
    }
    const body = piece.trimEnd();
    const trail = piece.slice(body.length);
    let start: number;
    if (suffix === TABLE_SUFFIX) {
      const lines = body.split("\n");
      let k = lines.length;
      while (k > 0 && TABLE_LINE.test(lines[k - 1]!)) k--;
      start = k < lines.length ? lines.slice(0, k).join("\n").length + (k ? 1 : 0) : body.length;
    } else {
      start = body.lastIndexOf("\n") + 1;
    }
    const core = body.slice(start).trim();
    if (!core) {
      plain(piece);
      continue;
    }
    plain(body.slice(0, body.indexOf(core, start)));
    out.push({ kind: "commentary", text: core });
    plain(trail);
  }
  return out;
}

/** `text` without commentary suffixes: TABLE_SUFFIX, COMMENTARY_SUFFIX, or the bare label. */
export function stripCommentarySuffix(text: string): string {
  return text.split(TABLE_SUFFIX).join("").split(COMMENTARY_SUFFIX).join("").split(COMMENTARY_SUFFIX.trim()).join("");
}

// Private-use code points bracket each commentary span in the markdown
// handed to the renderer; remarkCommentary turns them into marked spans
// after markdown has been parsed, so lists, bold and links keep working
// inside and around commentary.
export const COMMENTARY_START = "\uE000";
export const COMMENTARY_END = "\uE001";

// Block syntax a sentence can open with: kept outside the brackets so the
// markdown still parses as a list item / heading / quote.
const LEADING_BLOCK = /^(\s*(?:(?:[-*+]|\d+[.)])\s+|#{1,6}\s+|>\s*)*)/;

const TABLE_ROW = /^[ \t]*\|.*\|[ \t]*$/;

/**
 * `prepare` (e.g. the chat's table stripping) applied segment by segment, so
 * a commentary table the chat moves to the Data table panel takes its label
 * with it. Each segment keeps its surrounding whitespace; a commentary
 * segment left empty is dropped.
 */
export function prepareSegments(segments: ReplySegment[], prepare: (text: string) => string): ReplySegment[] {
  const out: ReplySegment[] = [];
  for (const s of segments) {
    const text = stripCommentarySuffix(s.text);
    if (!text.trim()) {
      out.push({ ...s, text });
      continue;
    }
    const core = prepare(text).trim();
    if (!core && s.kind === "commentary") continue;
    const lead = text.match(/^\s*/)?.[0] ?? "";
    const trail = text.match(/\s*$/)?.[0] ?? "";
    out.push({ ...s, text: core ? `${lead}${core}${trail}` : `${lead}${trail}` });
  }
  return out;
}

// A list marker alone ("1.", "-"): block syntax, not a sentence.
const MARKER_ONLY = /^\s*(?:\d+[.)]|[-*+])\s*$/;
const isTableText = (t: string) =>
  t.split("\n").every((l) => !l.trim() || TABLE_ROW.test(l)) && t.split("\n").some((l) => TABLE_ROW.test(l));

/** One markdown string with every commentary segment bracketed, suffix removed. */
export function segmentsToMarkdown(segments: ReplySegment[]): string {
  const out: string[] = [];
  // Brackets inside a table would break it, so a table a model wrote (one
  // segment, as the contract sends it) is emitted as it is and labelled once,
  // on its own line after the table. Rows sent one by one with the separator
  // plain between (segments saved before the contract fixed that) are
  // gathered the same way, and so is a list marker sent apart from its item.
  let tableLabel = false;
  const closeTable = () => {
    if (tableLabel) out.push(`\n\n${COMMENTARY_START}${COMMENTARY_END}\n\n`);
    tableLabel = false;
  };
  for (const { text, kind } of segments) {
    const clean = stripCommentarySuffix(text).replace(/[\uE000\uE001]/g, "");
    if (isTableText(clean)) {
      if (kind === "commentary") tableLabel = true;
      out.push(clean);
      continue;
    }
    if (tableLabel && !clean.trim()) {
      out.push(clean);
      continue;
    }
    closeTable();
    if (kind !== "commentary" || MARKER_ONLY.test(clean)) {
      out.push(clean);
      continue;
    }
    const lead = clean.match(LEADING_BLOCK)?.[1] ?? "";
    const rest = clean.slice(lead.length);
    const core = rest.trimEnd();
    out.push(core ? `${lead}${COMMENTARY_START}${core}${COMMENTARY_END}${rest.slice(core.length)}` : clean);
  }
  closeTable();
  return out.join("");
}

function hnode(className: string, children: RootContent[], props: Record<string, unknown> = {}): RootContent {
  // An mdast node remark-rehype renders as <span>: data.hName/hProperties.
  return {
    type: "commentary",
    data: { hName: "span", hProperties: { className, ...props } },
    children,
  } as unknown as RootContent;
}

function marker(): RootContent[] {
  return [
    hnode("reply-commentary-label", [{ type: "text", value: COMMENTARY_LABEL } as Text], {
      title: COMMENTARY_TOOLTIP,
      "data-commentary-label": "",
    }),
    hnode("sr-only", [{ type: "text", value: `: ${COMMENTARY_TOOLTIP}` } as Text]),
  ];
}

const LEAF_INLINE = new Set(["text", "inlineCode", "image", "break"]);

/**
 * Remark plugin: wraps everything between COMMENTARY_START and COMMENTARY_END
 * in spans with class `reply-commentary`, and closes each commentary run with
 * a visible "commentary — not checked" label carrying the tooltip.
 */
export function remarkCommentary() {
  return (tree: Root) => {
    let inside = false;
    let first = false;
    const walk = (parent: Parent) => {
      const next: RootContent[] = [];
      for (const child of parent.children as RootContent[]) {
        if (child.type === "text") {
          const parts = (child as Text).value.split(/([\uE000\uE001])/);
          for (const part of parts) {
            if (part === COMMENTARY_START) {
              inside = true;
              first = true;
            } else if (part === COMMENTARY_END) {
              if (inside) next.push(...marker());
              inside = false;
            } else if (part) {
              const text = { type: "text", value: part } as Text;
              if (inside) {
                next.push(
                  hnode(first ? "reply-commentary reply-commentary-start" : "reply-commentary", [text], {
                    title: COMMENTARY_TOOLTIP,
                    "data-commentary": "",
                  }),
                );
                first = false;
              } else {
                next.push(text);
              }
            }
          }
        } else if (LEAF_INLINE.has(child.type)) {
          next.push(
            inside
              ? hnode(first ? "reply-commentary reply-commentary-start" : "reply-commentary", [child], {
                  title: COMMENTARY_TOOLTIP,
                  "data-commentary": "",
                })
              : child,
          );
          if (inside) first = false;
        } else {
          if ("children" in child) walk(child as Parent);
          next.push(child);
        }
      }
      parent.children = next as Parent["children"];
    };
    walk(tree);
    // A bracket left open (its close fell inside a stripped table) still gets its label.
    if (inside) {
      const last = tree.children[tree.children.length - 1] as Parent | undefined;
      if (last && "children" in last) (last.children as RootContent[]).push(...marker());
    }
  };
}
