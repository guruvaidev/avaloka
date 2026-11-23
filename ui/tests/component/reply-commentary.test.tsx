// 2.0 (c-strict): model-written sentences in a chat reply are marked
// "commentary — not checked"; code- and template-written text stays plain.
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MarkdownContent } from "@/components/chat/MarkdownContent";
import {
  COMMENTARY_LABEL,
  COMMENTARY_SUFFIX,
  COMMENTARY_TOOLTIP,
  TABLE_SUFFIX,
  segmentsFromSuffixed,
  suffixFor,
  type ReplySegment,
} from "@/lib/reply-segments";
import live from "../fixtures/commentary/live-shape.json";

const S = COMMENTARY_SUFFIX;

// Shaped like rweb15's rw7 reply: a code-written lead, model-written bullets.
const segments: ReplySegment[] = [
  { kind: "plain", text: "Enterprise customers spend the most on average, with $545.08 per order." },
  { kind: "commentary", text: `  \n\n- Enterprise leads by a factor of roughly **3×** SMB.${S}` },
  { kind: "commentary", text: `  \n- SMB orders are about three times larger than consumer orders.${S}` },
  { kind: "plain", text: "\n\nI've held back 1 statement from this answer." },
];
const content = segments.map((s) => s.text).join("");

const CHECKS = /[✓✔✅☑\u{1F5F8}]/u;

describe("MarkdownContent with reply_segments", () => {
  it("marks each commentary segment with the class, the tooltip and a visible label", () => {
    const { container } = render(<MarkdownContent content={content} segments={segments} />);
    const marked = [...container.querySelectorAll(".reply-commentary")];
    expect(marked.length).toBeGreaterThanOrEqual(2);
    for (const el of marked) expect(el.getAttribute("title")).toBe(COMMENTARY_TOOLTIP);
    expect(marked.map((e) => e.textContent).join("")).toContain("SMB orders are about three times larger");
    // A sentence with bold inside is still marked end to end.
    const bullet = container.querySelectorAll("li")[0]!;
    expect(bullet.querySelector("strong .reply-commentary")?.textContent).toBe("3×");
    // One left rule per commentary run, where it starts.
    expect(container.querySelectorAll(".reply-commentary-start")).toHaveLength(2);
    const labels = container.querySelectorAll("[data-commentary-label]");
    expect(labels).toHaveLength(2);
    expect(labels[0]!.textContent).toBe(COMMENTARY_LABEL);
    expect(labels[0]!.getAttribute("title")).toBe(COMMENTARY_TOOLTIP);
  });

  it("leaves code-written text plain", () => {
    const { container } = render(<MarkdownContent content={content} segments={segments} />);
    expect(container.querySelector(".reply-commentary")).not.toBeNull();
    const lead = container.querySelector("p")!;
    expect(lead.textContent).toContain("$545.08 per order");
    expect(lead.querySelector(".reply-commentary")).toBeNull();
    expect(container.textContent).toContain("I've held back 1 statement");
  });

  it("keeps the markdown structure: commentary bullets are still list items", () => {
    const { container } = render(<MarkdownContent content={content} segments={segments} />);
    const items = container.querySelectorAll("ul > li");
    expect(items).toHaveLength(2);
    for (const li of items) expect(li.querySelector(".reply-commentary-start")).not.toBeNull();
    expect(container.textContent).not.toMatch(/[\uE000\uE001]/);
  });

  it("strips the suffix from the text so the marking is not shown twice", () => {
    const { container } = render(<MarkdownContent content={content} segments={segments} />);
    const text = container.textContent ?? "";
    expect(text).not.toContain(S.trim());
    // The label is the only place the words appear: once per commentary run.
    expect(text.split(COMMENTARY_LABEL).length - 1).toBe(2);
  });

  it("gives screen readers the tooltip text", () => {
    const { container } = render(<MarkdownContent content={content} segments={segments} />);
    const hidden = [...container.querySelectorAll(".sr-only")].map((e) => e.textContent);
    expect(hidden).toHaveLength(2);
    for (const t of hidden) expect(t).toContain(COMMENTARY_TOOLTIP);
  });

  it("shows no check mark anywhere", () => {
    const { container } = render(<MarkdownContent content={content} segments={segments} />);
    expect(container.querySelectorAll("[data-commentary-label]").length).toBeGreaterThan(0);
    expect(container.innerHTML).not.toMatch(CHECKS);
    expect(container.querySelector("[class*='check' i], [data-verified]")).toBeNull();
  });

  it("applies prepare (the chat strips tables) to the marked markdown", () => {
    const withTable: ReplySegment[] = [
      { kind: "commentary", text: `Consumers trail.${S}` },
      { kind: "plain", text: "\n\n| a | b |\n|---|---|\n| 1 | 2 |\n" },
    ];
    const { container } = render(
      <MarkdownContent content="" segments={withTable} prepare={(md) => md.replace(/\n\|[\s\S]*$/, "")} />,
    );
    expect(container.querySelector("table")).toBeNull();
    expect(container.querySelector(".reply-commentary")?.textContent).toBe("Consumers trail.");
  });
});

describe("threads saved before 2.0 (no reply_segments)", () => {
  it("marks from segments when present, and renders the string as-is when absent or malformed", () => {
    const marked = render(<MarkdownContent content={content} segments={segments} />).container;
    expect(marked.querySelectorAll("[data-commentary-label]")).toHaveLength(2);
    expect(marked.textContent).not.toContain(S.trim());

    const old = render(<MarkdownContent content={content} />).container;
    expect(old.querySelector(".reply-commentary")).toBeNull();
    expect((old.textContent ?? "").split(S.trim()).length - 1).toBe(2);
    expect(old.querySelectorAll("ul > li")).toHaveLength(2);

    const bad = [{ kind: "verified", text: "x" }] as unknown as ReplySegment[];
    const fallback = render(<MarkdownContent content="Plain reply." segments={bad} />).container;
    expect(fallback.textContent).toContain("Plain reply.");
    expect(fallback.querySelector("[data-reply-segments]")).toBeNull();
  });
});

// Live shape: rweb16 replies as the backend assembles them (generate_live_shape.py).
describe("live shape: stored rweb16 replies through the backend's assembly", () => {
  for (const [key, raw] of Object.entries(live.chat)) {
    const msg = raw as { content: string; reply_segments: ReplySegment[] };
    const commentary = msg.reply_segments.filter((s) => s.kind === "commentary");

    it(`${key}: every commentary segment is marked once, the suffix is gone, and old threads keep it`, () => {
      const { container } = render(<MarkdownContent content={msg.content} segments={msg.reply_segments} />);
      expect(container.querySelectorAll("[data-commentary-label]")).toHaveLength(commentary.length);
      expect(container.textContent).not.toContain(S.trim());
      expect(container.textContent).not.toMatch(/[]/);
      expect(container.innerHTML).not.toMatch(CHECKS);
      if (/\n- /.test(msg.content)) expect(container.querySelectorAll("li").length).toBeGreaterThanOrEqual(2);
      if (msg.content.includes("held back")) {
        const note = [...container.querySelectorAll("p")].find((p) => p.textContent?.includes("held back"))!;
        expect(note.querySelector(".reply-commentary")).toBeNull();
      }
      // The same reply saved before 2.0 (no segments) shows the label once per commentary segment.
      const old = render(<MarkdownContent content={msg.content} />).container;
      expect((old.textContent ?? "").split(S.trim()).length - 1).toBe(commentary.length);
    });
  }
});

// The contract (C1 6fffcc8c): a model's table is ONE commentary segment with
// TABLE_SUFFIX (the label on its own line); a list item is one segment, marker
// kept. rw7b's related/ask_tables reply has both.
const tables = live.chat.rw7b_related_tables as { content: string; reply_segments: ReplySegment[] };
const recovered = segmentsFromSuffixed(tables.content)!;

const tableAndList = (container: HTMLElement) => {
  const table = container.querySelector("table")!;
  expect(table).not.toBeNull();
  expect(table.querySelectorAll("tr").length).toBe(3); // header + customers + orders
  expect(table.textContent).not.toContain(S.trim());
  // One label for the whole table, right after it.
  const after = table.nextElementSibling!;
  expect(after.querySelectorAll("[data-commentary-label]")).toHaveLength(1);
  expect(after.textContent?.replace(/:.*/s, "")).toBe(COMMENTARY_LABEL);
  // The numbered list keeps its markers: an <ol>, each item marked, no stray "1.".
  const items = container.querySelectorAll("ol > li");
  expect(items.length).toBeGreaterThanOrEqual(2);
  for (const li of items) {
    expect(li.querySelector(".reply-commentary-start")).not.toBeNull();
    expect(li.textContent).not.toMatch(/^\s*\d+[.)]\s/);
  }
  expect(container.textContent).not.toMatch(/\n\s*\d+[.)]\s*\n/);
  expect(container.textContent).not.toContain(S.trim());
};

describe("tables and numbered lists (TABLE_SUFFIX)", () => {
  it("the contract's suffixes: TABLE_SUFFIX after a table, COMMENTARY_SUFFIX otherwise", () => {
    expect(TABLE_SUFFIX).toBe("\n\n(commentary — not checked)");
    const table = tables.reply_segments.find((s) => s.kind === "commentary" && s.text.includes("\n"))!;
    expect(suffixFor(table.text)).toBe(TABLE_SUFFIX);
    expect(suffixFor("Spend rose.")).toBe(COMMENTARY_SUFFIX);
    // The reply string is exactly the segments with suffix_for after each commentary one.
    const rendered = tables.reply_segments.map((s) => s.text + (s.kind === "commentary" ? suffixFor(s.text) : "")).join("");
    expect(rendered).toBe(tables.content);
  });

  it("a table renders as a table with one label, from segments and from a pasted reply", () => {
    tableAndList(render(<MarkdownContent content={tables.content} segments={tables.reply_segments} />).container);
    tableAndList(render(<MarkdownContent content={tables.content} segments={recovered} />).container);
    // Guard: segments saved before the contract fix (rows one by one, a marker apart) still render.
    const S1 = [
      { kind: "commentary", text: "| a | b |" },
      { kind: "plain", text: "\n|---|---|\n" },
      { kind: "commentary", text: "| 1 | 2 |" },
      { kind: "plain", text: "\n\n" },
      { kind: "commentary", text: "1." },
      { kind: "plain", text: " " },
      { kind: "commentary", text: "**Join** on id." },
    ] as ReplySegment[];
    const { container: older } = render(<MarkdownContent content="" segments={S1} />);
    expect(older.querySelectorAll("table tr")).toHaveLength(2);
    expect(older.querySelectorAll("ol > li")).toHaveLength(1);
    expect(older.querySelectorAll("[data-commentary-label]")).toHaveLength(2);
  });

  it("a numbered list keeps its markers, and the recovered segments match the backend's", () => {
    const kinds = (segs: ReplySegment[]) =>
      segs.filter((s) => s.kind === "commentary").map((s) => s.text.trim());
    expect(kinds(recovered)).toEqual(kinds(tables.reply_segments));
    const c = live.chat.rw7c_related_tables as { content: string; reply_segments: ReplySegment[] };
    expect(kinds(segmentsFromSuffixed(c.content)!)).toEqual(kinds(c.reply_segments));
    const { container } = render(<MarkdownContent content={c.content} segments={segmentsFromSuffixed(c.content)} />);
    const items = container.querySelectorAll("ol > li");
    expect(items.length).toBeGreaterThanOrEqual(2);
    for (const li of items) expect(li.querySelector(".reply-commentary-start")).not.toBeNull();
  });
});
