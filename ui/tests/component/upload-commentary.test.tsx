// 2.0 (c-strict) on upload: the card fields a model wrote (lead, headline,
// title, watch-outs, readiness) carry `<field>_segments` and are marked
// "commentary — not checked"; the code-written facts stay plain.
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DatasetOverview } from "@/components/dashboard/AutoInsightsModal";
import { CommentaryText } from "@/components/chat/CommentaryText";
import { MarkdownContent } from "@/components/chat/MarkdownContent";
import { normalizeChart } from "@/components/dashboard/dynamicChart";
import { buildUploadSummaryMessage, buildUploadSummarySegments } from "@/lib/upload-summary";
import { promptsToMessages } from "@/lib/assistant-prompts";
import { COMMENTARY_LABEL, COMMENTARY_SUFFIX, COMMENTARY_TOOLTIP } from "@/lib/reply-segments";
import live from "../fixtures/commentary/live-shape.json";

const S = COMMENTARY_SUFFIX;
const CHECKS = /[✓✔✅☑\u{1F5F8}]/u;
const labels = (root: ParentNode) => root.querySelectorAll("[data-commentary-label]");

const upload = live.upload as any;
const overview = upload.visualization_config.profile_overview;
const llmChart = upload.visualization_config.charts.find((c: any) => c.insight_source === "llm");

describe("Dataset overview (upload card)", () => {
  it("marks the model's watch-outs and readiness reasoning and strips their suffix", () => {
    const { container } = render(<DatasetOverview config={upload.visualization_config} />);
    const text = container.textContent ?? "";
    expect(text).not.toContain(S.trim());
    const marked = [...container.querySelectorAll("[data-commentary]")];
    expect(marked.length).toBeGreaterThanOrEqual(2);
    for (const el of marked) {
      expect(el.className).toMatch(/italic/);
      expect(el.className).toMatch(/border-l-2/);
      expect(el.getAttribute("title")).toBe(COMMENTARY_TOOLTIP);
    }
    expect(labels(container).length).toBe(marked.length);
    expect(text).toContain(overview.red_flags_segments[0][0].text);
    expect(container.innerHTML).not.toMatch(CHECKS);
    // An older overview (no segments) is shown as sent, suffix and all; its
    // readiness score is still the profiling model's, so it is marked.
    const old = { profile_overview: { readiness_score: 70, red_flags: [`Billing is skewed.${S}`] } };
    const older = render(<DatasetOverview config={old} />).container;
    expect([...older.querySelectorAll("[data-commentary]")].map((e) => e.textContent)).toEqual(["70/100"]);
    expect(older.textContent).toContain(`Billing is skewed.${S}`);
  });

  it("prefills the chat with a suggestion's words, not its label", () => {
    const onPrefill = vi.fn();
    render(<DatasetOverview config={upload.visualization_config} onPrefillChat={onPrefill} />);
    const first = overview.analysis_suggestions_segments[0][0].text as string;
    fireEvent.click(screen.getByText(first));
    expect(onPrefill).toHaveBeenCalledWith(first);
  });
});

describe("chart cards", () => {
  it("carry title, headline and insight segments through normalizeChart", () => {
    const slide = normalizeChart(llmChart)!;
    expect(slide.insightSegments?.some((s) => s.kind === "commentary")).toBe(true);
    const { container } = render(
      <p>
        <CommentaryText text={slide.insights[0]!} segments={slide.insightSegments} tone="light" />
      </p>,
    );
    expect(container.textContent).not.toContain(S.trim());
    expect(labels(container).length).toBeGreaterThanOrEqual(1);
  });

  it("keeps the label visible when a title is truncated to one line", () => {
    const { container } = render(
      <CommentaryText text={`Spend by segment${S}`} segments={[{ kind: "commentary", text: "Spend by segment" }]} truncate />,
    );
    const label = labels(container)[0]!;
    expect(label.textContent).toBe(COMMENTARY_LABEL);
    expect(label.closest(".truncate")).toBeNull();
  });
});

describe("the upload chat message", () => {
  it("renders upload_message_segments, marked, without the suffix", () => {
    const content = buildUploadSummaryMessage(upload)!;
    const segments = buildUploadSummarySegments(upload);
    expect(content).toContain(S);
    expect(segments).not.toBeNull();
    const { container } = render(<MarkdownContent content={content} segments={segments} />);
    expect(container.textContent).not.toContain(S.trim());
    expect(labels(container).length).toBe(segments!.filter((s) => s.kind === "commentary").length);
    // An older backend sends no upload_message: the browser builds it, unsegmented.
    const older = { ...upload, visualization_config: { ...upload.visualization_config, upload_message: undefined } };
    expect(buildUploadSummarySegments(older)).toBeNull();
  });
});

describe("assistant_prompts", () => {
  it("mark the message's commentary and leave the question plain", () => {
    const [msg] = promptsToMessages(
      {
        assistant_prompts: [
          {
            step: "auto_insights",
            message: `Rows: 1,200.\n\nChurn looks seasonal.${S}`,
            message_segments: [
              { kind: "plain", text: "Rows: 1,200.\n\n" },
              { kind: "commentary", text: "Churn looks seasonal." },
            ],
            question: "Which column is the outcome?",
          },
        ],
      },
      "p",
      "now",
    );
    const { container } = render(<MarkdownContent content={msg!.content} segments={msg!.segments} />);
    expect(container.querySelector("[data-commentary]")?.textContent).toBe("Churn looks seasonal.");
    expect(container.textContent).toContain("Which column is the outcome?");
    expect(labels(container)).toHaveLength(1);
  });
});
