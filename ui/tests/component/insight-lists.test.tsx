// The insight lists collected outside the cards (analysis.tsx: the tab's Auto
// Insights, a section's user insights, the chart carousel, pasted analyses)
// mark commentary like the cards do, and never show the raw suffix.
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CollectedInsight, collectSlidesInsights } from "@/components/chat/InsightLine";
import { normalizeChart } from "@/components/dashboard/dynamicChart";
import { COMMENTARY_SUFFIX, COMMENTARY_TOOLTIP } from "@/lib/reply-segments";
import live from "../fixtures/commentary/live-shape.json";

const S = COMMENTARY_SUFFIX;
const charts = (live.upload as any).visualization_config.charts as any[];
// A model-written card whose insight is the model's sentence plus code-written facts.
const llmChart = charts.find(
  (c) => c.insight_source === "llm" && c.insight_segments?.some((s: any) => s.kind === "plain" && s.text.trim()),
);

const List = ({ items, segments }: { items: string[]; segments?: Record<string, any> }) => (
  <ul>
    {items.map((t, i) => (
      <li key={i}>
        <CollectedInsight text={t} segments={segments?.[t]} />
      </li>
    ))}
  </ul>
);

describe("collected insight lists", () => {
  it("mark a card's model-written sentence from its segments, suffix stripped, facts plain", () => {
    // The upload sends the sentence as both reason (the subtitle) and insight:
    // it is collected once, as a summary, and must be marked there too.
    const { summaries, insights, segments } = collectSlidesInsights([normalizeChart(llmChart)!]);
    const items = [...summaries, ...insights];
    expect(items).toHaveLength(1);
    const { container } = render(<List items={items} segments={segments} />);
    const marked = container.querySelector("[data-commentary]")!;
    expect(marked.textContent).toBe(llmChart.insight_segments[0].text);
    expect(marked.getAttribute("title")).toBe(COMMENTARY_TOOLTIP);
    expect(container.querySelectorAll("[data-commentary-label]")).toHaveLength(1);
    expect(container.textContent).not.toContain(S.trim());
    const facts = llmChart.insight_segments.find((s: any) => s.kind === "plain").text.trim();
    expect(container.textContent).toContain(facts);
    expect(marked.textContent).not.toContain(facts);
  });

  it("mark a pasted insight that kept only its suffix, and leave an unsuffixed one as it is", () => {
    const pasted = `Spend rose after the launch.${S} Records by month: 12 rows.`;
    const older = "Spend rose after the launch.";
    const { container } = render(<List items={[pasted, older]} />);
    const [first, second] = container.querySelectorAll("li");
    expect(first!.querySelector("[data-commentary]")?.textContent).toBe("Spend rose after the launch.");
    expect(first!.textContent).toContain("Records by month: 12 rows.");
    expect(first!.textContent).not.toContain(S.trim());
    expect(second!.querySelector("[data-commentary]")).toBeNull();
    expect(second!.textContent).toBe(older);
  });
});
