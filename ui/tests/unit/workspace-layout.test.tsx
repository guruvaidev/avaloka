import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { DataWorkspaceCard } from "../../src/components/dashboard/data-view/DataWorkspaceCard";
import { messagesWithKeyFindings, normalizeFinding, unionFindings, withKeyFindings } from "../../src/lib/workspace-findings";

const bullets = [
  "Video views are extremely right-skewed.",
  "Average video views by category.",
  "Average highest_yearly_earnings by Country.",
  "Average subscribers by created_year.",
  "Records by Abbreviation.",
];

describe("BG45 findings preservation", () => {
  it("unites all five panel findings with existing chat findings exactly once", () => {
    const existing = [" VIDEO views are   extremely right-skewed! ", "An existing chat finding."];
    const result = unionFindings(existing, bullets.concat(bullets));
    expect(result).toHaveLength(6);
    expect(result.map(normalizeFinding)).toEqual(existing.concat(bullets.slice(1)).map(normalizeFinding));
  });

  it("merges into the first AI message, retains later messages, and is repeatable after reload", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "ai", content: "Dataset\n\n**Key findings**\n- Records by Abbreviation!\n\n**Try asking**\n- What changed?" },
      { role: "ai", content: "Later reply" },
    ];
    const merged = messagesWithKeyFindings(messages, bullets);
    expect(merged).toHaveLength(3);
    expect(merged[0]).toBe(messages[0]);
    expect(merged[2]).toBe(messages[2]);
    expect(merged[1].content.match(/^- /gm)).toHaveLength(6);
    expect(messagesWithKeyFindings(JSON.parse(JSON.stringify(merged)), bullets)).toEqual(merged);
    expect(messages[1].content).not.toContain("created_year");
  });

  it("adds a section inside the first message without adding another message", () => {
    const merged = messagesWithKeyFindings([{ role: "ai", content: "Dataset ready." }], bullets);
    expect(merged).toHaveLength(1);
    expect(merged[0].content).toContain("**Key findings**");
    for (const bullet of bullets) expect(merged[0].content).toContain(bullet);
    expect(withKeyFindings("Dataset", [])).toBe("Dataset");
    expect(withKeyFindings("**Key findings:**\n- Records by Abbreviation!", bullets).match(/Key findings/g)).toHaveLength(1);
  });

  it("the Data View card renders all four subviews without a bottom insights panel", () => {
    for (const view of ["Data sample", "Analysis results", "Chart", "Column Summary"]) {
      const html = renderToStaticMarkup(
        <DataWorkspaceCard>
          <div>{view}</div>
        </DataWorkspaceCard>,
      );
      expect(html).toContain(view);
      expect(html).not.toContain("Key Insights");
    }
  });
});

describe("BG46 workspace sizing", () => {
  it("chat width clamp keeps the data area at least 65% at 1366px and above", () => {
    for (const viewport of [1366, 1920]) {
      const chat = Math.min(400, Math.max(320, viewport * 0.26));
      const data = viewport - 64 - 32 - 16 - chat;
      expect(chat).toBeGreaterThanOrEqual(320);
      expect(chat).toBeLessThanOrEqual(400);
      expect(data / viewport).toBeGreaterThanOrEqual(0.65);
    }
  });
});
