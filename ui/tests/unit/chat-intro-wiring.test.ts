// @vitest-environment node
// The analysis page is too large to mount in a unit test, so this pins the
// wiring the goal-first and intro tests rely on: the composer sends through
// the goal gate, and the backend turn (runTurn) is reached only through it.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const page = readFileSync(
  fileURLToPath(new URL("../../src/routes/analysis.tsx", import.meta.url)),
  "utf8",
);

describe("analysis page chat wiring", () => {
  it("the composer sends through the goal gate", () => {
    expect(page).toContain("onSend={send}");
    expect(page).toMatch(/const send = async[\s\S]{0,400}return goalGate\.send\(content\);/);
  });

  it("the backend turn is only handed to the gate, never called directly", () => {
    const uses = page.match(/\brunTurn\b/g) ?? [];
    // its definition, plus the one hand-off to useGoalFirstGate
    expect(uses).toHaveLength(2);
    expect(page).toMatch(/useGoalFirstGate\(\{\s*runTurn,/);
  });

  it("renders the intro, the page commentary and the Go control in the chat", () => {
    expect(page).toContain("{showChatIntro && <ChatIntroCard onDismiss={chatIntro.dismiss} />}");
    expect(page).toContain("<PageCommentaryCard lines={pageCommentaryLines}");
    expect(page).toMatch(/goalGate\.pending && \(\s*<GoalPlanActions/);
    expect(page).toContain("const showChatIntro = !chatIntro.dismissed && isNewChat;");
  });
});
