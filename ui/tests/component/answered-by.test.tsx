// @vitest-environment happy-dom
// Owner ruling 2026-11-02: a model swap is never silent. The fixtures are the
// backend's own records (app/core/model_tiers.stamp); tests/test_default_openrouter_gptoss.py
// asserts the backend produces exactly these.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

// Loaded through import.meta.glob so that a tree without these modules (one that
// swaps models silently) fails on an assertion, not on import.
type NoteMod = typeof import("../../src/components/analysis/AnsweredByNote");
type LibMod = typeof import("../../src/lib/answered-by");
type PanelMod = typeof import("../../src/components/settings/LlmModelsPanel");
const noteMods = import.meta.glob<NoteMod>("../../src/components/analysis/AnsweredByNote.tsx");
const libMods = import.meta.glob<LibMod>("../../src/lib/answered-by.ts");
const panelMods = import.meta.glob<PanelMod>("../../src/components/settings/LlmModelsPanel.tsx");
let AnsweredByNote: NoteMod["AnsweredByNote"];
let answeredBy: LibMod["answeredBy"];
let ANSWERED_BY_TITLE: LibMod["ANSWERED_BY_TITLE"];
let LlmModelsPanel: PanelMod["LlmModelsPanel"];

beforeAll(async () => {
  for (const [mods, name] of [
    [noteMods, "ui/src/components/analysis/AnsweredByNote.tsx"],
    [libMods, "ui/src/lib/answered-by.ts"],
    [panelMods, "ui/src/components/settings/LlmModelsPanel.tsx"],
  ] as const) {
    expect(Object.values(mods), `${name} is missing: a model swap would be silent`).toHaveLength(1);
  }
  ({ AnsweredByNote } = await Object.values(noteMods)[0]());
  ({ answeredBy, ANSWERED_BY_TITLE } = await Object.values(libMods)[0]());
  ({ LlmModelsPanel } = await Object.values(panelMods)[0]());
});

const source = (rel: string) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf8");
const FIXTURE = JSON.parse(source("../fixtures/answered-by.json"));
const MODELS = JSON.parse(source("../fixtures/llm-models.json"));

function renderNote(response: unknown) {
  const record = answeredBy(response);
  if (!record) return null;
  render(<AnsweredByNote answeredBy={record} />);
  return screen.getByTestId("answered-by");
}

describe("which model answered", () => {
  it("a model refused for having no zero-retention endpoint says so, and names who answered", () => {
    const note = renderNote(FIXTURE.zdr_swap)!;
    expect(note).not.toBeNull();
    expect(note.textContent).toContain(ANSWERED_BY_TITLE);
    expect(screen.getByTestId("answered-by-sentence").textContent).toBe(
      "openai/gpt-4o did not answer (no allowed zero-retention provider serves openai/gpt-4o); " +
        "openai/gpt-6-luna via openrouter answered instead.",
    );
  });

  it("the Groq last tier names its provider even when the model id is the same", () => {
    // gpt-oss-120b on OpenRouter -> gpt-oss-120b on Groq: same id, different provider.
    const record = answeredBy(FIXTURE.groq_last);
    expect(record).not.toBeNull();
    render(<AnsweredByNote answeredBy={record!} />);
    expect(screen.getByTestId("answered-by-sentence").textContent).toContain(
      "openai/gpt-oss-120b via groq answered instead.",
    );
    expect(screen.getByTestId("answered-by-sentence").textContent).toContain(
      "out of credit (HTTP 402)",
    );
  });

  it("no record (the configured model answered) shows nothing", () => {
    expect(answeredBy(FIXTURE.configured_answered)).toBeNull();
    expect(answeredBy({})).toBeNull();
    expect(answeredBy(null)).toBeNull();
  });
});

describe("the model screen", () => {
  it("lists each role's configured model and its fallback chain, including an unarmed tier", async () => {
    render(<LlmModelsPanel load={() => Promise.resolve(MODELS)} />);
    const tier0 = await screen.findByTestId("llm-tier-planning-0");
    expect(tier0.textContent).toBe(
      "Configured: openrouter/openai/gpt-oss-120b (providers: deepinfra/bf16, together, crusoe, parasail)",
    );
    expect(screen.getByTestId("llm-tier-planning-1").textContent).toBe(
      "Fallback 1: openrouter/openai/gpt-6-luna (providers: azure)",
    );
    expect(screen.getByTestId("llm-tier-planning-2").textContent).toBe(
      "Fallback 2: groq/openai/gpt-oss-120b (not available: no key set)",
    );
  });
});
