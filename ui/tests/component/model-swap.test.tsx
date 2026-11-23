// Owner ruling 2026-11-02: a model swap is never silent. The fixture is what the
// backend's trace recorder writes for a fallback (tests/test_default_openrouter_gptoss.py
// asserts the backend produces exactly these records).
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import fixture from "../fixtures/agent-trace.json";
import { StepDetail } from "@/components/agent-trace/TraceSteps";
import type { AgentTrace, TraceStep } from "@/lib/agent-trace/types";

vi.mock("@/integrations/supabase/external-client", () => ({ supabase: { auth: { getSession: vi.fn() } } }));

const source = (rel: string) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf8");
const SWAP = JSON.parse(source("../fixtures/model-swap.json"));
const MODELS = JSON.parse(source("../fixtures/llm-models.json"));

// Loaded through import.meta.glob so that a tree without these modules (one that
// swaps models silently) fails on an assertion, not on import.
type LibMod = typeof import("../../src/lib/agent-trace/model-swap");
type PanelMod = typeof import("../../src/components/settings/LlmModelsPanel");
const libMods = import.meta.glob<LibMod>("../../src/lib/agent-trace/model-swap.ts");
const panelMods = import.meta.glob<PanelMod>("../../src/components/settings/LlmModelsPanel.tsx");
let modelSwapLine: LibMod["modelSwapLine"];
let LlmModelsPanel: PanelMod["LlmModelsPanel"];

beforeAll(async () => {
  for (const [mods, name] of [
    [libMods, "ui/src/lib/agent-trace/model-swap.ts"],
    [panelMods, "ui/src/components/settings/LlmModelsPanel.tsx"],
  ] as const) {
    expect(Object.values(mods), `${name} is missing: a model swap would be silent`).toHaveLength(1);
  }
  ({ modelSwapLine } = await Object.values(libMods)[0]());
  ({ LlmModelsPanel } = await Object.values(panelMods)[0]());
});

const trace = fixture.trace as unknown as AgentTrace;
const turn = trace.turns[0];
const step = { ...(turn.steps[1] as TraceStep), inference: SWAP.inference } as TraceStep;

describe("which model answered, on the agent trace", () => {
  it("a call answered by a fallback tier says which model and why", () => {
    render(<StepDetail step={step} turn={turn} />);
    const row = screen.getByTestId("trace-model-swap");
    expect(row.textContent).toBe(
      "configured openai/gpt-4o → answered by openai/gpt-6-luna via openrouter " +
        "(no allowed zero-retention provider serves openai/gpt-4o)",
    );
  });

  it("the failed call names its recorded reason", () => {
    render(<StepDetail step={step} turn={turn} />);
    expect(screen.getByText(/failed · TierUnavailable · no allowed zero-retention provider serves openai\/gpt-4o/)).toBeTruthy();
  });

  it("a call the configured model answered shows no swap line", () => {
    expect(modelSwapLine(SWAP.inference[0])).toBeNull();
    expect(modelSwapLine({ provider: "openrouter", model: "openai/gpt-oss-120b", outcome: "success" })).toBeNull();
  });
});

describe("the model screen", () => {
  it("lists each role's configured model and its fallback chain, including an unarmed tier", async () => {
    render(<LlmModelsPanel load={() => Promise.resolve(MODELS)} />);
    expect((await screen.findByTestId("llm-tier-planning-0")).textContent).toBe(
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
