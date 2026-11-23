// The agent trace page and panel, rendered from what the real backend emits
// (tests/fixtures/agent-trace.json, written by scripts/gen_agent_trace_fixture.py).
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import fixture from "../fixtures/agent-trace.json";
import { PlanGraphView } from "@/components/agent-trace/PlanGraphView";
import { AgentTracePanel } from "@/components/agent-trace/AgentTracePanel";
import { AgentTraceView } from "@/routes/_authenticated/agent-trace";
import type { AgentTrace, PlanGraph, SwarmCapability } from "@/lib/agent-trace/types";

vi.mock("@/integrations/supabase/external-client", () => ({ supabase: { auth: { getSession: vi.fn() } } }));
vi.mock("@tanstack/react-router", async (orig) => ({
  ...(await orig<typeof import("@tanstack/react-router")>()),
  Link: ({ children }: { children: React.ReactNode }) => <a>{children}</a>,
}));

const trace = fixture.trace as unknown as AgentTrace;
const graph = fixture.graph as unknown as PlanGraph;
const trainingGraph = fixture.graph_training as unknown as PlanGraph;
const liveGraph = fixture.graph_live as unknown as PlanGraph;
const swarm = fixture.swarm as unknown as SwarmCapability;

function fakeApi(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    getAgentTrace: vi.fn(async () => trace),
    getPlanGraph: vi.fn(async () => graph),
    getCapabilities: vi.fn(async () => ({ features: { swarm } })),
    getAgentMetrics: vi.fn(async () => ({ threads: 3, turns: 7, agents: trace.agents })),
    ...overrides,
  } as never;
}

describe("PlanGraphView", () => {
  it("shows how the prompt was acted upon by default: the question and the data lane only", () => {
    const { container } = render(<PlanGraphView graph={graph} />);
    const shown = [...container.querySelectorAll("[data-node-type]")].map((el) => el.getAttribute("data-node-type"));
    expect(new Set(shown)).toEqual(new Set(["question", "data", "operation", "result"]));
    expect(shown).toHaveLength(graph.nodes.filter((n) => n.type === "question" || n.lane === "data").length);
    // The columns each operation touches are on the node, and the load names the dataset's columns.
    expect(screen.getByText("Filter on monthly_charges")).toBeInTheDocument();
    expect(screen.getByText("⌗ annual, monthly_charges")).toBeInTheDocument();
    expect(screen.getByText(/⌗ customer_id, tenure_months/, { selector: "text" })).toBeInTheDocument();
    expect(screen.getByText("Group by region → mean")).toBeInTheDocument();
    expect(screen.getAllByText(/df\.groupby\('region'\)/).length).toBeGreaterThan(0);
    expect(screen.getByText(/East\s+·\s+61\.5/)).toBeInTheDocument();
    // The question's edge to the load survives the filter.
    expect(container.querySelectorAll("path[d]").length).toBeGreaterThanOrEqual(5);
  });

  it("brings the agents, decisions and checks back on one toggle", () => {
    const { container } = render(<PlanGraphView graph={graph} />);
    fireEvent.click(screen.getByRole("checkbox", { name: "Show agents and checks" }));
    expect(container.querySelectorAll("[data-node-id]")).toHaveLength(graph.nodes.length);
    for (const type of ["question", "agent", "decision", "check", "data", "operation", "result", "evidence"]) {
      expect(container.querySelector(`[data-node-type="${type}"]`), type).not.toBeNull();
    }
    expect(container.querySelector('[data-edge-type="retry"]')).not.toBeNull();
  });

  it("hides a real run's agent loop: 34 nodes become the 6 that touched the data", () => {
    const { container } = render(<PlanGraphView graph={liveGraph} />);
    expect(liveGraph.nodes).toHaveLength(34);
    expect(container.querySelectorAll("[data-node-id]")).toHaveLength(6);
    expect(screen.getByText(/Group by job_title/, { selector: "text" })).toBeInTheDocument();
    expect(screen.getByText("Append rows", { selector: "text" })).toBeInTheDocument();
    expect(screen.queryByText("Join")).toBeNull();
    expect(screen.queryByText(/^Write/)).toBeNull();
  });

  it("draws a training turn as train and evaluate nodes carrying the metric", () => {
    const { container } = render(<PlanGraphView graph={trainingGraph} />);
    expect(container.querySelector('[data-node-type="train"]')).not.toBeNull();
    expect(container.querySelector('[data-node-type="evaluate"]')).not.toBeNull();
    expect(screen.getByText("Train churn_predictor", { selector: "text" })).toBeInTheDocument();
    expect(screen.getByText("Evaluate · accuracy", { selector: "text" })).toBeInTheDocument();
    expect(screen.getByText("accuracy 0.9100 vs baseline 0.7300", { selector: "text" })).toBeInTheDocument();
    expect(screen.getByText(/beats the baseline/, { selector: "text" })).toBeInTheDocument();
    expect(screen.getByText(/target churned · features 2/, { selector: "text" })).toBeInTheDocument();
  });

  it("selects the trace step behind a clicked node", () => {
    const onSelect = vi.fn();
    render(<PlanGraphView graph={graph} onSelect={onSelect} showAgents />);
    const coder = graph.nodes.find((n) => n.label === "Coder · attempt 2")!;
    fireEvent.click(screen.getByRole("button", { name: /agent: Coder · attempt 2/ }));
    expect(onSelect).toHaveBeenCalledWith(coder.step_id, coder.id);
  });
});

describe("AgentTracePanel", () => {
  it("states the swarm limit plainly and summarises the turn", async () => {
    render(<AgentTracePanel threadId="thread-fixture" api={fakeApi()} />);
    expect(await screen.findByRole("note")).toHaveTextContent(
      "Swarm limited to 2 clones per mission on the free open-source edition, run locally.",
    );
    expect(screen.getByText("“What is the average monthly_charges by region?”")).toBeInTheDocument();
    expect(screen.getByText("Tokens in / out").nextSibling).toHaveTextContent("1.8k / 212");
  });

  it("opens a step's decision, calculation and inference from the timeline", async () => {
    render(<AgentTracePanel threadId="thread-fixture" api={fakeApi()} />);
    fireEvent.click(await screen.findByRole("tab", { name: /Steps/ }));
    const steps = screen.getByRole("list", { name: "Agent steps" });
    expect(within(steps).getAllByText("Coder")).toHaveLength(2); // the retry is visible
    fireEvent.click(within(steps).getByText("Avaloka"));
    const detail = screen.getByTestId("step-detail");
    expect(detail).toHaveTextContent("intent: statistical_analysis");
    expect(detail).toHaveTextContent("asks for an average by group");
    expect(detail).toHaveTextContent("openrouter/openai/gpt-oss-120b");
    expect(detail).toHaveTextContent("unpriced");
  });

  it("aggregates agents and shows which swarm clones ran", async () => {
    render(<AgentTracePanel threadId="thread-fixture" api={fakeApi()} />);
    fireEvent.click(await screen.findByRole("tab", { name: "Agents" }));
    const row = screen.getByText("coder", { selector: "div" }).closest("tr")!;
    expect(row).toHaveTextContent("2"); // activations
    fireEvent.click(screen.getByRole("tab", { name: "Swarm" }));
    expect(screen.getByText("Ava-Coder").closest("li")).toHaveTextContent("done");
    expect(screen.getByText("Ava-Scientist").closest("li")).toHaveTextContent("not run this turn");
  });

  it("says so when the thread has no trace", async () => {
    const err = Object.assign(new Error("Not Found"), { status: 404 });
    render(<AgentTracePanel threadId="nope" api={fakeApi({ getAgentTrace: vi.fn(async () => { throw err; }) })} />);
    expect(await screen.findByText("No agent trace for this analysis.")).toBeInTheDocument();
  });

  it("asks for the requested turn", async () => {
    const api = fakeApi();
    render(<AgentTracePanel threadId="thread-fixture" turnId="turn-9" api={api} />);
    await waitFor(() => expect((api as any).getAgentTrace).toHaveBeenCalledWith("thread-fixture", "turn-9"));
    expect((api as any).getPlanGraph).toHaveBeenCalledWith("thread-fixture", "turn-9");
  });
});

describe("Agent trace page", () => {
  it("renders one analysis's trace when given a thread", async () => {
    render(<AgentTraceView thread="thread-fixture" api={fakeApi()} />);
    expect(screen.getByRole("heading", { name: "Agent trace" })).toBeInTheDocument();
    expect(await screen.findByTestId("agent-trace-panel")).toBeInTheDocument();
  });

  it("shows every agent's aggregate when no thread is given and the console is open", async () => {
    const open = { features: { swarm, swarm_console: { enabled: true, reason: "open", page: "/swarm", message: null } } };
    render(<AgentTraceView api={fakeApi({ getCapabilities: vi.fn(async () => open) })} />);
    expect(await screen.findByText(/Across 7 recorded turns in 3 analyses/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Agent metrics" })).toBeInTheDocument();
  });

  it("keeps the cross-analysis metrics behind the console when it is closed", async () => {
    const api = fakeApi();
    render(<AgentTraceView api={api} />);
    expect(await screen.findByText(/available in Avaloka Professional and Enterprise/)).toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "Agent metrics" })).not.toBeInTheDocument();
    expect((api as unknown as { getAgentMetrics: ReturnType<typeof vi.fn> }).getAgentMetrics).not.toHaveBeenCalled();
  });
});
