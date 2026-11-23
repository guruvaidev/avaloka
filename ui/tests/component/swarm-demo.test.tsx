// The live swarm demo, driven by a fake API that pages out the records a real
// two-clone mission produced (tests/fixtures/swarm-trace.json), a few per poll,
// the way the server's /records?after= endpoint does.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import fixture from "../fixtures/swarm-trace.json";
import { SwarmDemoPanel } from "@/components/agent-trace/SwarmDemoPanel";
import { SwarmView } from "@/routes/_authenticated/swarm";
import type { SwarmMissionDoc, SwarmRecord } from "@/lib/agent-trace/swarm";

vi.mock("@/integrations/supabase/external-client", () => ({ supabase: { auth: { getSession: vi.fn() } } }));
vi.mock("@tanstack/react-router", async (orig) => ({
  ...(await orig<typeof import("@tanstack/react-router")>()),
  Link: ({ children }: { children: React.ReactNode }) => <a>{children}</a>,
}));

const mission = fixture.mission as unknown as SwarmMissionDoc;
const limits = mission.limits!;

/** The fixture's records with Ava-01's model selected by the merge, as a training mission would. */
function withSelection(records: SwarmRecord[]): SwarmRecord[] {
  return records.map((r) =>
    r.type === "swarm.merged"
      ? {
          ...r,
          selected_model: { from_shard: 1, score: 0.87, name: "gbm" },
          contributions: (r.contributions as Array<Record<string, unknown>>).map((c) => ({ ...c, selected: c.clone_id === "ava-01" })),
        }
      : r,
  );
}

/** A server that holds `records` and pages them out `page` at a time per poll. */
function fakeApi(records: SwarmRecord[], page = 6) {
  const start = fixture.start_response;
  return {
    getCapabilities: vi.fn(async () => ({ features: { swarm: { enabled: true, limits, limited: true, message: limits.summary } } })),
    startSwarmMission: vi.fn(async () => start),
    getSwarmRecords: vi.fn(async (_id: string, after = 0) => {
      const rest = records.filter((r) => r.seq > after);
      const out = rest.slice(0, page);
      const cursor = out.length ? out[out.length - 1].seq : after;
      return { mission_id: mission.mission_id, status: rest.length <= page ? "done" : "running", records: out, cursor, done: rest.length <= page };
    }),
    getSwarmMission: vi.fn(async () => ({ ...mission, records: undefined })),
    listSwarmMissions: vi.fn(async () => ({
      missions: [{ mission_id: mission.mission_id, status: "done", kind: "analyze", goal: mission.goal, started_at: mission.started_at, ended_at: mission.ended_at, clones: 2, verdict: "pass" }],
    })),
    getSwarmClone: vi.fn(async (_id: string, cloneId: string) => ({ ...mission.clones.find((c) => c.clone_id === cloneId)!, mission_id: mission.mission_id, records: [] })),
    getSwarmEpisodes: vi.fn(async () => fixture.episodes),
    getSwarmTelemetry: vi.fn(async () => ({ mission_id: mission.mission_id, status: "done", events: [{ type: "swarm.completed", clones_granted: 2, outcome: "done" }] })),
  };
}

describe("SwarmDemoPanel", () => {
  it("bounds the clone count by the edition's cap and states the limit", async () => {
    render(<SwarmDemoPanel api={fakeApi([]) as never} pollMs={5} />);
    const clones = (await screen.findByLabelText(/Clones \(up to 2\)/)) as HTMLInputElement;
    expect(clones.max).toBe("2");
    expect(screen.getByText(limits.summary)).toBeInTheDocument();
  });

  it("shows the fleet multiply, work and converge as the records arrive", async () => {
    const api = fakeApi(withSelection(mission.records as SwarmRecord[]));
    render(<SwarmDemoPanel api={api as never} pollMs={5} />);
    fireEvent.click(await screen.findByRole("button", { name: "Start" }));
    await waitFor(() =>
      expect(api.startSwarmMission).toHaveBeenCalledWith({ source: { sample: "churn" }, goal: expect.any(String), kind: "analyze", workers: 2 }),
    );

    // Avaloka announces the fleet; the limit is said in amber, not only coloured.
    const narration = await screen.findByTestId("narration");
    await waitFor(() => expect(narration).toHaveTextContent(/Cloning myself into 2 copies/));
    expect(within(narration).getByRole("note")).toHaveTextContent(/you asked for 3, so 2 will run/);

    // Clone cards appear as they are dispatched, with rows and the column count.
    await waitFor(() => expect(screen.getAllByTestId("clone-card")).toHaveLength(2));
    const card0 = screen.getByLabelText(/^Ava-00:/);
    expect(card0).toHaveTextContent("600 rows · 7 columns");

    // Steps accumulate; a step runs before it completes.
    await waitFor(() => expect(within(card0).getAllByRole("listitem").length).toBeGreaterThan(0));
    await waitFor(() => expect(screen.getAllByText(/data scout/).length).toBeGreaterThan(0));
    await waitFor(() => {
      const steps = within(screen.getByLabelText("Ava-00 steps")).getAllByRole("listitem");
      expect(steps).toHaveLength(6);
      expect(steps.every((s) => s.getAttribute("data-step-status") === "ok")).toBe(true);
    });
    expect(within(card0).getByText(/Profiled 7 columns x 600 rows/)).toBeInTheDocument();
    expect(within(card0).getByText(/Data analyst/)).toBeInTheDocument();

    // The cards settle on their verdicts, and the convergence panel highlights the selected clone.
    await waitFor(() => expect(screen.getByLabelText(/^Ava-01: returned/)).toBeInTheDocument());
    const convergence = await screen.findByTestId("convergence");
    expect(convergence).toHaveTextContent("verdict pass");
    expect(convergence).toHaveTextContent("rule: worst case");
    expect(convergence).toHaveTextContent("1,200 rows");
    const rows = within(within(convergence).getByRole("table", { name: "Contributions" })).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveAttribute("data-selected", "true");
    expect(rows[1]).toHaveTextContent("Ava-01");
    expect(rows[0]).toHaveAttribute("data-selected", "false");
    expect(screen.getByLabelText(/^Ava-01: returned/)).toHaveTextContent("selected by the merge");
    await waitFor(() => expect(screen.getByTestId("finished")).toHaveTextContent(/done in \d+ ms/));
    expect(screen.getByTestId("finished")).toHaveTextContent(/17.7 h of manual effort \(rough heuristic estimate, not measured\)/);
    expect(narration).toHaveTextContent(/Done in/);

    // Polling stopped once the server said done, and the mission list was refreshed.
    const polls = api.getSwarmRecords.mock.calls.length;
    await new Promise((r) => setTimeout(r, 40));
    expect(api.getSwarmRecords.mock.calls.length).toBe(polls);
    expect(api.getSwarmRecords.mock.calls.map((c) => c[1])).toEqual([0, 6, 12, 18, 24, 30]);
    await waitFor(() => expect(api.listSwarmMissions.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it("opens the per-clone document, the RL episodes and the telemetry projection", async () => {
    const api = fakeApi(mission.records as SwarmRecord[], 40);
    render(<SwarmDemoPanel api={api as never} pollMs={5} />);
    fireEvent.click(await screen.findByRole("button", { name: "Start" }));
    await screen.findByTestId("convergence");

    fireEvent.click(screen.getByRole("tab", { name: "Clone documents" }));
    await waitFor(() => expect(api.getSwarmClone).toHaveBeenCalledWith(mission.mission_id, "ava-00"));
    expect(await screen.findByText(/"human_role": "Data analyst"/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Ava-01" }));
    await waitFor(() => expect(api.getSwarmClone).toHaveBeenCalledWith(mission.mission_id, "ava-01"));

    fireEvent.click(screen.getByRole("tab", { name: "RL episodes" }));
    expect(await screen.findByText(/"reward": null/)).toBeInTheDocument();
    expect(screen.getByText(/"selected_by_merge": false/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Telemetry/ }));
    expect(await screen.findByText(/"type": "swarm.completed"/)).toBeInTheDocument();
    expect(screen.getByText(/No goal, no column, no output, no path/)).toBeInTheDocument();
  });

  it("reopens a finished mission from the list", async () => {
    const api = fakeApi(mission.records as SwarmRecord[], 40);
    render(<SwarmDemoPanel api={api as never} pollMs={5} />);
    const list = await screen.findByRole("list", { name: "Missions" });
    expect(list).toHaveTextContent("what drives churn");
    fireEvent.click(within(list).getByRole("button", { name: "Open" }));
    await waitFor(() => expect(api.getSwarmMission).toHaveBeenCalledWith(mission.mission_id));
    await waitFor(() => expect(screen.getAllByTestId("clone-card")).toHaveLength(2));
    expect(api.startSwarmMission).not.toHaveBeenCalled();
  });

  it("says why a mission was refused", async () => {
    const api = fakeApi([]);
    api.startSwarmMission.mockRejectedValueOnce(new Error("Failed to start the swarm mission: a training swarm requires a target column"));
    render(<SwarmDemoPanel api={api as never} pollMs={5} />);
    fireEvent.click(await screen.findByRole("button", { name: "Start" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/requires a target column/);
  });

  it("offers the analysis's own dataset when opened from one", async () => {
    const api = fakeApi([]);
    render(<SwarmDemoPanel api={api as never} threadId="thread-9" datasetName="churn.csv" pollMs={5} />);
    const dataset = (await screen.findByLabelText("Dataset")) as HTMLSelectElement;
    expect(dataset.value).toBe("thread");
    expect(within(dataset).getByRole("option", { name: "This analysis: churn.csv" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(api.startSwarmMission).toHaveBeenCalledWith(expect.objectContaining({ source: { thread_id: "thread-9" } })));
  });
});

describe("Swarm page", () => {
  it("renders the demo with links back to the agent trace", async () => {
    render(<SwarmView api={fakeApi([]) as never} pollMs={5} />);
    expect(screen.getByRole("heading", { name: "Swarm" })).toBeInTheDocument();
    expect(screen.getByText("Agent trace")).toBeInTheDocument();
    expect(await screen.findByTestId("swarm-demo")).toBeInTheDocument();
  });
});
