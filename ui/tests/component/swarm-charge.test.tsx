// rweb18b: the swarm panel showed each clone's nominal local-CPU charge as a
// plain "$0.0001" in a "Cost" column and a "mission cost", which reads as money
// spent. It now says what the CLI narration says (avaloka/swarm_narration.py):
// "compute $X at a nominal CPU rate", model calls only when there were some,
// and "under $0.0001" for amounts four decimals cannot show.
// The records are the swarm-demo fixture's, with the split the coordinator now
// records on clone.returned (tests/avaloka/test_swarm_cost_truth.py).
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import fixture from "../fixtures/swarm-trace.json";
import { SwarmDemoPanel } from "@/components/agent-trace/SwarmDemoPanel";
import type { SwarmMissionDoc, SwarmRecord } from "@/lib/agent-trace/swarm";

vi.mock("@/integrations/supabase/external-client", () => ({ supabase: { auth: { getSession: vi.fn() } } }));

const mission = fixture.mission as unknown as SwarmMissionDoc;
const limits = mission.limits!;

/** clone.returned with its compute/model split; ava-01 also made a model call; ava-00's charge is tiny. */
function withSplit(records: SwarmRecord[]): SwarmRecord[] {
  return records.map((r) => {
    if (r.type === "clone.returned") {
      return r.clone_id === "ava-01"
        ? { ...r, compute_cost_usd: 0.000088, model_cost_usd: 0.0123 }
        : { ...r, compute_cost_usd: 0.00003, model_cost_usd: 0 };
    }
    if (r.type === "swarm.finished") {
      const economics = { ...((r.economics as Record<string, unknown>) ?? {}), compute_cost_usd: 0.000118, model_cost_usd: 0.0123, mission_charge_usd: 0 };
      return { ...r, economics };
    }
    return r;
  });
}

function fakeApi(records: SwarmRecord[]) {
  return {
    getCapabilities: vi.fn(async () => ({ features: { swarm: { enabled: true, limits, limited: true, message: limits.summary } } })),
    startSwarmMission: vi.fn(async () => fixture.start_response),
    // Everything in one page: this test is about the words, not the polling.
    getSwarmRecords: vi.fn(async (_id: string, after = 0) => {
      const out = records.filter((r) => r.seq > after);
      return { mission_id: mission.mission_id, status: "done", records: out, cursor: out.length ? out[out.length - 1].seq : after, done: true };
    }),
    getSwarmMission: vi.fn(async () => ({ ...mission, records: undefined })),
    listSwarmMissions: vi.fn(async () => ({ missions: [] })),
    getSwarmClone: vi.fn(),
    getSwarmEpisodes: vi.fn(),
    getSwarmTelemetry: vi.fn(),
  };
}

describe("Swarm panel charges", () => {
  it("names the nominal compute charge and a model call, never a bare cost", async () => {
    render(<SwarmDemoPanel api={fakeApi(withSplit(mission.records as SwarmRecord[])) as never} pollMs={5} />);
    fireEvent.click(await screen.findByRole("button", { name: "Start" }));
    await waitFor(() => expect(screen.getByLabelText(/^Ava-01: returned/)).toBeInTheDocument());

    const ava00 = screen.getByLabelText(/^Ava-00: returned/);
    const ava01 = screen.getByLabelText(/^Ava-01: returned/);
    expect(ava00).toHaveTextContent("compute under $0.0001 at a nominal CPU rate");
    expect(ava00).not.toHaveTextContent("model calls");                 // none were made
    expect(ava01).toHaveTextContent("compute $0.0001 at a nominal CPU rate · model calls $0.0123");

    const convergence = await screen.findByTestId("convergence");
    const table = within(convergence).getByRole("table", { name: "Contributions" });
    expect(table).not.toHaveTextContent(/\bCost\b/);
    expect(table).toHaveTextContent("compute under $0.0001 at a nominal CPU rate");

    await waitFor(() => expect(screen.getByTestId("finished")).toBeInTheDocument());
    const finished = screen.getByTestId("finished");
    expect(finished).not.toHaveTextContent(/mission cost/);
    expect(finished).toHaveTextContent("compute $0.0001 at a nominal CPU rate · model calls $0.0123");
  });

  it("says what an older total is made of when the record has no split", async () => {
    render(<SwarmDemoPanel api={fakeApi(mission.records as SwarmRecord[]) as never} pollMs={5} />);
    fireEvent.click(await screen.findByRole("button", { name: "Start" }));
    await waitFor(() => expect(screen.getByLabelText(/^Ava-01: returned/)).toBeInTheDocument());
    expect(screen.getByLabelText(/^Ava-01: returned/)).toHaveTextContent("a nominal compute rate plus any model calls");
  });
});
