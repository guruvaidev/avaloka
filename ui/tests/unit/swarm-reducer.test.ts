// The swarm reducer, fed the records a real two-clone mission produced
// (tests/fixtures/swarm-trace.json, written by scripts/gen_swarm_trace_fixture.py).
import { describe, expect, it } from "vitest";
import fixture from "../fixtures/swarm-trace.json";
import { emptySwarm, reduceSwarm, viewFromMission, type SwarmMissionDoc, type SwarmRecord } from "@/lib/agent-trace/swarm";

const mission = fixture.mission as unknown as SwarmMissionDoc;
const records = mission.records as SwarmRecord[];

describe("reduceSwarm", () => {
  it("folds a whole mission into clones, steps, merge and the end", () => {
    const view = reduceSwarm(records);
    expect(records).toHaveLength(32);
    expect(view.status).toBe("done");
    expect(view.missionId).toBe(mission.mission_id);
    expect(view.cursor).toBe(32);
    expect(view.request).toMatchObject({ requestedClones: 3, grantedClones: 2, limited: true });
    expect(view.request!.message).toMatch(/limited to 2 clones/);
    expect(view.sharding).toMatchObject({ clones: 2, totalRows: 1200, rowsPerClone: [600, 600] });
    expect(view.clones.map((c) => c.name)).toEqual(["Ava-00", "Ava-01"]);
    for (const c of view.clones) {
      expect(c.status).toBe("done");
      expect(c.rows).toBe(600);
      expect(c.columns).toHaveLength(7);
      expect(c.steps.map((s) => s.firefly)).toEqual([
        "data_scout", "sampling_specialist", "data_engineer", "analysis_planner", "validator", "reporter",
      ]);
      expect(c.steps.every((s) => s.status === "ok" && s.output && s.humanRole)).toBe(true);
      expect(c.verdict).toBe("pass");
    }
    expect(view.merge).toMatchObject({ verdict: "pass", nRows: 1200, rules: { verdict: "worst case" } });
    expect(view.merge!.contributions).toHaveLength(2);
    expect(view.finished).toMatchObject({ status: "done" });
    expect(view.finished!.economics).toHaveProperty("estimated_manual_effort_hours");
  });

  it("is incremental: pages applied one at a time give the same view, and a repeated page changes nothing", () => {
    const whole = reduceSwarm(records);
    let view = emptySwarm();
    for (let i = 0; i < records.length; i += 5) view = reduceSwarm(records.slice(i, i + 5), view);
    expect(view).toEqual(whole);
    expect(reduceSwarm(records.slice(0, 10), view)).toBe(view);
    expect(reduceSwarm([], view)).toBe(view);
  });

  it("does not mutate the previous view", () => {
    const first = reduceSwarm(records.slice(0, 6));
    const snapshot = JSON.stringify(first);
    reduceSwarm(records.slice(6), first);
    expect(JSON.stringify(first)).toBe(snapshot);
  });

  it("shows a running step until its completion arrives, then settles the clone when it returns", () => {
    const firstStart = records.find((r) => r.type === "clone.step.started")!;
    const upTo = records.filter((r) => r.seq <= firstStart.seq);
    let view = reduceSwarm(upTo);
    const clone = view.clones.find((c) => c.cloneId === firstStart.clone_id)!;
    expect(clone.status).toBe("working");
    expect(clone.steps).toHaveLength(1);
    expect(clone.steps[0]).toMatchObject({ status: "running", firefly: "data_scout", output: null });
    const done = records.find((r) => r.type === "clone.step.completed" && r.step_id === firstStart.step_id)!;
    view = reduceSwarm(records.filter((r) => r.seq > firstStart.seq && r.seq <= done.seq), view);
    const step = view.clones.find((c) => c.cloneId === firstStart.clone_id)!.steps[0];
    expect(step.status).toBe("ok");
    expect(step.output).toMatch(/Profiled 7 columns/);
    expect(step.humanRole).toBe("Data analyst");
    const returned = records.find((r) => r.type === "clone.returned" && r.clone_id === firstStart.clone_id)!;
    view = reduceSwarm(records.filter((r) => r.seq > done.seq && r.seq <= returned.seq), view);
    const back = view.clones.find((c) => c.cloneId === firstStart.clone_id)!;
    expect(back.status).toBe("done");
    expect(back.steps).toHaveLength(6);
    expect(view.status).toBe("running"); // the other clone and the merge are still to come
  });

  it("narrates in Avaloka's voice, and says the limit in amber", () => {
    const view = reduceSwarm(records);
    const texts = view.narration.map((l) => l.text);
    expect(texts[0]).toMatch(/^Cloning myself into 2 copies/);
    expect(view.narration[1]).toMatchObject({ tone: "warn" });
    expect(texts[1]).toMatch(/you asked for 3, so 2 will run/);
    expect(texts).toContainEqual(expect.stringMatching(/^Sharded 1,200 rows into 2 shards/));
    expect(texts).toContainEqual(expect.stringMatching(/^Ava-00 takes shard 0: 600 rows, 7 columns/));
    expect(texts).toContainEqual(expect.stringMatching(/^Ava-01 is back in .*verdict pass/));
    expect(texts).toContainEqual(expect.stringMatching(/^Converging 2 answers over 1,200 rows: verdict pass \(worst case rule\)/));
    expect(texts[texts.length - 1]).toMatch(/^Done in/);
  });

  it("marks a failed step, a failed clone and a failed mission as such", () => {
    const failing: SwarmRecord[] = [
      { seq: 1, ts: "t", type: "swarm.requested", mission_id: "m", kind: "train", goal: "g", requested_clones: 1, granted_clones: 1, granted_parallel: 1, limited: false },
      { seq: 2, ts: "t", type: "clone.dispatched", mission_id: "m", clone_id: "ava-00", name: "Ava-00", shard_id: 0, received: { rows: 10, columns: ["a"] } },
      { seq: 3, ts: "t", type: "clone.step.started", mission_id: "m", clone_id: "ava-00", step_id: "ava-00/s01", step_no: 1, firefly: "model_scientist" },
      { seq: 4, ts: "t", type: "clone.step.completed", mission_id: "m", clone_id: "ava-00", step_id: "ava-00/s01", step_no: 1, firefly: "model_scientist", outcome: "failed", error_class: "ValueError" },
      { seq: 5, ts: "t", type: "clone.returned", mission_id: "m", clone_id: "ava-00", status: "failed", error_class: "ValueError" },
      { seq: 6, ts: "t", type: "swarm.finished", mission_id: "m", status: "failed", duration_ms: 12, error_class: "ValueError" },
    ];
    const view = reduceSwarm(failing);
    expect(view.status).toBe("failed");
    expect(view.clones[0]).toMatchObject({ status: "failed", errorClass: "ValueError" });
    expect(view.clones[0].steps[0]).toMatchObject({ status: "failed", errorClass: "ValueError" });
    expect(view.narration.filter((l) => l.tone === "bad").map((l) => l.text)).toEqual([
      "Ava-00: model_scientist failed (ValueError).",
      "Ava-00 came back without an answer (ValueError).",
      "The mission failed (ValueError).",
    ]);
  });

  it("highlights the clone whose model the merge selected", () => {
    const selected = records.map((r) =>
      r.type === "swarm.merged"
        ? { ...r, selected_model: { from_shard: 1, score: 0.9 }, contributions: (r.contributions as Array<Record<string, unknown>>).map((c) => ({ ...c, selected: c.clone_id === "ava-01" })) }
        : r,
    );
    const view = reduceSwarm(selected);
    expect(view.merge!.selectedCloneId).toBe("ava-01");
    expect(view.clones.map((c) => c.selected)).toEqual([false, true]);
    expect(view.narration.map((l) => l.text)).toContainEqual(expect.stringMatching(/Ava-01's model is the one I stand behind/));
  });

  it("rebuilds the same view from a stored mission document", () => {
    expect(viewFromMission(mission)).toEqual(reduceSwarm(records));
  });
});
