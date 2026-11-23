// J1: a deferred turn's poll never ends empty.
//
// The poller used to stop on "superseded" and the analysis view then removed
// its placeholder, so a browser user saw nothing at all although the server
// had the answer. It also ended an older turn's poll as soon as a newer turn
// started on the same thread, though the older turn kept running server-side
// and its answer was stored. Now: on "superseded" the poller follows the turn
// that replaced it (or the thread's newest turn) and resolves with its result.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { backendApi, type PendingTurnResponse } from "@/lib/api/backendApi";
import { startTrainingJob, stopAllTrainingJobs } from "@/lib/training-jobs";

const POLL_MS = 5_000;
const answer = { messages: [{ role: "assistant", content: "the taxi answer" }] };

function scripted(script: (id: string | null | undefined, n: number) => PendingTurnResponse) {
  const calls: Array<string | null | undefined> = [];
  vi.spyOn(backendApi, "getPendingTurn").mockImplementation(async (_t, _s, id) => {
    calls.push(id);
    return script(id, calls.length);
  });
  return calls;
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  stopAllTrainingJobs();
  vi.useRealTimers();
});

describe("deferred-turn poller on superseded", () => {
  it("follows the turn that replaced it and shows that turn's result", async () => {
    const calls = scripted((id, n) => {
      if (id === "old") return { status: "superseded", superseded_by: "new" };
      return n < 3 ? { status: "running" } : { status: "done", result: answer };
    });
    const pending = startTrainingJob({ threadId: "t1", deferredId: "old", sessionId: "s" });
    await vi.advanceTimersByTimeAsync(POLL_MS * 4);
    const out = await pending;
    expect(out.status).toBe("done");
    expect(out.result).toEqual(answer);
    expect(calls.slice(0, 3)).toEqual(["old", "new", "new"]);
  });

  it("without a named successor polls the thread's newest turn, never resolving empty", async () => {
    const calls = scripted((id) =>
      id === "old" ? { status: "superseded" } : { status: "done", result: answer },
    );
    const pending = startTrainingJob({ threadId: "t2", deferredId: "old", sessionId: "s" });
    await vi.advanceTimersByTimeAsync(POLL_MS * 3);
    const out = await pending;
    expect(out.status).toBe("done");
    expect(out.result).toEqual(answer);
    expect(calls[1] ?? null).toBeNull();
  });

  it("a newer turn on the same thread does not end the older turn's poll", async () => {
    scripted((id) =>
      id === "older"
        ? { status: "done", result: answer }
        : { status: "running" },
    );
    const older = startTrainingJob({ threadId: "t3", deferredId: "older", sessionId: "s" });
    void startTrainingJob({ threadId: "t3", deferredId: "newer", sessionId: "s" });
    await vi.advanceTimersByTimeAsync(POLL_MS * 2);
    const out = await older;
    expect(out.status).toBe("done");
    expect(out.result).toEqual(answer);
  });

  it("hands over to the newer turn's own job instead of showing its result twice", async () => {
    scripted((id) =>
      id === "old" ? { status: "superseded", superseded_by: "new" } : { status: "running" },
    );
    void startTrainingJob({ threadId: "t4", deferredId: "new", sessionId: "s" });
    const old = startTrainingJob({ threadId: "t4", deferredId: "old", sessionId: "s" });
    await vi.advanceTimersByTimeAsync(POLL_MS * 2);
    const out = await old;
    expect(out.status).toBe("superseded");
    expect(out.superseded_by).toBe("new");
  });
});
