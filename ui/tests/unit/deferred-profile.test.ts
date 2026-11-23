import { describe, expect, it } from "vitest";
import {
  PROFILE_FAILED_FALLBACK,
  deferredProfileOutcome,
  pendingProfileDatasetId,
  waitForDeferredProfile,
} from "@/lib/deferred-profile";

describe("pendingProfileDatasetId", () => {
  it("finds a pending single-file upload", () => {
    expect(pendingProfileDatasetId({ dataset_id: "d1", profiling_status: "pending" })).toBe("d1");
  });

  it("finds a pending multi-file upload by its first dataset", () => {
    expect(
      pendingProfileDatasetId({ datasets: [{ dataset_id: "d2", profiling_status: "pending" }] }),
    ).toBe("d2");
  });

  it("ignores an upload whose profile came back inline, and old responses", () => {
    expect(pendingProfileDatasetId({ dataset_id: "d1", profiling_status: "full_profile" })).toBeNull();
    expect(pendingProfileDatasetId({ dataset_id: "d1" })).toBeNull();
    expect(pendingProfileDatasetId(null)).toBeNull();
  });
});

describe("deferredProfileOutcome", () => {
  it("keeps waiting while pending", () => {
    expect(deferredProfileOutcome({ profiling_status: "pending" })).toBeNull();
  });

  it("returns the landed charts", () => {
    const cfg = { charts: [{ title: "t" }] };
    expect(
      deferredProfileOutcome({ profiling_status: "full_profile", visualization_config: cfg, visualization_status: "ready" }),
    ).toEqual({ kind: "ready", visualizationConfig: cfg });
  });

  it("reports a failure in the server's words, or a plain fallback", () => {
    expect(deferredProfileOutcome({ profiling_status: "failed", profiling_error: "No profile." })).toEqual({
      kind: "failed",
      message: "No profile.",
    });
    expect(deferredProfileOutcome({ profiling_status: "failed" })).toEqual({
      kind: "failed",
      message: PROFILE_FAILED_FALLBACK,
    });
  });
});

describe("waitForDeferredProfile", () => {
  it("polls until the profile lands", async () => {
    const answers = [{ profiling_status: "pending" }, { profiling_status: "failed", profiling_error: "x" }];
    let calls = 0;
    const outcome = await waitForDeferredProfile(async () => answers[calls++], { intervalMs: 1 });
    expect(calls).toBe(2);
    expect(outcome).toEqual({ kind: "failed", message: "x" });
  });

  it("gives up quietly after repeated network errors", async () => {
    const outcome = await waitForDeferredProfile(
      async () => {
        throw new Error("offline");
      },
      { intervalMs: 1 },
    );
    expect(outcome).toEqual({ kind: "gone" });
  });

  it("stops when aborted", async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    const outcome = await waitForDeferredProfile(async () => ({ profiling_status: "pending" }), {
      intervalMs: 1,
      signal: ctrl.signal,
    });
    expect(outcome).toEqual({ kind: "gone" });
  });
});
