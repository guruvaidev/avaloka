import { describe, expect, it } from "vitest";
import { isDeferredTurn } from "@/lib/api/backendApi";

const base = {
  messages: [],
  analysis_task_id: "t-1",
  output_json: null,
  training_completed: false,
};

describe("isDeferredTurn", () => {
  it("treats a deferred training turn as deferred", () => {
    expect(
      isDeferredTurn({ ...base, training_status: "running", pending_status: "running", pending_kind: "training" } as any)
    ).toBe(true);
  });

  it("treats a deferred analysis turn as deferred", () => {
    expect(
      isDeferredTurn({ ...base, training_status: null, pending_status: "running", pending_kind: "analysis" } as any)
    ).toBe(true);
  });

  it("does not treat a finished turn as deferred", () => {
    expect(
      isDeferredTurn({ ...base, training_status: null, pending_status: null, pending_kind: null } as any)
    ).toBe(false);
  });
});