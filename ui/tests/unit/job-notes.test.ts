import { describe, expect, it } from "vitest";
import { batchIdOfTranscriptRow, isThreadJobNotification, jobNoteChatMessages } from "@/lib/job-notes";

const batch = (id: string, text = "✓ Data transfer finished.") => ({
  batch_id: id, thread_id: "t-1", text, job_ids: ["j"], count: 1,
});

describe("jobNoteChatMessages", () => {
  it("turns a batch into one ai message tagged as a job note, once", () => {
    const seen = new Set<string>();
    const first = jobNoteChatMessages([batch("b1")], "10:00", seen);
    expect(first).toHaveLength(1);
    expect(first[0]).toMatchObject({ id: "job-note-b1", role: "ai", jobNote: true, batchId: "b1", time: "10:00" });
    expect(first[0].content).toBe("✓ Data transfer finished.");
    expect(jobNoteChatMessages([batch("b1")], "10:01", seen)).toHaveLength(0);   // same batch again: nothing
  });

  it("strips the model frame people should not see", () => {
    const seen = new Set<string>();
    const msgs = jobNoteChatMessages(
      [batch("b2", "[Background job update — written by the system, not an assistant reply]\n✓ Model training finished.")],
      "t", seen,
    );
    expect(msgs[0].content).toBe("✓ Model training finished.");
  });

  it("ignores empty, null and malformed input", () => {
    const seen = new Set<string>();
    expect(jobNoteChatMessages(null, "t", seen)).toEqual([]);
    expect(jobNoteChatMessages(undefined, "t", seen)).toEqual([]);
    expect(jobNoteChatMessages([{ ...batch("b3"), batch_id: "" }], "t", seen)).toEqual([]);
  });
});

describe("isThreadJobNotification", () => {
  it("matches only a bell item about a job of this thread", () => {
    expect(isThreadJobNotification({ metadata: { thread_id: "t-1", job_id: "j-1" } }, "t-1")).toBe(true);
    expect(isThreadJobNotification({ metadata: { thread_id: "t-2", job_id: "j-1" } }, "t-1")).toBe(false);
    expect(isThreadJobNotification({ metadata: { thread_id: "t-1" } }, "t-1")).toBe(false);      // not a job
    expect(isThreadJobNotification({ metadata: { thread_id: "t-1", job_id: "j" } }, null)).toBe(false);
    expect(isThreadJobNotification({}, "t-1")).toBe(false);
  });
});

describe("batchIdOfTranscriptRow", () => {
  it("reads the batch id the API stores on a note's transcript row", () => {
    expect(batchIdOfTranscriptRow({ job_note: { batch_id: "b9", job_ids: ["j"] } })).toBe("b9");
    expect(batchIdOfTranscriptRow({ output_json: [] })).toBeNull();
    expect(batchIdOfTranscriptRow(null)).toBeNull();
  });
});
