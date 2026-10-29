import { describe, expect, it } from "vitest";
import { canReviewUpload, issueHasDetails, readableDatasetName, reviewDatasetName, reviewIssues } from "../../src/lib/upload-review";
describe("upload dataset names", () => {
  it("removes copy suffix and extension", () => expect(readableDatasetName("Global YouTube Statistics (1).csv")).toBe("Global YouTube Statistics"));
  it("removes underscore copy suffixes and normalizes separators", () => expect(readableDatasetName("Global_YouTube_Statistics__1.csv")).toBe("Global YouTube Statistics"));
  it("preserves legitimate year suffixes", () => expect(readableDatasetName("mortgage_2025.csv")).toBe("Mortgage 2025"));
  it("collapses spaces and title cases", () => expect(readableDatasetName("mortgage--LOANS__test.json")).toBe("Mortgage Loans Test"));
  it("prefers profile title and limits names to 80 characters", () => {
    expect(reviewDatasetName({ filename: "data.csv", profiling_result: { dataset_title: "Video audience" } })).toBe("Video audience");
    expect(readableDatasetName(`${"a".repeat(100)}.csv`)).toHaveLength(80);
  });
});
describe("upload review readiness", () => {
  it("waits for 100 percent and profiling completion for every file", () => {
    expect(canReviewUpload(2, 2, 99, false)).toBe(false);
    expect(canReviewUpload(2, 2, 100, true)).toBe(false);
    expect(canReviewUpload(2, 1, 100, false)).toBe(false);
    expect(canReviewUpload(2, 2, 100, false)).toBe(true);
    expect(canReviewUpload(0, 0, 100, false)).toBe(false);
  });
});
describe("profile issue details", () => {
  it("makes supplied counts expandable and limits duplicate examples to five", () => {
    const issues = reviewIssues({ profiling_result: { duplicate_rows: 12, duplicate_row_numbers: [2, 3, 4, 5, 6, 7] } });
    expect(issues[0].summary).toBe("12 duplicate rows");
    expect(issueHasDetails(issues[0])).toBe(true);
    expect(issues[0].details[1]).toBe("Example row numbers: 2, 3, 4, 5, 6");
  });
  it("keeps issues without details plain and never invents default issues", () => {
    expect(reviewIssues({})).toEqual([]);
    const issue = reviewIssues({ profiling_result: { red_flags: ["Missing values reported"] } })[0];
    expect(issueHasDetails(issue)).toBe(false);
  });
  it("uses actual missing metadata without inferring cell counts from a sample", () => {
    const issue = reviewIssues({ samples: [{ a: null }], visualization_config: { column_stats: [{ name: "a", missing_pct: 10 }] } })[0];
    expect(issue.columns).toEqual([{ name: "a", count: null, percent: 10 }]);
    expect(issueHasDetails(issue)).toBe(true);
  });
});