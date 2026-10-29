// Regression tests for the Data View "Analysis results" run list: rebuilding
// from thread messages, prompt/time capture, multi-table pills, navigation
// bounds and CSV file names.

import { describe, expect, it } from "vitest";
//import type { AnalysisMessage } from "@/lib/analysis-messages";
import type { AnalysisMessage } from "../../src/lib/analysis-messages";
import {
  analysisCsvName,
  clampAnalysisIndex,
  extractTableRunsFromMessages,
  formatRunTime,
  resultDisplayPrompt,
  sampleCsvName,
  tablesForRun,
  type TableRun,
} from "../../src/lib/analysis-results";

let seq = 0;
function msg(role: "user" | "assistant", content: string, output: unknown = null): AnalysisMessage {
  seq += 1;
  return {
    id: `m${seq}`,
    analysis_id: "a1",
    role,
    content,
    output,
    author_id: null,
    created_at: `2026-10-08T21:1${seq % 10}:00+05:30`,
  };
}

describe("extractTableRunsFromMessages", () => {
  it("rebuilds one run per analysis from thread messages, with prompt and time", () => {
    const rows = [
      msg("user", "Average video views by country"),
      msg("assistant", "Done.", { output_json: [{ country: "IN", views: 10 }], section_id: "s1" }),
      msg("user", "Now by category"),
      msg("assistant", "Done.", { output_json: [{ category: "A", views: 5 }] }),
    ];
    const runs = extractTableRunsFromMessages(rows, "main");
    expect(runs).toHaveLength(2);
    expect(runs[0]!.prompt).toBe("Average video views by country");
    expect(runs[0]!.messageId).toBe("m1");
    expect(runs[0]!.ranAt).toBe(rows[1]!.created_at);
    expect(runs[0]!.rows).toEqual([{ country: "IN", views: 10 }]);
    expect(runs[1]!.prompt).toBe("Now by category");
    // Sample isolation: runs come only from assistant outputs, never from samples.
    expect(runs.every((r) => r.rows.length === 1)).toBe(true);
  });

  it("keeps several markdown tables of one reply as extra tables of a single run", () => {
    const content = [
      "| a | b |",
      "| - | - |",
      "| 1 | 2 |",
      "",
      "| c | d |",
      "| - | - |",
      "| 3 | 4 |",
    ].join("\n");
    const runs = extractTableRunsFromMessages([msg("user", "list things"), msg("assistant", content)], "main");
    expect(runs).toHaveLength(1);
    const tables = tablesForRun(runs[0]!);
    expect(tables).toHaveLength(2);
    expect(tables[0]!.rows).toEqual([{ a: "1", b: "2" }]);
    expect(tables[1]!.rows).toEqual([{ c: "3", d: "4" }]);
  });

  it("uses a backend output_tables payload as the run's tables", () => {
    const runs = extractTableRunsFromMessages(
      [
        msg("user", "two tables please"),
        msg("assistant", "Done.", {
          output_tables: [
            { name: "Top", rows: [{ x: 1 }] },
            { name: "Bottom", rows: [{ y: 2 }] },
          ],
        }),
      ],
      "main",
    );
    expect(runs).toHaveLength(1);
    const tables = tablesForRun(runs[0]!);
    expect(tables.map((t) => t.name)).toEqual([undefined, "Bottom"]);
    expect(tables[1]!.rows).toEqual([{ y: 2 }]);
  });

  it("ignores pasted analyses and replies without tables", () => {
    const rows = [
      msg("user", "hi"),
      msg("assistant", "Just text, no table."),
      msg("assistant", "", { pasted_analyses: [{ id: "p1", rows: [{ a: 1 }] }] }),
    ];
    expect(extractTableRunsFromMessages(rows, "main")).toHaveLength(0);
  });
});

describe("navigation bounds", () => {
  it("clamps the selected index into range", () => {
    expect(clampAnalysisIndex(0, 3)).toBe(0);
    expect(clampAnalysisIndex(2, 3)).toBe(2);
    expect(clampAnalysisIndex(5, 3)).toBe(2);
    expect(clampAnalysisIndex(-1, 3)).toBe(0);
    expect(clampAnalysisIndex(0, 0)).toBe(0);
  });
});

describe("prompt display", () => {
  const run = (patch: Partial<TableRun>): TableRun => ({ id: "r", sectionId: "s", title: "Result 1", rows: [], ...patch });
  it("shows the user's prompt and falls back to Analysis {i}", () => {
    expect(resultDisplayPrompt(run({ prompt: "Average subscribers by category" }), 0)).toBe(
      "Average subscribers by category",
    );
    expect(resultDisplayPrompt(run({ prompt: null }), 1)).toBe("Analysis 2");
    expect(resultDisplayPrompt(run({ prompt: "", title: "Restored result" }), 2)).toBe("Analysis 3");
  });
});

describe("time and CSV names", () => {
  it("formats the run time and tolerates missing values", () => {
    expect(formatRunTime(null)).toBe("");
    expect(formatRunTime("not-a-date")).toBe("");
    expect(formatRunTime("2026-10-08T21:15:00+05:30")).toMatch(/\d{1,2}:\d{2}/);
  });

  it("names CSV downloads after the dataset and view", () => {
    expect(sampleCsvName("mortgage_loans_test.csv")).toBe("mortgage_loans_test_sample.csv");
    expect(analysisCsvName("mortgage_loans_test.csv", 2)).toBe("mortgage_loans_test_analysis_2.csv");
    expect(analysisCsvName("My Data", 1)).toBe("My_Data_analysis_1.csv");
  });
});
