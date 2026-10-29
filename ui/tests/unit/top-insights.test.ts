import { describe, expect, it } from "vitest";
import { chartScore, pickTopInsights, type TopInsightDataset } from "../../src/lib/top-insights";

const ds = (id: string, rows: number, charts: Record<string, unknown>[]): TopInsightDataset => ({
  id,
  name: `Sheet ${id}`,
  rows,
  visualization_config: { charts },
});

describe("pickTopInsights", () => {
  it("orders by importance, highest first", () => {
    const picks = pickTopInsights([ds("a", 10, [{ title: "a1", importance: 40 }, { title: "a2", importance: 90 }]), ds("b", 10, [{ title: "b1", importance: 70 }])]);
    expect(picks.map((p) => p.chart.title)).toEqual(["a2", "b1", "a1"]);
    expect(picks[0]).toMatchObject({ datasetId: "a", datasetName: "Sheet a", score: 90 });
  });

  it("takes at most 2 charts per dataset while other datasets still have charts", () => {
    const picks = pickTopInsights([
      ds("a", 10, [{ title: "a1", importance: 99 }, { title: "a2", importance: 98 }, { title: "a3", importance: 97 }]),
      ds("b", 10, [{ title: "b1", importance: 10 }, { title: "b2", importance: 9 }]),
      ds("c", 10, [{ title: "c1", importance: 5 }]),
    ]);
    expect(picks.map((p) => p.chart.title)).toEqual(["a1", "a2", "b1", "b2", "c1"]);
  });

  it("fills up to the limit beyond the cap once every dataset is exhausted", () => {
    const picks = pickTopInsights([
      ds("a", 10, [{ title: "a1", importance: 99 }, { title: "a2", importance: 98 }, { title: "a3", importance: 97 }, { title: "a4", importance: 96 }]),
      ds("b", 10, [{ title: "b1", importance: 10 }]),
    ]);
    expect(picks.map((p) => p.chart.title)).toEqual(["a1", "a2", "a3", "a4", "b1"]);
    expect(picks).toHaveLength(5);
    expect(picks.filter((p) => p.datasetId === "a")).toHaveLength(4);
  });

  it("falls back to 100 − 10 × (rank − 1) when importance is missing", () => {
    expect(chartScore({ rank: 1 }, 0)).toBe(100);
    expect(chartScore({ rank: 3 }, 0)).toBe(80);
    expect(chartScore({}, 1)).toBe(90);
    expect(chartScore({ importance: 0, rank: 1 }, 0)).toBe(0);
  });

  it("breaks score ties by rows, then rank, then dataset order", () => {
    const byRows = pickTopInsights([ds("small", 5, [{ title: "s", importance: 50 }]), ds("big", 500, [{ title: "b", importance: 50 }])]);
    expect(byRows.map((p) => p.chart.title)).toEqual(["b", "s"]);
    const byRank = pickTopInsights([ds("a", 5, [{ title: "late", importance: 50, rank: 4 }]), ds("b", 5, [{ title: "early", importance: 50, rank: 2 }])]);
    expect(byRank.map((p) => p.chart.title)).toEqual(["early", "late"]);
    const byOrder = pickTopInsights([ds("first", 5, [{ title: "f", importance: 50, rank: 1 }]), ds("second", 5, [{ title: "s", importance: 50, rank: 1 }])]);
    expect(byOrder.map((p) => p.chart.title)).toEqual(["f", "s"]);
  });

  it("returns every chart when there are fewer than 5 in total", () => {
    const picks = pickTopInsights([ds("a", 1, [{ title: "a1" }]), ds("b", 1, [{ title: "b1" }]), ds("c", 1, [])]);
    expect(picks).toHaveLength(2);
  });
});
