// UI-B03 / UI-B04 (UI Bugs): with several files uploaded together, charts
// must not vanish when one file is left unselected, and the Auto Insights
// tabs must keep each file's own charts (own config, own rows).
// Tests buildAutoInsightsDatasets, moved verbatim from routes/analysis.tsx.
import { describe, expect, it } from "vitest";
import { buildAutoInsightsDatasets, type DatasetChip, type UploadedDataset } from "@/lib/analysis-workspace";

const cfg = (title: string) => ({ charts: [{ type: "bar", title, x: "region", y: "sales" }] });
const rows = (tag: string) => [
  { region: `${tag}-north`, sales: 10 },
  { region: `${tag}-south`, sales: 20 },
];
const chip = (id: string, name: string, withRows = true): DatasetChip => ({
  id,
  name,
  samples: withRows ? rows(id) : undefined,
  visualizationConfig: cfg(`${name} chart`),
  visualizationStatus: "ready",
});

describe("UI-B03 charts survive multi-file selection changes", () => {
  const a = chip("ds-a", "sales.csv");
  const b = chip("ds-b", "returns.csv");
  const c = chip("ds-c", "stores.csv");

  it("UI-B03: leaving one file unselected keeps the charts of the selected files", () => {
    const tabs = buildAutoInsightsDatasets([a, b, c], null, ["ds-a", "ds-c"]);
    const byId = Object.fromEntries(tabs.map((t) => [t.id, t]));
    expect(byId["ds-a"]?.config).toBe(a.visualizationConfig);
    expect(byId["ds-c"]?.config).toBe(c.visualizationConfig);
    expect(byId["ds-a"]?.status).toBe("ready");
  });

  it("UI-B03: re-selecting the file brings its own charts back unchanged", () => {
    buildAutoInsightsDatasets([a, b, c], null, ["ds-a"]);
    const tabs = buildAutoInsightsDatasets([a, b, c], null, ["ds-a", "ds-b", "ds-c"]);
    expect(tabs.map((t) => t.id)).toEqual(["ds-a", "ds-b", "ds-c"]);
    expect(tabs.find((t) => t.id === "ds-b")?.config).toBe(b.visualizationConfig);
  });
});

describe("UI-B04 Auto Insights keep each file's charts", () => {
  it("UI-B04: every Auto Insights tab pairs a file's own chart spec with its own rows", () => {
    const a = chip("ds-a", "sales.csv");
    const b = chip("ds-b", "returns.csv");
    const tabs = buildAutoInsightsDatasets([a, b], null, ["ds-a", "ds-b"]);
    expect(tabs).toHaveLength(2);
    for (const t of tabs) {
      const own = t.id === "ds-a" ? a : b;
      expect(t.config).toBe(own.visualizationConfig);
      expect(t.samples).toBe(own.samples);
    }
  });

  it("UI-B04: a file without inline rows never borrows another file's preview rows", () => {
    const a = chip("ds-a", "sales.csv");
    const b = chip("ds-b", "returns.csv", false);
    const preview: UploadedDataset = { filename: "sales.csv", schema: ["region", "sales"], samples: rows("preview"), rows_sampled: 2 };
    const tabs = buildAutoInsightsDatasets([a, b], preview, ["ds-a", "ds-b"]);
    const tabB = tabs.find((t) => t.id === "ds-b");
    expect(tabB?.config).toBe(b.visualizationConfig);
    expect(tabB?.samples).toBeUndefined();
  });
});
