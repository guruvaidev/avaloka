// UI-05 (Master Test Plan): files uploaded together are grouped into one card,
// titled by filename (never by dataset UUID). The backend side, one
// group_session_id per multi-file upload, is pinned by
// tests/test_server_unit.py::test_multi_upload_builds_group_index.
import { describe, expect, it } from "vitest";
import { groupUploadedDatasets, type UploadedDataset } from "@/lib/uploaded-datasets";

function ds(over: Partial<UploadedDataset> & { dataset_id: string }): UploadedDataset {
  return {
    name: over.dataset_id,
    filename: null,
    alias: null,
    columnCount: 3,
    columns: [],
    sizeBytes: 100,
    createdAt: "2026-09-01T10:00:00Z",
    sessionId: null,
    groupSessionId: null,
    sourceKind: "upload",
    ...over,
  };
}

describe("UI-05 multi-upload grouping", () => {
  const uuidA = "0b1e2c3d-0000-4000-8000-00000000000a";
  const datasets = [
    // First file of the batch: the API name is its UUID, so the card must fall back to the filename.
    ds({ dataset_id: uuidA, filename: "sales_2024.csv", groupSessionId: "grp-1", createdAt: "2026-09-01T10:00:00Z" }),
    ds({ dataset_id: "ds-b", name: "returns.csv", filename: "returns.csv", groupSessionId: "grp-1", createdAt: "2026-09-01T10:00:01Z" }),
    ds({ dataset_id: "ds-c", name: "stores.csv", filename: "stores.csv", groupSessionId: "grp-1", createdAt: "2026-09-01T10:00:02Z" }),
    // Uploaded on its own: its own card.
    ds({ dataset_id: "ds-d", name: "budget.xlsx", filename: "budget.xlsx", createdAt: "2026-09-02T09:00:00Z" }),
  ];

  it("UI-05: files sharing a group_session_id collapse into one card titled '<first file> + N more'", () => {
    const groups = groupUploadedDatasets(datasets);
    expect(groups).toHaveLength(2);
    const batch = groups.find((g) => g.key === "grp-1");
    expect(batch).toBeDefined();
    expect(batch!.members.map((m) => m.dataset_id)).toEqual([uuidA, "ds-b", "ds-c"]);
    expect(batch!.title).toBe("sales_2024.csv + 2 more");
    expect(batch!.totalSizeBytes).toBe(300);
  });

  it("UI-05: a card is never titled by the dataset UUID", () => {
    const groups = groupUploadedDatasets(datasets);
    for (const g of groups) expect(g.title).not.toContain(uuidA);
    const single = groups.find((g) => g.key === "ds-d");
    expect(single!.title).toBe("budget.xlsx");
  });
});
