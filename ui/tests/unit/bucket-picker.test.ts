import { describe, expect, it } from "vitest";
import { folderHint, presentBucketListing, selectionLabel, useSelectionLabel } from "../../src/lib/bucket-picker";

describe("bucket picker selection wording", () => {
  it.each([
    [1, 0, "1 folder selected"],
    [2, 0, "2 folders selected"],
    [0, 1, "1 file selected"],
    [0, 2, "2 files selected"],
    [1, 2, "1 folder and 2 files selected"],
    [2, 1, "2 folders and 1 file selected"],
    [0, 0, "0 files selected"],
  ])("labels %i folders and %i files", (folders, files, expected) => {
    expect(selectionLabel(folders, files)).toBe(expected);
  });
  it.each([
    [1, 0, "Use folder"], [2, 0, "Use folders"],
    [0, 1, "Use file"], [0, 2, "Use files"], [1, 2, "Use folder"],
  ])("labels the action for %i folders and %i files", (folders, files, expected) => {
    expect(useSelectionLabel(folders, files)).toBe(expected);
  });
});

describe("metadata-based folder presentation", () => {
  it.each(["4936-3761.csv", "result.parquet", "result.JSON"])("hints extension folder %s", (name) => {
    expect(folderHint(name)).toBe("Folder of files (written by a job)");
  });
  it("does not hint an ordinary folder or a file extension embedded in its name", () => {
    expect(folderHint("exports")).toBeUndefined();
    expect(folderHint("result.csv.backup")).toBeUndefined();
  });
  it("keeps an extension-named prefix without file metadata a folder", () => {
    const folder = { name: "4936-3761.csv", prefix: "4936-3761.csv/" };
    expect(presentBucketListing([], [folder])).toEqual({ objects: [], folders: [folder] });
  });
  it("presents even zero-byte folder entries with size metadata as files", () => {
    expect(presentBucketListing([], [{ name: "result.csv", prefix: "result.csv/", size: 0 }])).toEqual({
      objects: [{ key: "result.csv", size: 0, updated: undefined, content_type: undefined }], folders: [],
    });
  });
  it("presents entries with content-type metadata as files and retains original keys", () => {
    const listing = presentBucketListing([], [{ name: "result.json", prefix: "exports/result.json", content_type: "application/json" }]);
    expect(listing.folders).toEqual([]);
    expect(listing.objects[0]).toMatchObject({ key: "exports/result.json", content_type: "application/json" });
  });
  it("does not overwrite an already-loaded file when folder metadata names the same key", () => {
    const object = { key: "result.csv", size: "1024" };
    expect(presentBucketListing([object], [{ name: "result.csv", prefix: "result.csv/", size: 0 }]).objects).toEqual([object]);
  });
});