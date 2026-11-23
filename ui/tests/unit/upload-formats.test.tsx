// The upload modal reads its accept list from /api/capabilities and renders
// the deferred formats greyed out with the backend's reason. Before this the
// list was hardcoded to four extensions against a server that took nine.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { UploadFormatHint } from "@/components/dashboard/UploadFormatHint";
import {
  FALLBACK_ACCEPT,
  describeAccepted,
  isAccepted,
  uploadFormatsFrom,
} from "@/lib/upload-formats";

const CAPABILITIES = {
  features: {
    upload_formats: [
      "avro",
      "csv",
      "json",
      "jsonl",
      "ndjson",
      "orc",
      "parquet",
      "tsv",
      "xls",
      "xlsx",
      "xml",
    ],
    upload_format_status: {
      avro: { status: "enabled" as const, statistics: true, reason: null },
      csv: { status: "enabled" as const, statistics: true, reason: null },
      jsonl: { status: "enabled" as const, statistics: true, reason: null },
      iceberg: {
        status: "deferred" as const,
        statistics: false,
        reason: "a table directory, not a file: register it through the existing-folder route",
      },
    },
  },
};

afterEach(cleanup);

describe("uploadFormatsFrom", () => {
  it("falls back to the shipped list when the backend says nothing", () => {
    expect(uploadFormatsFrom(null).accept).toEqual(FALLBACK_ACCEPT);
    expect(uploadFormatsFrom({ features: {} }).fromBackend).toBe(false);
  });

  it("accepts every enabled extension and lists each deferral with its reason", () => {
    const formats = uploadFormatsFrom(CAPABILITIES);
    expect(formats.fromBackend).toBe(true);
    expect(formats.accept).toEqual([
      ".avro",
      ".csv",
      ".json",
      ".jsonl",
      ".ndjson",
      ".orc",
      ".parquet",
      ".tsv",
      ".xls",
      ".xlsx",
      ".xml",
    ]);
    expect(formats.deferred).toEqual([
      {
        format: "iceberg",
        reason: "a table directory, not a file: register it through the existing-folder route",
      },
    ]);
    expect(isAccepted(formats, "events.PARQUET")).toBe(true);
    expect(isAccepted(formats, "table.iceberg")).toBe(false);
    expect(describeAccepted(formats)).toBe(
      "Avro, CSV, JSON, JSON Lines, ORC, Parquet, TSV, Excel, XML",
    );
  });
});

describe("UploadFormatHint", () => {
  it("renders the deferred format as deferred, with the reason on hover", () => {
    render(<UploadFormatHint formats={uploadFormatsFrom(CAPABILITIES)} limitsText="max. 100 MB" />);
    expect(screen.getByTestId("upload-accepted").textContent).toContain("Parquet");
    const deferred = screen.getByTestId("upload-deferred");
    expect(deferred.textContent).toBe("Not yet: iceberg");
    expect(deferred.querySelector("abbr[data-format='iceberg']")?.getAttribute("title")).toBe(
      "a table directory, not a file: register it through the existing-folder route",
    );
  });

  it("renders no deferred line on an older backend", () => {
    render(<UploadFormatHint formats={uploadFormatsFrom(null)} limitsText="max. 100 MB" />);
    expect(screen.getByTestId("upload-accepted").textContent).toBe(
      "CSV, Excel, JSON (max. 100 MB)",
    );
    expect(screen.queryByTestId("upload-deferred")).toBeNull();
  });
});
