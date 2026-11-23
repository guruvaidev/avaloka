// Which upload formats the backend takes, read from /api/capabilities.
//
// The modal used to hardcode four extensions while the server accepted
// nine, and nothing said why a table layout it knows (Iceberg) was missing.
// `features.upload_format_status` is additive: every accepted extension is
// "enabled", every known-but-not-taken source is "deferred" with its reason.
// Older builds lack the key, so the shipped list stays as the fallback.

export interface UploadFormatStatus {
  status: "enabled" | "deferred";
  statistics?: boolean;
  reason?: string | null;
}

export interface UploadCapabilities {
  features?: {
    upload_formats?: string[];
    upload_format_status?: Record<string, UploadFormatStatus>;
  };
}

export interface UploadFormats {
  /** Extensions with the leading dot, sorted, for `accept=` and validation. */
  accept: string[];
  /** Known but not accepted, with the backend's reason. */
  deferred: Array<{ format: string; reason: string }>;
  /** True when the list came from the backend rather than the fallback. */
  fromBackend: boolean;
}

/** What the shipped modal accepted before capabilities carried the list. */
export const FALLBACK_ACCEPT = [".csv", ".xlsx", ".xls", ".json"];

export const FALLBACK_FORMATS: UploadFormats = {
  accept: FALLBACK_ACCEPT,
  deferred: [],
  fromBackend: false,
};

export function uploadFormatsFrom(
  capabilities: UploadCapabilities | null | undefined,
): UploadFormats {
  const features = capabilities?.features;
  const status = features?.upload_format_status ?? {};
  const enabled = new Set<string>(
    Object.entries(status)
      .filter(([, entry]) => entry?.status === "enabled")
      .map(([ext]) => ext.toLowerCase()),
  );
  for (const ext of features?.upload_formats ?? []) enabled.add(ext.toLowerCase());
  if (!enabled.size) return FALLBACK_FORMATS;
  const deferred = Object.entries(status)
    .filter(([, entry]) => entry?.status === "deferred")
    .map(([format, entry]) => ({ format, reason: entry.reason ?? "not accepted by this build" }))
    .sort((a, b) => a.format.localeCompare(b.format));
  return {
    accept: [...enabled].sort().map((ext) => `.${ext}`),
    deferred,
    fromBackend: true,
  };
}

/** `.csv` -> true when the extension is accepted. */
export function isAccepted(formats: UploadFormats, fileName: string): boolean {
  const ext = "." + (fileName.split(".").pop() ?? "").toLowerCase();
  return formats.accept.includes(ext);
}

/** "CSV, JSON, Parquet, …" for the dropzone hint. */
export function describeAccepted(formats: UploadFormats): string {
  const labels: Record<string, string> = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".json": "JSON",
    ".jsonl": "JSON Lines",
    ".ndjson": "JSON Lines",
    ".xml": "XML",
    ".parquet": "Parquet",
    ".orc": "ORC",
    ".avro": "Avro",
  };
  const seen = new Set<string>();
  const names: string[] = [];
  for (const ext of formats.accept) {
    const label = labels[ext] ?? ext.slice(1).toUpperCase();
    if (seen.has(label)) continue;
    seen.add(label);
    names.push(label);
  }
  return names.join(", ");
}
