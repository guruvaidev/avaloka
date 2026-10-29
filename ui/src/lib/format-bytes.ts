const UNITS = ["B", "KB", "MB", "GB", "TB"];

/** Human-readable binary size with one decimal, dropping a trailing ".0"; unknown sizes show "—". */
export function formatBytes(n: number | string | null | undefined): string {
  if (n === null || n === undefined || n === "") return "—";
  const bytes = typeof n === "string" ? Number(n) : n;
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  if (unit === 0) return `${Math.round(value)} B`;
  const text = value.toFixed(1).replace(/\.0$/, "");
  return `${text} ${UNITS[unit]}`;
}

/** Exact byte count for tooltips, e.g. "200,279 bytes"; empty for unknown sizes. */
export function exactBytesLabel(n: number | string | null | undefined): string {
  if (n === null || n === undefined || n === "") return "";
  const bytes = typeof n === "string" ? Number(n) : n;
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  return `${Math.round(bytes).toLocaleString("en-US")} bytes`;
}
