// Pure helpers for the analysis Data View (data table, column summary).
// Display-only: they work on rows already loaded in the browser and never fetch.


export type ColumnProfile = { name: string; dtype?: string; role?: string };
export type ColumnKind = "number" | "date" | "boolean" | "identifier" | "text";
export type SortDir = "asc" | "desc";
export type ColumnFilter = {
  contains?: string;
  values?: string[];
  min?: number;
  max?: number;
  from?: string;
  to?: string;
};
type Row = Record<string, unknown>;

const IDENTIFIER_NAME = /(^id$|_id$|^code$|_code$|census_tract|(^|_)tract$|(^|_)zip(code)?$|postal|(^|_)fips|(^|_)msa(_|$)|_md$|geoid)/i;
const NUMERIC_TEXT = /^-?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;
const DATE_TEXT = /^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$|^\d{1,2}\/\d{1,2}\/\d{2,4}$/;

export function isEmpty(value: unknown): boolean {
  if (value == null) return true;
  if (typeof value === "number") return Number.isNaN(value);
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    return text === "" || text === "nan" || text === "null" || text === "none";
  }
  return false;
}

export function isNumericValue(value: unknown): boolean {
  if (typeof value === "number") return Number.isFinite(value);
  return typeof value === "string" && NUMERIC_TEXT.test(value.trim());
}

export const isYearName = (name: string) => /(^|_)year$/i.test(name);
export const isIdentifierName = (name: string) => IDENTIFIER_NAME.test(name);

/** Column kind from the backend column profile first, then the loaded values. */
export function inferColumnKind(name: string, values: unknown[], profile?: ColumnProfile): ColumnKind {
  if (profile?.role?.toLowerCase() === "identifier" || isIdentifierName(name)) return "identifier";
  const dtype = (profile?.dtype ?? "").toLowerCase();
  if (/bool/.test(dtype)) return "boolean";
  if (/date|time/.test(dtype)) return "date";
  if (/int|float|double|decimal|numeric|number/.test(dtype)) return "number";
  const present = values.filter((value) => !isEmpty(value));
  if (!present.length) return "text";
  if (present.every((value) => typeof value === "boolean" || /^(true|false)$/i.test(String(value)))) return "boolean";
  if (present.every(isNumericValue)) return "number";
  if (present.every((value) => typeof value === "string" && DATE_TEXT.test(value.trim()) && !Number.isNaN(Date.parse(value)))) return "date";
  return "text";
}

export function columnKinds(rows: Row[], columns: string[], profiles: ColumnProfile[] = []): Record<string, ColumnKind> {
  const byName = new Map(profiles.map((profile) => [profile.name, profile]));
  const sample = rows.slice(0, 1000);
  return Object.fromEntries(columns.map((column) => [column, inferColumnKind(column, sample.map((row) => row?.[column]), byName.get(column))]));
}

/** BG41: numbers right-aligned; text, dates, booleans, IDs and codes left-aligned. */
export const cellAlign = (kind: ColumnKind): "right" | "left" => (kind === "number" ? "right" : "left");

export function formatValue(value: unknown, kind: ColumnKind, name: string): string {
  if (isEmpty(value)) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  if (kind === "number" && !isYearName(name) && isNumericValue(value)) {
    return Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  return String(value);
}

function comparable(value: unknown, kind: ColumnKind): number | string {
  if (kind === "number") return Number(value);
  if (kind === "date") return Date.parse(String(value));
  if (kind === "boolean") return /^true$/i.test(String(value)) ? 1 : 0;
  return String(value).toLowerCase();
}

/** BG38: numeric/date/text sorting; empty values always last in either direction. */
export function sortRows(rows: Row[], column: string, dir: SortDir, kind: ColumnKind): Row[] {
  const sign = dir === "asc" ? 1 : -1;
  return rows
    .map((row, index) => ({ row, index }))
    .sort((a, b) => {
      const av = a.row?.[column];
      const bv = b.row?.[column];
      const ae = isEmpty(av) || (kind === "number" && !isNumericValue(av));
      const be = isEmpty(bv) || (kind === "number" && !isNumericValue(bv));
      if (ae || be) return ae === be ? a.index - b.index : ae ? 1 : -1;
      const x = comparable(av, kind);
      const y = comparable(bv, kind);
      if (typeof x === "string" || typeof y === "string") {
        const result = String(x).localeCompare(String(y), undefined, { numeric: kind === "identifier" });
        return result * sign || a.index - b.index;
      }
      return (x === y ? 0 : x < y ? -1 : 1) * sign || a.index - b.index;
    })
    .map(({ row }) => row);
}

export function isFilterActive(filter?: ColumnFilter): boolean {
  if (!filter) return false;
  return Boolean(filter.contains?.trim()) || Boolean(filter.values?.length) || filter.min != null || filter.max != null || Boolean(filter.from) || Boolean(filter.to);
}

export function applyFilters(rows: Row[], filters: Record<string, ColumnFilter>): Row[] {
  const active = Object.entries(filters).filter(([, filter]) => isFilterActive(filter));
  if (!active.length) return rows;
  return rows.filter((row) =>
    active.every(([column, filter]) => {
      const value = row?.[column];
      if (filter.values?.length && !filter.values.includes(isEmpty(value) ? "" : String(value))) return false;
      if (filter.contains?.trim() && !String(value ?? "").toLowerCase().includes(filter.contains.trim().toLowerCase())) return false;
      if (filter.min != null || filter.max != null) {
        if (!isNumericValue(value)) return false;
        const n = Number(value);
        if (filter.min != null && n < filter.min) return false;
        if (filter.max != null && n > filter.max) return false;
      }
      if (filter.from || filter.to) {
        const time = Date.parse(String(value ?? ""));
        if (Number.isNaN(time)) return false;
        if (filter.from && time < Date.parse(filter.from)) return false;
        if (filter.to && time > Date.parse(filter.to) + 86_399_999) return false;
      }
      return true;
    }),
  );
}

export function topValues(rows: Row[], column: string, limit = 20): { value: string; count: number }[] {
  const counts = new Map<string, number>();
  rows.forEach((row) => {
    const value = row?.[column];
    const key = isEmpty(value) ? "" : String(value);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  });
  return [...counts.entries()].map(([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value)).slice(0, limit);
}

export function filterChipText(column: string, filter: ColumnFilter): string {
  const parts: string[] = [];
  if (filter.values?.length) parts.push(filter.values.map((value) => value || "(empty)").join(", "));
  if (filter.contains?.trim()) parts.push(`contains "${filter.contains.trim()}"`);
  if (filter.min != null || filter.max != null) parts.push(`${filter.min ?? "min"} – ${filter.max ?? "max"}`);
  if (filter.from || filter.to) parts.push(`${filter.from || "start"} → ${filter.to || "end"}`);
  return `${column}: ${parts.join("; ")}`;
}

/** BG40: first id / *_id / key-or-time column; otherwise the first column. */
export function choosePinnedColumn(columns: string[], profiles: ColumnProfile[] = []): string | null {
  const byName = new Map(profiles.map((profile) => [profile.name, profile]));
  const match = columns.find((column) => {
    if (/^id$/i.test(column) || /_id$/i.test(column)) return true;
    const role = byName.get(column)?.role?.toLowerCase() ?? "";
    return /key|time|temporal/.test(role) || /(^|_)(year|date)$/i.test(column);
  });
  return match ?? columns[0] ?? null;
}

// ---------------------------------------------------------------- column summary

export type ColumnSummaryRow = {
  name: string;
  type: "Integer" | "Decimal" | "Text" | "Date" | "Boolean";
  nonNull: number;
  missing: number;
  missingPct: number;
  unique: number;
  samples: string[];
};

export function summarizeColumns(rows: Row[], columns: string[], kinds: Record<string, ColumnKind>): ColumnSummaryRow[] {
  return columns.map((name) => {
    const values = rows.map((row) => row?.[name]);
    const present = values.filter((value) => !isEmpty(value));
    const kind = kinds[name] ?? "text";
    const type: ColumnSummaryRow["type"] =
      kind === "number" ? (present.every((value) => Number.isInteger(Number(value))) ? "Integer" : "Decimal")
        : kind === "date" ? "Date" : kind === "boolean" ? "Boolean" : "Text";
    const distinct = [...new Set(present.map((value) => (typeof value === "object" ? JSON.stringify(value) : String(value))))];
    return {
      name,
      type,
      nonNull: present.length,
      missing: values.length - present.length,
      missingPct: values.length ? ((values.length - present.length) / values.length) * 100 : 0,
      unique: distinct.length,
      samples: distinct.slice(0, 3),
    };
  });
}
