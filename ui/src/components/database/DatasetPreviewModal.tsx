import React from "react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

export type DatasetPreviewData = {
  filename?: string;
  schema?: any;
  columns?: any;
  samples?: any;
  rows?: any;
  data?: any;
  rows_sampled?: number;
};

/** Local boundary so a bad preview payload can never take down the route. */
class PreviewErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(error: unknown) {
    console.warn("[DatasetPreviewModal] preview render failed", error);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="p-10 text-center text-sm text-muted-foreground">
          Preview couldn't be rendered for this dataset.
        </div>
      );
    }
    return this.props.children as React.ReactElement;
  }
}

function toColumnName(c: any): string {
  if (c == null) return "";
  if (typeof c === "string" || typeof c === "number") return String(c);
  if (typeof c === "object") return String(c.name ?? c.column ?? c.field ?? c.key ?? "");
  return String(c);
}

function normalizeColumns(data?: DatasetPreviewData | null, rows?: any[]): string[] {
  const raw = data?.schema ?? data?.columns;
  let cols: string[] = [];
  if (Array.isArray(raw)) cols = raw.map(toColumnName).filter(Boolean);
  else if (raw && typeof raw === "object") {
    const nested = (raw as any).columns ?? (raw as any).fields;
    if (Array.isArray(nested)) cols = nested.map(toColumnName).filter(Boolean);
    else cols = Object.keys(raw);
  }
  if (cols.length === 0 && rows?.length) {
    const first = rows.find((r) => r && typeof r === "object" && !Array.isArray(r));
    if (first) cols = Object.keys(first);
  }
  return Array.from(new Set(cols));
}

function normalizeRows(data?: DatasetPreviewData | null): any[] {
  const raw = data?.samples ?? data?.rows ?? data?.data;
  if (Array.isArray(raw)) return raw;
  return [];
}

function cellText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return "";
    }
  }
  return String(value);
}

export function DatasetPreviewModal({
  open,
  onOpenChange,
  loading,
  data,
  error,
  title,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  loading?: boolean;
  data?: DatasetPreviewData | null;
  error?: string | null;
  title?: string;
}) {
  let rows: any[] = [];
  let columns: string[] = [];
  try {
    rows = normalizeRows(data).slice(0, 50);
    columns = normalizeColumns(data, rows);
  } catch {
    rows = [];
    columns = [];
  }
  const cellOf = (row: any, col: string, i: number) =>
    Array.isArray(row) ? row[i] : row && typeof row === "object" ? row[col] : row;

  const empty = !loading && !error && (rows.length === 0 || columns.length === 0);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="grid h-[min(720px,calc(100vh-64px))] w-[min(1080px,calc(100vw-48px))] max-w-none grid-rows-[auto_1fr] gap-0 overflow-hidden rounded-2xl p-0">
        <div className="border-b border-border px-6 py-4">
          <DialogTitle className="truncate text-lg font-semibold">
            {title || data?.filename || "Dataset preview"}
          </DialogTitle>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {columns.length} columns · {data?.rows_sampled ?? rows.length} rows sampled
          </p>
        </div>
        <div className="min-h-0 overflow-auto p-4">
          <PreviewErrorBoundary>
            {loading ? (
              <div className="p-10 text-center text-sm text-muted-foreground">Loading preview…</div>
            ) : error ? (
              <div className="p-10 text-center text-sm text-muted-foreground">
                Couldn't load preview.
                <span className="mt-1 block text-xs opacity-70">{error}</span>
              </div>
            ) : empty ? (
              <div className="p-10 text-center text-sm text-muted-foreground">
                No sample rows available.
              </div>
            ) : (
              <table className="w-full border-collapse text-left text-xs">
                <thead className="sticky top-0 bg-card">
                  <tr>
                    {columns.map((c, i) => (
                      <th
                        key={`${c}-${i}`}
                        className="whitespace-nowrap border-b border-border px-3 py-2 font-semibold"
                      >
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, ri) => (
                    <tr key={ri} className="odd:bg-muted/40">
                      {columns.map((c, ci) => (
                        <td
                          key={`${c}-${ci}`}
                          className="whitespace-nowrap border-b border-border px-3 py-1.5"
                        >
                          {cellText(cellOf(row, c, ci))}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </PreviewErrorBoundary>
        </div>
      </DialogContent>
    </Dialog>
  );
}
