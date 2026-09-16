// Reusable Output View (data table + Download CSV) for analysis results.
// Same visual language as the normal analysis page — used both there and on
// the Scheduled Analysis page so scheduled task outputs look identical.

import { useMemo, useState } from "react";
import { cx } from "@/lib/utils/cx";

type Row = Record<string, unknown>;

function formatCell(v: unknown): { text: string; isNumber: boolean } {
  if (v == null) return { text: "—", isNumber: false };
  if (typeof v === "number") {
    return { text: Number.isInteger(v) ? String(v) : v.toFixed(2), isNumber: true };
  }
  if (typeof v === "boolean") return { text: v ? "true" : "false", isNumber: false };
  if (typeof v === "object") return { text: JSON.stringify(v), isNumber: false };
  const s = String(v);
  const n = Number(s);
  if (!Number.isNaN(n) && s.trim() !== "") {
    return { text: s, isNumber: true };
  }
  return { text: s, isNumber: false };
}

function rowsToCsv(rows: Row[], columns: string[]): string {
  const escape = (v: unknown) => {
    if (v == null) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };
  const head = columns.join(",");
  const body = rows.map((r) => columns.map((c) => escape(r[c])).join(",")).join("\n");
  return `${head}\n${body}`;
}

export function AnalysisOutputTable({
  rows,
  title = "Output",
  filename = "output.csv",
  pageSize = 8,
  emptyMessage = "No output rows yet.",
}: {
  rows: Row[] | null | undefined;
  title?: string;
  filename?: string;
  pageSize?: number;
  emptyMessage?: string;
}) {
  const [page, setPage] = useState(1);
  const columns = useMemo(() => {
    const keys = new Set<string>();
    (rows ?? []).slice(0, 100).forEach((r) => Object.keys(r ?? {}).forEach((k) => keys.add(k)));
    return Array.from(keys);
  }, [rows]);

  const total = rows?.length ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * pageSize;
  const pageRows = (rows ?? []).slice(start, start + pageSize);

  const downloadCsv = () => {
    if (!rows || !rows.length) return;
    const csv = rowsToCsv(rows, columns);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-primary">
          {title} <span className="ml-1 text-xs font-normal text-tertiary">({total} rows)</span>
        </h3>
        <button
          type="button"
          onClick={downloadCsv}
          disabled={!rows || rows.length === 0}
          className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          Download CSV
        </button>
      </div>

      <div className="overflow-hidden rounded-xl border border-secondary bg-primary">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-secondary text-tertiary">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium">#</th>
                {columns.map((c) => (
                  <th key={c} className="px-4 py-2.5 text-left font-medium whitespace-nowrap">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageRows.length === 0 ? (
                <tr>
                  <td
                    colSpan={Math.max(1, columns.length) + 1}
                    className="px-4 py-10 text-center text-sm text-tertiary"
                  >
                    {emptyMessage}
                  </td>
                </tr>
              ) : (
                pageRows.map((row, i) => (
                  <tr key={start + i} className="border-t border-secondary">
                    <td className="px-4 py-2.5 text-tertiary">{start + i + 1}</td>
                    {columns.map((c) => {
                      const { text, isNumber } = formatCell(row[c]);
                      return (
                        <td
                          key={c}
                          className={cx(
                            "px-4 py-2.5 whitespace-nowrap",
                            isNumber
                              ? "text-right font-medium text-primary"
                              : "text-secondary",
                          )}
                        >
                          {text}
                        </td>
                      );
                    })}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-secondary px-4 py-3">
            <p className="text-xs text-tertiary">
              Showing {start + 1}-{Math.min(start + pageSize, total)} of {total.toLocaleString()}
            </p>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={safePage === 1}
                className="rounded-md border border-secondary bg-primary px-3 py-1.5 text-xs font-semibold text-primary disabled:cursor-not-allowed disabled:opacity-50"
              >
                Previous
              </button>
              <span className="px-2 text-xs text-tertiary">
                {safePage} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={safePage === totalPages}
                className="rounded-md border border-secondary bg-primary px-3 py-1.5 text-xs font-semibold text-primary disabled:cursor-not-allowed disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
