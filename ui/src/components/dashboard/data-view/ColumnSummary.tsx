// BG43: "Column Summary" view — one row per column describing structure.

import { useMemo } from "react";
import { columnKinds, summarizeColumns, type ColumnProfile } from "@/lib/data-view";

type Row = Record<string, unknown>;

export function ColumnSummary({ title, rows, columns, profiles = [], totalRows, sizeLabel, onColumn }: {
  title: string;
  rows: Row[];
  columns: string[];
  profiles?: ColumnProfile[];
  totalRows?: number;
  sizeLabel?: string | null;
  onColumn: (column: string) => void;
}) {
  const summary = useMemo(() => summarizeColumns(rows, columns, columnKinds(rows, columns, profiles)), [rows, columns, profiles]);
  const rowCount = totalRows ?? rows.length;
  return (
    <div className="flex flex-col">
      <div className="mb-3 min-w-0">
        <h3 className="truncate text-sm font-semibold text-primary">{title}</h3>
        <p className="mt-0.5 text-xs text-tertiary">
          {rowCount.toLocaleString()} rows · {columns.length} columns{sizeLabel ? ` · ${sizeLabel}` : ""}
          {rows.length && rows.length < rowCount ? ` · statistics from ${rows.length.toLocaleString()} loaded rows` : ""}
        </p>
      </div>
      {columns.length === 0 ? (
        <p className="text-sm text-tertiary">No dataset structure is available. Try re-uploading the file.</p>
      ) : (
        <div className="max-h-[calc(100dvh-280px)] overflow-auto rounded-lg border border-secondary">
          <table className="w-full border-separate border-spacing-0 text-[13px]">
            <thead>
              <tr className="text-tertiary">
                {["Column", "Type", "Non-null", "Missing", "Unique", "Sample values"].map((label, index) => (
                  <th key={label} className={`sticky top-0 border-b border-secondary bg-secondary px-3 py-2 font-medium ${index >= 2 && index <= 4 ? "text-right" : "text-left"}`}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summary.map((column) => (
                <tr key={column.name} className="hover:[&>td]:bg-secondary">
                  <td className="border-b border-secondary px-3 py-1.5">
                    <button type="button" onClick={() => onColumn(column.name)} title="Show this column in the Data table" className="font-semibold text-[#1a5fe0] hover:underline">{column.name}</button>
                  </td>
                  <td className="border-b border-secondary px-3 py-1.5 text-secondary">{column.type}</td>
                  <td className="border-b border-secondary px-3 py-1.5 text-right tabular-nums text-primary">{column.nonNull.toLocaleString()}</td>
                  <td className="border-b border-secondary px-3 py-1.5 text-right tabular-nums text-primary">
                    {column.missing.toLocaleString()} <span className="text-tertiary">({column.missingPct.toFixed(1).replace(/\.0$/, "")}%)</span>
                  </td>
                  <td className="border-b border-secondary px-3 py-1.5 text-right tabular-nums text-primary">{column.unique.toLocaleString()}</td>
                  <td className="max-w-[320px] truncate border-b border-secondary px-3 py-1.5 text-secondary" title={column.samples.join(", ")}>{column.samples.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
