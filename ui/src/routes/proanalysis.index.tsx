import { useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { createFileRoute, useLocation, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, ArrowRight } from "@untitledui/icons";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { UploadDropzone } from "@/components/dashboard/UploadDropzone";
import { DataSourceGrid } from "@/components/dashboard/DataSourceGrid";
import { SecurityFooter } from "@/components/dashboard/SecurityFooter";
import { Button } from "@/components/base/buttons/button";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { cx } from "@/lib/utils/cx";
import { getMyFirstName } from "@/lib/me.functions";

export const Route = createFileRoute("/proanalysis/")({
  ssr: false,
  head: () => ({
    meta: [
      { title: "Dataset Preview · Avaloka AI" },
      { name: "description", content: "Preview rows, schema and summary of your uploaded dataset before starting analysis." },
    ],
  }),
  pendingComponent: PreviewPagePending,
  component: PreviewPage,
});


type UploadResponse = {
  dataset_id: string;
  session_id: string;
  thread_id: string;
  schema: string[] | Record<string, string>;
  samples: any[];
  rows_sampled: number;
  analysis_fidelity?: string;
  selected_sample_name?: string;
  visualization_config?: any;
  portfolio_samples?: Record<string, any[]>;
  available_samples?: string[];
};

type LocState = { uploadResponse?: UploadResponse; filename?: string };

type TabKey = "rows" | "schema" | "summary";

function PreviewPage() {
  const fetchFirstName = useServerFn(getMyFirstName);
  const { data: me } = useQuery({
    queryKey: ["my-first-name"],
    queryFn: () => fetchFirstName(),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
  const firstName = me?.firstName ?? "";

  const navigate = useNavigate();
  const location = useLocation();
  const state = (location.state ?? {}) as LocState;
  const uploadResponse = state.uploadResponse;
  const filename = state.filename ?? "Dataset";
  const [tab, setTab] = useState<TabKey>("rows");

  if (!uploadResponse) {
    return (
      <DashboardShell>
        <div className="flex min-h-screen flex-1 flex-col">
          <div className="flex flex-1 flex-col items-center justify-center px-8 py-10">
            <div className="mx-auto w-full max-w-[640px]">
              <header className="text-center">
                <h1 className="text-display-md font-semibold tracking-tight text-primary">
                  {firstName ? `Welcome ${firstName}` : "Welcome"}
                </h1>
                <p className="mt-2 text-sm text-tertiary">
                  Get started by uploading your data to get instant Insights
                </p>
              </header>

              <section className="mt-10">
                <UploadDropzone redirectTo="preview" />
              </section>

              <div className="my-8 flex items-center gap-4">
                <div className="h-px flex-1 bg-border-secondary" />
                <span className="text-sm text-tertiary">Or</span>
                <div className="h-px flex-1 bg-border-secondary" />
              </div>

              <section>
                <DataSourceGrid />
              </section>
            </div>
          </div>

          <div className="mt-auto">
            <SecurityFooter />
          </div>
        </div>
      </DashboardShell>
    );
  }


  const { schema, samples = [], rows_sampled } = uploadResponse;
  const columns: string[] = Array.isArray(schema)
    ? schema
    : Object.keys(schema ?? {});

  return (
    <DashboardShell>
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between gap-4 border-b border-secondary px-8 py-4">
          <div className="flex min-w-0 items-center gap-3">
            <ButtonUtility
              size="sm"
              color="tertiary"
              icon={ArrowLeft}
              tooltip="Back to dashboard"
              onClick={() => navigate({ to: "/dashboard" })}
            />
            <h1 className="truncate text-xl font-semibold text-primary">{filename}</h1>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Pill>{columns.length} columns</Pill>
            <Pill>{rows_sampled ?? samples.length} rows</Pill>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 border-b border-secondary px-8 pt-3">
          {(["rows", "schema", "summary"] as TabKey[]).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setTab(k)}
              className={cx(
                "rounded-t-md px-3 py-2 text-sm font-medium transition-colors -mb-px border-b-2",
                tab === k
                  ? "border-brand text-primary"
                  : "border-transparent text-tertiary hover:text-primary",
              )}
            >
              {k === "rows" ? "Rows" : k === "schema" ? "Schema" : "Summary"}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-auto px-8 py-6">
          {tab === "rows" && <RowsTab schema={schema} samples={samples} rowsSampled={rows_sampled} />}
          {tab === "schema" && <SchemaTab schema={schema} samples={samples} />}
          {tab === "summary" && <SummaryTab response={uploadResponse} />}
        </div>

        {/* Footer */}
        <div className="flex justify-end border-t border-secondary px-8 py-4">
          <Button
            color="primary"
            size="md"
            iconTrailing={ArrowRight}
            onClick={() =>
              navigate({
                to: "/analysis" as never,
                state: { uploadResponse, filename } as never,
              })
            }
          >
            Start Analysis
          </Button>
        </div>
      </div>
    </DashboardShell>
  );
}

function PreviewPagePending() {
  return (
    <DashboardShell>
      <div className="flex min-h-screen flex-1 flex-col items-center justify-center px-8 py-10">
        <div className="mx-auto w-full max-w-[640px]">
          <header className="text-center">
            <div className="mx-auto h-10 w-56 animate-pulse rounded-lg bg-secondary" />
            <div className="mx-auto mt-3 h-4 w-72 animate-pulse rounded bg-secondary" />
          </header>
          <div className="mt-10 h-56 w-full animate-pulse rounded-xl bg-secondary" />
          <div className="my-8 flex items-center gap-4">
            <div className="h-px flex-1 bg-border-secondary" />
            <div className="h-4 w-8 animate-pulse rounded bg-secondary" />
            <div className="h-px flex-1 bg-border-secondary" />
          </div>
          <div className="h-40 w-full animate-pulse rounded-xl bg-secondary" />
        </div>
      </div>
    </DashboardShell>
  );
}

function Pill({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-1 text-xs font-medium text-secondary">
      {children}
    </span>
  );
}

/* ---------------- Rows ---------------- */
function RowsTab({
  schema,
  samples,
  rowsSampled,
}: {
  schema: UploadResponse["schema"];
  samples: any[];
  rowsSampled: number;
}) {
  const columns: string[] = Array.isArray(schema) ? schema : Object.keys(schema ?? {});
  const rows = samples.slice(0, 50);

  if (rows.length === 0) {
    return <p className="text-sm text-tertiary">No sample rows available.</p>;
  }

  const cellOf = (row: any, col: string) => {
    if (Array.isArray(row)) {
      const idx = columns.indexOf(col);
      return row[idx];
    }
    return row?.[col];
  };

  return (
    <div>
      <div className="overflow-auto rounded-lg border border-secondary">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-secondary">
            <tr>
              <th className="w-10 px-3 py-2 text-left text-xs font-medium text-tertiary">#</th>
              {columns.map((c) => (
                <th key={c} className="px-3 py-2 text-left text-xs font-medium text-secondary whitespace-nowrap">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-t border-secondary hover:bg-secondary/60">
                <td className="px-3 py-2 text-xs text-tertiary">{i + 1}</td>
                {columns.map((c) => {
                  const v = cellOf(row, c);
                  const s = v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
                  return (
                    <td key={c} className="px-3 py-2 text-primary">
                      <div className="max-w-[160px] truncate" title={s}>{s}</div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-tertiary">
        Showing {rows.length} of {rowsSampled ?? samples.length} sampled rows
      </p>
    </div>
  );
}

/* ---------------- Schema ---------------- */
function inferType(v: unknown): string {
  if (v == null) return "unknown";
  const t = typeof v;
  if (t === "number") return Number.isInteger(v) ? "integer" : "float";
  if (t === "boolean") return "boolean";
  if (t === "string") {
    if (/^\d{4}-\d{2}-\d{2}/.test(v as string)) return "date";
    return "string";
  }
  return t;
}

function typeTint(type: string) {
  const t = type.toLowerCase();
  if (/string|text|varchar|char/.test(t)) return "bg-utility-purple-50 text-utility-purple-700 ring-utility-purple-200";
  if (/int|float|number|double|decimal|numeric/.test(t)) return "bg-utility-success-50 text-utility-success-700 ring-utility-success-200";
  if (/bool/.test(t)) return "bg-utility-warning-50 text-utility-warning-700 ring-utility-warning-200";
  if (/date|time/.test(t)) return "bg-utility-blue-50 text-utility-blue-700 ring-utility-blue-200";
  return "bg-utility-gray-50 text-utility-gray-700 ring-utility-gray-200";
}

function SchemaTab({ schema, samples }: { schema: UploadResponse["schema"]; samples: any[] }) {
  const entries: Array<[string, string]> = Array.isArray(schema)
    ? schema.map((name) => {
        const first = samples[0];
        const v = first ? (Array.isArray(first) ? first[schema.indexOf(name)] : first?.[name]) : undefined;
        return [name, inferType(v)];
      })
    : Object.entries(schema ?? {});

  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" }}>
      {entries.map(([name, type]) => (
        <div key={name} className="rounded-lg border border-secondary bg-primary p-2.5">
          <p className="truncate text-[13px] font-medium text-primary" title={name}>{name}</p>
          <span
            className={cx(
              "mt-1.5 inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset",
              typeTint(String(type)),
            )}
          >
            {type}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ---------------- Summary ---------------- */
function SummaryTab({ response }: { response: UploadResponse }) {
  const { schema, samples = [], rows_sampled, analysis_fidelity } = response;
  const entries: Array<[string, string]> = Array.isArray(schema)
    ? schema.map((name) => {
        const first = samples[0];
        const v = first ? (Array.isArray(first) ? first[schema.indexOf(name)] : first?.[name]) : undefined;
        return [name, inferType(v)];
      })
    : Object.entries(schema ?? {});

  const numericCount = entries.filter(([, t]) => /int|float|number|double|decimal|numeric/i.test(t)).length;
  const textCount = entries.filter(([, t]) => /string|text|varchar|char/i.test(t)).length;

  let nonNull = 0;
  let total = 0;
  for (const row of samples) {
    for (const [name] of entries) {
      const v = Array.isArray(row) ? row[entries.findIndex(([n]) => n === name)] : row?.[name];
      total += 1;
      if (v !== null && v !== undefined && v !== "") nonNull += 1;
    }
  }
  const completeness = total > 0 ? Math.round((nonNull / total) * 100) : 0;
  const fidelity = (analysis_fidelity ?? "—").replace(/_/g, " ");

  const cards: Array<[string, string | number]> = [
    ["Columns", entries.length],
    ["Sampled rows", rows_sampled ?? samples.length],
    ["Numeric cols", numericCount],
    ["Text cols", textCount],
    ["Completeness", `${completeness}%`],
    ["Fidelity", fidelity],
  ];

  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))" }}>
      {cards.map(([label, value]) => (
        <div key={label} className="rounded-lg bg-secondary px-3.5 py-3">
          <p className="text-[11px] uppercase tracking-wide text-tertiary">{label}</p>
          <p className="mt-1 text-[22px] font-medium text-primary">{value}</p>
        </div>
      ))}
    </div>
  );
}
