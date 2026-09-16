import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Loader2, Database } from "lucide-react";

export const Route = createFileRoute("/_authenticated/db-tables")({
  component: DbTablesPage,
  errorComponent: ({ error }) => (
    <div className="p-6 text-sm text-red-600">{String(error)}</div>
  ),
  notFoundComponent: () => <div className="p-6">Not found</div>,
});

// Known public tables. Keep in sync with the database schema.
const TABLES = [
  "analyses",
  "analysis_dashboards",
  "app_users",
  "branches",
  "config_roles",
  "dashboard_graphs",
  "dashboard_tabs",
  "data_source_files",
  "data_sources",
  "departments",
  "models",
  "profiles",
  "projects",
  "teams",
  "user_roles",
] as const;

type TableName = (typeof TABLES)[number];

function DbTablesPage() {
  const [selected, setSelected] = useState<TableName>("profiles");
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    (async () => {
      const { data, error, count } = await supabase
        .from(selected)
        .select("*", { count: "exact" })
        .limit(50);
      if (cancelled) return;
      if (error) {
        setError(error.message);
        setRows([]);
        setCount(null);
      } else {
        setRows((data ?? []) as Record<string, unknown>[]);
        setCount(count ?? null);
      }
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

  return (
    <div className="flex h-full flex-1 overflow-hidden">
        {/* Sidebar with table list */}
        <aside className="w-64 shrink-0 overflow-y-auto border-r border-secondary bg-secondary p-3">
          <div className="mb-3 flex items-center gap-2 px-2 py-1">
            <Database className="size-4 text-fg-quaternary" />
            <h2 className="text-sm font-semibold text-primary">Tables</h2>
          </div>
          <ul className="flex flex-col gap-0.5">
            {TABLES.map((t) => (
              <li key={t}>
                <button
                  onClick={() => setSelected(t)}
                  className={`w-full rounded-md px-2 py-1.5 text-left text-sm transition-colors ${
                    selected === t
                      ? "bg-primary_hover font-medium text-primary"
                      : "text-secondary hover:bg-primary_hover"
                  }`}
                >
                  {t}
                </button>
              </li>
            ))}
          </ul>
        </aside>

        {/* Main content */}
        <main className="flex flex-1 flex-col overflow-hidden">
          <header className="flex items-center justify-between border-b border-secondary px-6 py-4">
            <div>
              <h1 className="text-lg font-semibold text-primary">
                public.{selected}
              </h1>
              <p className="text-xs text-tertiary">
                {loading
                  ? "Loading…"
                  : count !== null
                    ? `${count} row${count === 1 ? "" : "s"} · showing up to 50`
                    : "—"}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelected((s) => s)}
              disabled={loading}
            >
              {loading ? <Loader2 className="size-4 animate-spin" /> : "Refresh"}
            </Button>
          </header>

          <div className="flex-1 overflow-auto p-4">
            {error ? (
              <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
            ) : loading ? (
              <div className="flex items-center justify-center py-12 text-sm text-tertiary">
                <Loader2 className="mr-2 size-4 animate-spin" /> Loading rows…
              </div>
            ) : rows.length === 0 ? (
              <div className="py-12 text-center text-sm text-tertiary">
                No rows.
              </div>
            ) : (
              <div className="overflow-auto rounded-md border border-secondary">
                <table className="min-w-full text-xs">
                  <thead className="bg-secondary">
                    <tr>
                      {columns.map((c) => (
                        <th
                          key={c}
                          className="whitespace-nowrap border-b border-secondary px-3 py-2 text-left font-semibold text-secondary"
                        >
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, i) => (
                      <tr key={i} className="border-b border-secondary last:border-0">
                        {columns.map((c) => (
                          <td
                            key={c}
                            className="max-w-[280px] truncate px-3 py-2 align-top text-primary"
                            title={formatCell(row[c])}
                          >
                            {formatCell(row[c])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </main>
      </div>
  );
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
