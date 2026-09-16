import { Fragment, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { AlertTriangle, Database, Loader2, Table as TableIcon, Play } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { backendApi } from "@/lib/api/backendApi";
import { BACKEND_API_BASE } from "@/lib/api/backend-config";
import { MCP_TOOL_BASE } from "@/lib/api/mcp-config";
import { mcpProxyFetch, proxyFetch } from "@/lib/api/backendApi";

import { openDatasetsInProAnalysis } from "@/lib/open-analysis-from-datasets";


export type McpConnection = {
  id: string;
  connection_name: string;
  customer_id: string;
  database_type: string;
  host: string;
  port: number;
  database_name: string;
  mcp_endpoint: string | null;
  status: string | null;
  query_timeout: number | null;
  max_rows: number | null;
};

type QueryResult = {
  columns: string[];
  rows: any[][];
  error?: string;
};

export type TableColumnInfo = { name: string; type?: string };
export type TableInfo = {
  name: string;
  type?: string;
  column_count?: number;
  columns?: TableColumnInfo[];
};

/** Deep-search an MCP response for a list of tables and normalize it. */
function extractTables(payload: any): TableInfo[] | null {
  const seen = new Set<any>();
  const walk = (node: any, depth: number): TableInfo[] | null => {
    if (!node || depth > 6) return null;
    if (typeof node === "string") {
      try {
        return walk(JSON.parse(node), depth + 1);
      } catch {
        return null;
      }
    }
    if (typeof node !== "object") return null;
    if (seen.has(node)) return null;
    seen.add(node);

    const arr = Array.isArray(node) ? node : node.tables;
    if (Array.isArray(arr) && arr.length && arr.every((t) => t && (typeof t === "string" || typeof t === "object"))) {
      const mapped = arr
        .map((t: any): TableInfo | null => {
          if (typeof t === "string") return { name: t };
          const name = t.name ?? t.table_name ?? t.table;
          if (!name) return null;
          const columns = Array.isArray(t.columns)
            ? t.columns.map((c: any) =>
                typeof c === "string"
                  ? { name: c }
                  : { name: c?.name ?? c?.column_name ?? "", type: c?.type ?? c?.data_type },
              )
            : undefined;
          return { name, type: t.type, column_count: t.column_count ?? columns?.length, columns };
        })
        .filter(Boolean) as TableInfo[];
      if (mapped.length) return mapped;
    }

    for (const v of Object.values(node)) {
      const found = walk(v, depth + 1);
      if (found) return found;
    }
    return null;
  };
  return walk(payload, 0);
}


function toCsv(columns: string[], rows: any[][]): string {
  const esc = (v: any) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const head = columns.map(esc).join(",");
  const body = rows.map((r) => r.map(esc).join(",")).join("\n");
  return `${head}\n${body}`;
}

function extractQueryResult(payload: any): QueryResult | null {
  const candidates = [
    payload?.query_result,
    payload?.result,
    payload?.data,
    payload,
  ];
  for (const c of candidates) {
    if (!c) continue;
    if (Array.isArray(c?.columns) && Array.isArray(c?.rows)) {
      return { columns: c.columns, rows: c.rows, error: c.error };
    }
    if (Array.isArray(c?.rows) && c.rows.length && typeof c.rows[0] === "object" && !Array.isArray(c.rows[0])) {
      const cols = Object.keys(c.rows[0]);
      return { columns: cols, rows: c.rows.map((r: any) => cols.map((k) => r[k])), error: c.error };
    }
  }
  return null;
}

export class DecryptError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function decryptApiKey(connectionId: string): Promise<{ apiKey: string; username?: string }> {
  const { data: sess } = await supabase.auth.getSession();
  const token = sess.session?.access_token;
  if (!token) throw new DecryptError("Your session expired. Sign in again.", 401);
  const res = await proxyFetch(
    `/api/mcp-connections/${connectionId}/decrypt`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    },
  );
  if (!res.ok) {
    let msg = `Failed to decrypt credentials (${res.status})`;
    if (res.status === 401) msg = "Your session expired. Sign in again.";
    else if (res.status === 403) msg = "That connection belongs to another account.";
    else if (res.status === 409)
      msg = "Stored credentials can't be decrypted. Re-register this connection.";
    else if (res.status === 500)
      msg = "Server could not decrypt credentials (encryption key not configured).";
    throw new DecryptError(msg, res.status);
  }
  const data: any = await res.json().catch(() => ({}));
  const apiKey = data?.apiKey || data?.api_key;
  if (!apiKey) throw new DecryptError("No API key returned", 500);
  return { apiKey, username: data?.username };
}

export function DatabaseQueryModal({
  open,
  connection,
  onOpenChange,
  onNavigatedAway,
  onReregister,
}: {
  open: boolean;
  connection: McpConnection;
  onOpenChange: (o: boolean) => void;
  onNavigatedAway?: () => void;
  onReregister?: () => void;
}) {
  const navigate = useNavigate();
  const [sql, setSql] = useState("SELECT * FROM your_table LIMIT 100");
  const [tableForDescribe, setTableForDescribe] = useState("");
  const [busy, setBusy] = useState<"" | "query" | "list" | "describe" | "load" | "tables">("");
  const [error, setError] = useState("");
  const [needsReregister, setNeedsReregister] = useState(false);
  const [preview, setPreview] = useState<QueryResult | null>(null);
  const [rawText, setRawText] = useState<string>("");
  const [tables, setTables] = useState<TableInfo[] | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [selectedTables, setSelectedTables] = useState<string[]>([]);

  const toggleTable = (name: string) =>
    setSelectedTables((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    );

  /** session_id established for this DB connection (from a prior connect/query). */
  const getConnectSessionId = async (): Promise<string | null> => {
    try {
      const { data } = await (supabase.from as any)("sql_query_sessions")
        .select("session_id")
        .eq("connection_id", connection.id)
        .maybeSingle();
      return data?.session_id ?? null;
    } catch {
      return null;
    }
  };

  const analyzeSelectedTables = async () => {
    if (!selectedTables.length) return;
    setBusy("tables");
    setError("");
    try {
      const { apiKey } = await decryptApiKey(connection.id);
      const { data: sessionData } = await supabase.auth.getSession();
      const accessToken = sessionData.session?.access_token;
      if (!accessToken) throw new Error("Not signed in");
      const dbSessionId = await getConnectSessionId();

      const res = await proxyFetch(`/api/database/tables-to-analysis`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          tables: selectedTables,
          customer_id: connection.customer_id,
          api_key: apiKey,
          session_id: dbSessionId,
          metadata: { customer_id: connection.customer_id, api_key: apiKey },
        }),
      });
      const body: any = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(body?.detail || body?.error || `Failed to load tables (${res.status})`);
        return;
      }
      const datasets = Array.isArray(body?.datasets) ? body.datasets : [];
      if (!datasets.length) {
        setError("No datasets were returned for the selected tables");
        return;
      }
      onNavigatedAway?.();
      onOpenChange(false);
      await openDatasetsInProAnalysis({
        navigate: navigate as any,
        sessionId: body?.session_id ?? dbSessionId ?? null,
        threadId: body?.thread_id ?? null,
        datasets,
      });
    } catch (e: any) {
      setError(e?.message || "Failed to analyze selected tables");
    } finally {
      setBusy("");
    }
  };


  const callMcpTool = async (
    name: "list_tables" | "describe_table",
    args: Record<string, any>,
  ) => {
    setBusy(name === "list_tables" ? "list" : "describe");
    setError("");
    setNeedsReregister(false);
    setRawText("");
    setTables(null);
    setExpanded({});
    setSelectedTables([]);

    try {
      const { apiKey } = await decryptApiKey(connection.id);
      
      const res = await mcpProxyFetch(MCP_TOOL_BASE, "/call_tool", {
          method: "POST",
          headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
          body: JSON.stringify({ method: "call_tool", params: { name, arguments: args } }),
      });
      const body = await res.json().catch(() => ({} as any));
      if (!res.ok) {
        setError(body?.detail || body?.error || `${name} failed (${res.status})`);
        return;
      }
      setRawText(typeof body === "string" ? body : JSON.stringify(body, null, 2));
      setTables(extractTables(body));

    } catch (e: any) {
      if (e instanceof DecryptError && e.status === 409) setNeedsReregister(true);
      setError(e?.message || `${name} failed`);
    } finally {
      setBusy("");
    }
  };

  const runQuery = async () => {
    setBusy("query");
    setError("");
    setNeedsReregister(false);
    setPreview(null);
    try {
      const { apiKey } = await decryptApiKey(connection.id);
      const { data: sessionData } = await supabase.auth.getSession();
      const accessToken = sessionData.session?.access_token;
      if (!accessToken) throw new Error("Not signed in");

      const res = await proxyFetch(`/api/v1/database/query`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          role: "user",
          content: sql,
          customer_id: connection.customer_id,
          metadata: { customer_id: connection.customer_id, api_key: apiKey },
        }),
      });
      const body = await res.json().catch(() => ({} as any));
      if (!res.ok) {
        setError(body?.detail || body?.error || `Query failed (${res.status})`);
        return;
      }
      const qr = extractQueryResult(body);
      if (!qr) {
        setError("Query returned no tabular result");
        return;
      }
      if (qr.error) {
        setError(qr.error);
        return;
      }
      setPreview(qr);
    } catch (e: any) {
      if (e instanceof DecryptError && e.status === 409) setNeedsReregister(true);
      setError(e?.message || "Query failed");
    } finally {
      setBusy("");
    }
  };

  const loadIntoAnalysis = async () => {
    if (!preview) return;
    setBusy("load");
    setError("");
    try {
      const csv = toCsv(preview.columns, preview.rows);
      const filename = `${connection.customer_id}-query-${Date.now()}.csv`;
      const file = new File([csv], filename, { type: "text/csv" });
      const upload = await backendApi.uploadFile(file);

      const { data: userData } = await supabase.auth.getUser();
      const uid = userData.user?.id;

      if (uid && upload.dataset_id) {
        try {
          await (supabase.from as any)("uploaded_files").insert({
            user_id: uid,
            file_name: filename,
            file_path: `sql-queries/${upload.dataset_id}.csv`,
            dataset_id: upload.dataset_id,
            session_id: upload.session_id,
            thread_id: upload.thread_id,
            file_type: "text/csv",
            file_size: csv.length,
          });
        } catch {}
        try {
          await (supabase.from as any)("sql_query_sessions").upsert(
            {
              connection_id: connection.id,
              session_id: upload.session_id,
              thread_id: upload.thread_id,
              dataset_id: upload.dataset_id,
              active: true,
            },
            { onConflict: "connection_id" },
          );
        } catch {}
      }

      // Route into Pro Analysis with the same helper the multi-file upload uses,
      // carrying the session_id / thread_id returned by the query upload.
      onNavigatedAway?.();
      onOpenChange(false);
      await openDatasetsInProAnalysis({
        navigate: navigate as any,
        sessionId: upload.session_id ?? null,
        threadId: upload.thread_id ?? null,
        datasets: [
          {
            dataset_id: upload.dataset_id,
            filename,
            alias: filename,
            columns: preview.columns,
            rows: preview.rows,
            schema: (upload as any).schema ?? null,
            samples: (upload as any).samples ?? null,
            visualization_config:
              (upload as any).visualization_config ?? (upload as any).visualization_configs ?? null,
            visualization_status: (upload as any).visualization_status,
          },
        ],
      });

    } catch (e: any) {
      setError(e?.message || "Failed to load into analysis");
    } finally {
      setBusy("");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <div className="flex items-center justify-between border-b px-1 pb-3 pr-8">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-[#1565EF]" />
            <div>
              <h2 className="text-base font-semibold">{connection.connection_name}</h2>
              <p className="text-xs text-muted-foreground">
                {connection.host}:{connection.port} · {connection.database_name}
              </p>
            </div>
          </div>
        </div>

        <div className="max-h-[65vh] space-y-3 overflow-y-auto p-1 pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="gap-2"
              disabled={!!busy}
              onClick={() => callMcpTool("list_tables", {})}
            >
              {busy === "list" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <TableIcon className="h-3.5 w-3.5" />}
              List tables
            </Button>
            <Input
              value={tableForDescribe}
              onChange={(e) => setTableForDescribe(e.target.value)}
              placeholder="table name"
              className="h-8 w-48"
            />
            <Button
              variant="outline"
              size="sm"
              disabled={!!busy || !tableForDescribe.trim()}
              onClick={() => callMcpTool("describe_table", { table: tableForDescribe.trim() })}
            >
              {busy === "describe" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Describe"}
            </Button>
          </div>

          {tables && tables.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-muted-foreground">
                  {selectedTables.length
                    ? `${selectedTables.length} table${selectedTables.length === 1 ? "" : "s"} selected`
                    : "Select tables to analyze together"}
                </p>
                <Button
                  size="sm"
                  onClick={analyzeSelectedTables}
                  disabled={!!busy || selectedTables.length === 0}
                  className="gap-2 bg-[#1565EF] hover:bg-[#1257cf]"
                >
                  {busy === "tables" && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  Analyze selected tables ({selectedTables.length})
                </Button>
              </div>
              <div className="rounded border">
              <table className="w-full text-xs">
                <thead className="bg-muted">
                  <tr>
                    <th className="w-8 px-2 py-1 text-left font-medium">
                      <input
                        type="checkbox"
                        aria-label="Select all tables"
                        checked={selectedTables.length === tables.length}
                        onChange={(e) =>
                          setSelectedTables(e.target.checked ? tables.map((t) => t.name) : [])
                        }
                      />
                    </th>
                    <th className="px-2 py-1 text-left font-medium">Table</th>
                    <th className="px-2 py-1 text-left font-medium">Type</th>
                    <th className="px-2 py-1 text-left font-medium">Columns</th>
                    <th className="px-2 py-1" />
                  </tr>
                </thead>
                <tbody>
                  {tables.map((t) => (
                    <Fragment key={t.name}>
                      <tr key={t.name} className="border-t">
                        <td className="px-2 py-1">
                          <input
                            type="checkbox"
                            aria-label={`Select ${t.name}`}
                            checked={selectedTables.includes(t.name)}
                            onChange={() => toggleTable(t.name)}
                          />
                        </td>
                        <td className="px-2 py-1">

                          <button
                            type="button"
                            title={`Use ${t.name} in the SQL box`}
                            onClick={() => {
                              setSql(`SELECT * FROM ${t.name} LIMIT 100`);
                              setTableForDescribe(t.name);
                            }}
                            className="font-medium text-[#1565EF] underline-offset-2 hover:underline"
                          >
                            {t.name}
                          </button>
                        </td>
                        <td className="px-2 py-1 text-muted-foreground">{t.type ?? "table"}</td>
                        <td className="px-2 py-1 text-muted-foreground">
                          {t.columns?.length ?? t.column_count ?? "—"}
                        </td>
                        <td className="px-2 py-1 text-right">
                          {!!t.columns?.length && (
                            <button
                              type="button"
                              onClick={() =>
                                setExpanded((e) => ({ ...e, [t.name]: !e[t.name] }))
                              }
                              className="text-[11px] text-muted-foreground hover:underline"
                            >
                              {expanded[t.name] ? "Hide columns" : "Show columns"}
                            </button>
                          )}
                        </td>
                      </tr>
                      {expanded[t.name] && !!t.columns?.length && (
                        <tr key={`${t.name}-cols`} className="border-t bg-muted/30">
                          <td colSpan={5} className="px-4 py-2">
                            <table className="w-full text-[11px]">
                              <tbody>
                                {t.columns.map((col) => (
                                  <tr key={col.name}>
                                    <td className="py-0.5 pr-4 font-medium">{col.name}</td>
                                    <td className="py-0.5 text-muted-foreground">{col.type ?? ""}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
              </div>
            </div>
          )}


          {rawText && (
            <details className="rounded border bg-muted/30 p-2 text-[11px]">
              <summary className="cursor-pointer select-none text-muted-foreground">
                View raw JSON
              </summary>
              <pre className="mt-2 max-h-40 overflow-auto">{rawText}</pre>
            </details>
          )}


          <div>
            <label className="mb-1 block text-xs font-medium">SQL</label>
            <Textarea
              value={sql}
              onChange={(e) => setSql(e.target.value)}
              rows={4}
              className="font-mono text-xs"
            />
          </div>

          <div className="flex justify-end">
            <Button onClick={runQuery} disabled={!!busy || !sql.trim()} className="gap-2 bg-[#1565EF] hover:bg-[#1257cf]">
              {busy === "query" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Run query
            </Button>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <div className="flex-1">
                <div>{error}</div>
                {needsReregister && onReregister && (
                  <button
                    type="button"
                    onClick={() => {
                      onOpenChange(false);
                      onReregister();
                    }}
                    className="mt-1 font-medium underline underline-offset-2"
                  >
                    Re-register this connection
                  </button>
                )}
              </div>
            </div>
          )}

          {preview && (
            <div>
              <div className="mb-1 flex items-center justify-between">
                <p className="text-xs text-muted-foreground">
                  {preview.rows.length} row{preview.rows.length === 1 ? "" : "s"} · {preview.columns.length} column
                  {preview.columns.length === 1 ? "" : "s"}
                </p>
                <Button
                  size="sm"
                  onClick={loadIntoAnalysis}
                  disabled={busy === "load"}
                  className="gap-2 bg-[#1565EF] hover:bg-[#1257cf]"
                >
                  {busy === "load" && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  Open in Analysis
                </Button>
              </div>
              <div className="max-h-56 overflow-auto rounded border">
                <table className="w-full text-xs">
                  <thead className="bg-muted">
                    <tr>
                      {preview.columns.map((c) => (
                        <th key={c} className="border-b px-2 py-1 text-left font-medium">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.slice(0, 50).map((r, i) => (
                      <tr key={i} className="odd:bg-background even:bg-muted/20">
                        {r.map((v, j) => (
                          <td key={j} className="border-b px-2 py-1">
                            {v === null || v === undefined ? "" : String(v)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
