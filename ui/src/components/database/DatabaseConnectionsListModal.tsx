import { useEffect, useMemo, useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Database, Loader2, Plus, Play } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { backendApi } from "@/lib/api/backendApi";
import { DatabaseConnectModal, DB_LABEL, type DbType } from "./DatabaseConnectModal";
import { DatabaseQueryModal, type McpConnection } from "./DatabaseQueryModal";
import { cn } from "@/lib/utils";

export function DatabaseConnectionsListModal({
  open,
  dbType,
  onOpenChange,
}: {
  open: boolean;
  dbType: DbType;
  onOpenChange: (o: boolean) => void;
}) {
  const [rows, setRows] = useState<McpConnection[]>([]);
  const [loading, setLoading] = useState(false);
  const [connectOpen, setConnectOpen] = useState(false);
  const [queryFor, setQueryFor] = useState<McpConnection | null>(null);
  const [health, setHealth] = useState<{ status: string; healthy: boolean } | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const { data: userData } = await supabase.auth.getUser();
      const uid = userData.user?.id;
      if (!uid) {
        setRows([]);
        return;
      }
      const { data } = await (supabase.from as any)("mcp_connections")
        .select("*")
        .eq("user_id", uid)
        .order("created_at", { ascending: false });
      setRows((data as McpConnection[]) ?? []);
      try {
        const h = await backendApi.checkMcpHealth();
        setHealth(h);
      } catch {
        setHealth({ status: "unknown", healthy: true });
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) void load();
  }, [open]);

  // realtime refresh
  useEffect(() => {
    if (!open) return;
    let channel: any;
    (async () => {
      const { data: userData } = await supabase.auth.getUser();
      const uid = userData.user?.id;
      if (!uid) return;
      channel = supabase
        .channel(`mcp-conn-${uid}`)
        .on(
          "postgres_changes" as any,
          { event: "*", schema: "public", table: "mcp_connections", filter: `user_id=eq.${uid}` },
          () => load(),
        )
        .subscribe();
    })();
    return () => {
      if (channel) supabase.removeChannel(channel);
    };
  }, [open]);

  const filtered = useMemo(
    () => rows.filter((r) => r.database_type === dbType),
    [rows, dbType],
  );

  return (
    <>
      <Dialog open={open && !queryFor} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-2xl">
          <div className="flex items-center justify-between border-b px-1 pb-3 pr-8">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-[#1565EF]" />
              <div>
                <h2 className="text-base font-semibold">{DB_LABEL[dbType]} connections</h2>
                <p className="text-xs text-muted-foreground">
                  Select a connection to query, or add a new one.{" "}
                  {health && (
                    <span
                      title={
                        health.healthy
                          ? "The database query service is reachable and responding."
                          : "The database query service isn't responding right now. Queries may fail — try again shortly, or re-register the connection if it persists."
                      }
                      className={cn(
                        "ml-1 inline-flex cursor-help items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium",
                        health.healthy
                          ? "bg-emerald-50 text-emerald-700"
                          : "bg-amber-50 text-amber-700",
                      )}
                    >
                      <span
                        className={cn(
                          "h-1.5 w-1.5 rounded-full",
                          health.healthy ? "bg-emerald-500" : "bg-amber-500",
                        )}
                      />
                      Service {health.healthy ? "online" : health.status}
                    </span>
                  )}
                </p>
              </div>
            </div>
          </div>


          <div className="max-h-[55vh] overflow-y-auto p-1 pt-3">
            {loading ? (
              <div className="flex items-center justify-center py-10 text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading…
              </div>
            ) : filtered.length === 0 ? (
              <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
                No {DB_LABEL[dbType]} connections yet. Add a new connection to get started.
              </div>
            ) : (
              <ul className="divide-y">
                {filtered.map((c) => (
                  <li
                    key={c.id}
                    className="flex items-center justify-between gap-3 py-3"
                  >
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium">{c.connection_name}</div>
                      <div className="truncate text-xs text-muted-foreground">
                        {c.host}:{c.port} · {c.database_name} · {c.customer_id}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      title="Open the query panel for this connection"
                      onClick={() => setQueryFor(c)}
                      className="shrink-0 gap-1.5 bg-[#1565EF] px-3 text-white hover:bg-[#1257cf]"
                    >
                      <Play className="h-3.5 w-3.5" />
                      Select
                    </Button>

                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="mt-3 flex justify-end gap-2 border-t pt-3">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Close
            </Button>
            <Button
              onClick={() => setConnectOpen(true)}
              className="gap-2 bg-[#1565EF] hover:bg-[#1257cf]"
            >
              <Plus className="h-4 w-4" /> New connection
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <DatabaseConnectModal
        open={connectOpen}
        dbType={dbType}
        onOpenChange={setConnectOpen}
        onConnected={() => load()}
      />

      {queryFor && (
        <DatabaseQueryModal
          open={!!queryFor}
          connection={queryFor}
          onOpenChange={(o: boolean) => {
            if (!o) setQueryFor(null);
          }}
          onNavigatedAway={() => {
            setQueryFor(null);
            onOpenChange(false);
          }}
          onReregister={() => {
            setQueryFor(null);
            setConnectOpen(true);
          }}
        />
      )}
    </>
  );
}
