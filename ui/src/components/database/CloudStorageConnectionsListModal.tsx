import { useCallback, useEffect, useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Cloud, Loader2, Plus, RefreshCw, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { supabase } from "@/integrations/supabase/client";
import {
  CloudStorageConnectModal,
  type StorageConnection,
  type StorageProvider,
} from "./CloudStorageConnectModal";
import { PROVIDER_LOGOS } from "./logos";

const PROVIDER_META: Record<
  StorageProvider,
  { label: string; logoKey: keyof typeof PROVIDER_LOGOS | string }
> = {
  aws: { label: "Amazon S3", logoKey: "s3" },
  azure: { label: "Azure Blob Storage", logoKey: "azure" },
  gcp: { label: "Google Cloud Storage", logoKey: "gcs" },
};

type SavedConnection = {
  id: string;
  name: string;
  provider: StorageProvider;
  region: string | null;
  bucket_name: string | null;
  connection_status: string | null;
  created_at: string;
};

export function CloudStorageConnectionsListModal({
  open,
  provider,
  onOpenChange,
  onBrowse,
}: {
  open: boolean;
  provider: StorageProvider;
  onOpenChange: (open: boolean) => void;
  onBrowse: (conn: StorageConnection) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rows, setRows] = useState<SavedConnection[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [highlightId, setHighlightId] = useState<string | null>(null);

  const meta = PROVIDER_META[provider];
  const logo = (PROVIDER_LOGOS as Record<string, string>)[meta.logoKey as string];

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const { data: userData } = await supabase.auth.getUser();
      const uid = userData.user?.id;
      if (!uid) throw new Error("Not signed in");
      const { data, error: err } = await supabase
        .from("cloud_datasets")
        .select("id,name,provider,region,bucket_name,connection_status,created_at")
        .eq("user_id", uid)
        .eq("provider", provider)
        .order("created_at", { ascending: false });
      if (err) throw new Error(err.message);
      setRows((data ?? []) as SavedConnection[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load connections");
    } finally {
      setLoading(false);
    }
  }, [provider]);

  useEffect(() => {
    if (open) {
      setHighlightId(null);
      void load();
    }
  }, [open, load]);

  const close = () => onOpenChange(false);

  const handleCreated = (conn: StorageConnection) => {
    setShowCreate(false);
    setHighlightId(conn.id);
    void load();
  };

  return (
    <>
      <Dialog open={open} onOpenChange={(o) => (o ? onOpenChange(true) : close())}>
        <DialogContent className="max-w-[640px] gap-0 overflow-hidden rounded-xl border border-border p-0 [&>button]:hidden">
          <div className="flex items-center justify-between border-b border-border bg-background px-5 py-4">
            <div className="flex items-center gap-2.5">
              <Cloud className="h-5 w-5 text-foreground" strokeWidth={1.75} />
              <h2 className="text-[17px] font-semibold tracking-tight">
                {meta.label} Connections
              </h2>
              {logo && (
                <img src={logo} alt={meta.label} className="ml-1 h-4 w-auto object-contain" />
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                onClick={() => setShowCreate(true)}
                className="bg-[#1565EF] text-white hover:bg-[#1257d6]"
              >
                <Plus className="mr-1 h-4 w-4" /> New Connection
              </Button>
              <button
                onClick={close}
                className="rounded p-1 text-muted-foreground hover:bg-muted"
                aria-label="Close"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between border-b border-border bg-background px-5 py-2">
            <p className="text-xs text-muted-foreground">
              Reuse a saved {meta.label} connection or add a new one.
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void load()}
              disabled={loading}
              className="gap-1"
            >
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
              Refresh
            </Button>
          </div>

          <div className="max-h-[52vh] overflow-y-auto bg-background">
            {loading ? (
              <div className="flex items-center justify-center gap-2 px-5 py-16 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading connections…
              </div>
            ) : error ? (
              <div className="mx-5 my-6 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                <p className="text-xs text-amber-800">{error}</p>
              </div>
            ) : rows.length === 0 ? (
              <p className="px-5 py-16 text-center text-sm text-muted-foreground">
                No {meta.label} connections yet — click New Connection to add one.
              </p>
            ) : (
              <ul className="divide-y divide-border/60">
                {rows.map((r) => (
                  <li
                    key={r.id}
                    className={cn(
                      "flex items-center justify-between gap-3 px-5 py-3",
                      highlightId === r.id && "bg-[#eef4ff]",
                    )}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="truncate text-sm font-medium text-foreground">
                          {r.name}
                        </span>
                        {r.connection_status === "connected" && (
                          <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 ring-1 ring-emerald-200">
                            Connected
                          </span>
                        )}
                      </div>
                      <p className="truncate text-xs text-muted-foreground">
                        {r.region ? `${r.region} · ` : ""}
                        {r.bucket_name || "—"}
                      </p>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        onBrowse({
                          id: r.id,
                          name: r.name,
                          provider: r.provider,
                          bucket_name: r.bucket_name ?? "",
                        })
                      }
                    >
                      Browse
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {showCreate && (
        <CloudStorageConnectModal
          open={showCreate}
          provider={provider}
          onOpenChange={(o) => setShowCreate(o)}
          onConnected={handleCreated}
        />
      )}
    </>
  );
}
