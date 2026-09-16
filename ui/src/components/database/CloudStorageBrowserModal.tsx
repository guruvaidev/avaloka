import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  AlertTriangle,
  ChevronRight,
  CornerLeftUp,
  Database,
  Folder,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { backendApi } from "@/lib/api/backendApi";
import { supabase } from "@/integrations/supabase/client";
import { createAnalysis as persistCreateAnalysis } from "@/lib/analysis-messages";
import type { StorageConnection, StorageProvider } from "./CloudStorageConnectModal";

type RemoteObject = { key: string; size?: string; updated?: string };
type RemoteFolder = { name: string; prefix: string };
type RowStatus =
  | { state: "idle" }
  | { state: "running" }
  | { state: "success"; datasetId: string; sessionId: string; threadId: string; largeFidelity?: string | null }
  | { state: "error"; message: string };

const PROVIDER_LABEL: Record<StorageProvider, string> = {
  aws: "Amazon S3",
  azure: "Azure Blob Storage",
  gcp: "Google Cloud Storage",
};

function backendFor(p: StorageProvider): "gcs" | "s3" | "azure" {
  return p === "gcp" ? "gcs" : p === "aws" ? "s3" : "azure";
}

function schemeFor(p: StorageProvider): "gs" | "s3" | "az" {
  return p === "gcp" ? "gs" : p === "aws" ? "s3" : "az";
}

function splitBucket(bucketPath: string): { bucket: string; prefix: string } {
  const cleaned = bucketPath.replace(/^[a-z]+:\/\//, "").replace(/\/+$/, "");
  const parts = cleaned.split("/").filter(Boolean);
  return { bucket: parts[0] ?? "", prefix: parts.slice(1).join("/") };
}

function fileType(key: string): string {
  const ext = key.split("/").pop()?.split(".").pop()?.toLowerCase();
  if (!ext) return "—";
  return ext.toUpperCase();
}

function fileName(key: string): string {
  return key.split("/").pop() || key;
}

function isPopulatedDatasetResponse(value: any): boolean {
  return (
    typeof value?.dataset_id === "string" &&
    value.dataset_id.length > 0 &&
    Array.isArray(value.samples) &&
    value.samples.length > 0
  );
}

export function CloudStorageBrowserModal({
  open,
  connection,
  onOpenChange,
}: {
  open: boolean;
  connection: StorageConnection | null;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [objects, setObjects] = useState<RemoteObject[]>([]);
  const [folders, setFolders] = useState<RemoteFolder[]>([]);
  const [listBucket, setListBucket] = useState<string>("");
  const [listPrefix, setListPrefix] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [selectedFolders, setSelectedFolders] = useState<RemoteFolder[]>([]);
  const [folderProgress, setFolderProgress] = useState<
    Record<string, { state: "pending" | "running" | "done" | "error"; message?: string }>
  >({});
  const [rowStatus, setRowStatus] = useState<Record<string, RowStatus>>({});
  const [registering, setRegistering] = useState(false);
  const [folderBusy, setFolderBusy] = useState(false);
  const [folderError, setFolderError] = useState("");
  const [folderTableType, setFolderTableType] = useState<string | null>(null);


  const { bucket, prefix } = useMemo(
    () => (connection ? splitBucket(connection.bucket_name) : { bucket: "", prefix: "" }),
    [connection],
  );

  const HIDDEN_PREFIXES = ["code-registry/", "execution-outputs/", "visualization-configs/"];
  const ALLOWED_EXT = [
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".parquet",
    ".avro",
    ".orc",
    ".xls",
    ".xlsx",
  ];

  const visibleObjects = useMemo(() => {
    return objects.filter((o) => {
      const k = o.key.toLowerCase();
      if (HIDDEN_PREFIXES.some((p) => k.startsWith(p))) return false;
      return ALLOWED_EXT.some((ext) => k.endsWith(ext));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [objects]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return visibleObjects;
    return visibleObjects.filter((o) => o.key.toLowerCase().includes(q));
  }, [visibleObjects, search]);

  const filteredFolders = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return folders;
    return folders.filter((f) => f.name.toLowerCase().includes(q));
  }, [folders, search]);

  const allSelected = filtered.length > 0 && filtered.every((o) => selected.has(o.key));
  const someSelected = !allSelected && filtered.some((o) => selected.has(o.key));

  const loadObjects = async (targetPrefix?: string) => {
    if (!connection) return;
    const nextPrefix = targetPrefix ?? listPrefix ?? prefix;
    setLoading(true);
    setLoadError("");
    try {
      const res = await backendApi.listBucketObjects({
        backend: backendFor(connection.provider),
        bucket,
        prefix: nextPrefix,
        connectionId: connection.id,
      });
      const items = (res.objects || []).filter((o) => o.key && !o.key.endsWith("/"));
      setObjects(items);
      setFolders((res.folders || []).filter((f) => f && f.prefix));
      setListBucket(res.bucket || bucket);
      setListPrefix(res.prefix ?? nextPrefix ?? "");
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Failed to list files");
    } finally {
      setLoading(false);
    }
  };

  const navigateTo = (targetPrefix: string) => {
    if (loading || registering || folderBusy) return;
    setSearch("");
    setFolderError("");
    setFolderTableType(null);
    setSelectedFolders([]);
    setListPrefix(targetPrefix);
    void loadObjects(targetPrefix);
  };

  const crumbs = useMemo(() => {
    const segments = (listPrefix || "").split("/").filter(Boolean);
    return segments.map((seg, i) => ({
      label: seg,
      prefix: segments.slice(0, i + 1).join("/"),
    }));
  }, [listPrefix]);

  const parentPrefix = crumbs.length > 1 ? crumbs[crumbs.length - 2].prefix : "";

  useEffect(() => {
    if (open && connection) {
      setSelected(new Set());
      setSelectedFolders([]);
      setRowStatus({});
      setSearch("");
      setFolders([]);
      setListPrefix(prefix);
      void loadObjects(prefix);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, connection?.id]);

  const toggleAll = () => {
    setSelectedFolders([]);
    if (allSelected) {
      const next = new Set(selected);
      filtered.forEach((o) => next.delete(o.key));
      setSelected(next);
    } else {
      const next = new Set(selected);
      filtered.forEach((o) => next.add(o.key));
      setSelected(next);
    }
  };

  const toggleOne = (key: string) => {
    setSelectedFolders([]);
    const next = new Set(selected);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setSelected(next);
  };

  const toggleFolder = (folder: RemoteFolder) => {
    setFolderError("");
    setFolderProgress({});
    setSelectedFolders((prev: RemoteFolder[]) =>
      prev.some((f) => f.prefix === folder.prefix)
        ? prev.filter((f) => f.prefix !== folder.prefix)
        : [...prev, folder],
    );
    setSelected(new Set());
  };



  const close = () => {
    if (registering) return;
    onOpenChange(false);
  };

  const buildStorageUri = () => {
    const scheme = schemeFor(connection!.provider);
    const b = listBucket || bucket;
    // /buckets/list now returns the full object path in key. Keep storage_uri
    // anchored to the bucket root so the folder path is neither dropped nor doubled.
    return `${scheme}://${b}`;
  };

  const handleUse = async () => {
    if (!connection || selected.size === 0) return;

    const keys = Array.from(selected);
    const storageUri = buildStorageUri();

    setRegistering(true);
    setRowStatus((prev) => {
      const next = { ...prev };
      keys.forEach((k) => (next[k] = { state: "running" }));
      return next;
    });

    type SuccessEntry = { response: any; filename: string; key: string };
    let firstSuccess: SuccessEntry | null = null;
    const allSuccesses: SuccessEntry[] = [];
    let successCount = 0;

    // SEQUENTIAL — large cloud objects hang the proxy when fired in parallel.
    for (const key of keys) {
      try {
        const v: any = await backendApi.registerExistingStorage({
          storage_uri: storageUri,
          key,
          connection_id: connection.id,
        });
        successCount += 1;
        setRowStatus((prev) => ({
          ...prev,
          [key]: {
            state: "success",
            datasetId: v.dataset_id,
            sessionId: v.session_id,
            threadId: v.thread_id,
            largeFidelity: v.requires_fidelity_choice
              ? v.fidelity_prompt ?? "Large — choose fidelity"
              : null,
          },
        }));
        if (isPopulatedDatasetResponse(v)) {
          const entry: SuccessEntry = { response: v, filename: fileName(key), key };
          allSuccesses.push(entry);
          if (!firstSuccess) firstSuccess = entry;
        }
      } catch (e) {
        setRowStatus((prev) => ({
          ...prev,
          [key]: {
            state: "error",
            message: e instanceof Error ? e.message : String(e),
          },
        }));
      }
    }

    setRegistering(false);

    // Persist registered keys on the connection so the dataset count on the
    // connection card reflects reality. Merge new keys with any previously
    // registered ones for this connection.
    if (successCount > 0) {
      try {
        const successKeys = allSuccesses.map((s) => s.key);
        const { data: existing } = await supabase
          .from("cloud_datasets")
          .select("schema")
          .eq("id", connection.id)
          .single();
        const schema = ((existing?.schema as Record<string, unknown>) ?? {}) as Record<string, unknown>;
        const prev = Array.isArray(schema.selected_files)
          ? (schema.selected_files as string[])
          : [];
        const merged = Array.from(new Set([...prev, ...successKeys]));
        schema.selected_files = merged;
        await supabase
          .from("cloud_datasets")
          .update({ schema })
          .eq("id", connection.id);
        queryClient.invalidateQueries({ queryKey: ["data_sources"] });
      } catch {
        /* non-fatal: count refresh only */
      }
    }

    toast[successCount ? "success" : "error"](
      `Registered ${successCount} of ${keys.length}`,
    );

    if (firstSuccess) {
      const fs: SuccessEntry = firstSuccess;
      const v = fs.response;

      const additional = allSuccesses
        .filter((s) => s.response.dataset_id !== v.dataset_id)
        .map((s) => ({
          dataset_id: s.response.dataset_id,
          session_id: s.response.session_id,
          thread_id: s.response.thread_id,
          filename: s.filename,
          alias: s.filename,
          schema: s.response.schema,
          samples: s.response.samples,
          rows_sampled:
            s.response.rows_sampled ??
            (Array.isArray(s.response.samples) ? s.response.samples.length : undefined),
          visualization_config:
            s.response.visualization_config ?? s.response.visualization_configs,
          visualization_status: s.response.visualization_status,
          analysis_fidelity: s.response.analysis_fidelity,
          selected_sample_name: s.response.selected_sample_name,
        }));

      const uploadResponse: Record<string, any> = {
        dataset_id: v.dataset_id,
        session_id: v.session_id,
        thread_id: v.thread_id,
        schema: v.schema,
        samples: v.samples,
        rows_sampled:
          v.rows_sampled ?? (Array.isArray(v.samples) ? v.samples.length : undefined),
        file_size_mb: v.file_size_mb,
        file_size_bytes: v.file_size_bytes,

        analysis_fidelity: v.analysis_fidelity,
        selected_sample_name: v.selected_sample_name,
        available_samples: v.available_samples,
        visualization_config: v.visualization_config ?? v.visualization_configs,
        visualization_status: v.visualization_status,
        requires_fidelity_choice: v.requires_fidelity_choice,
        fidelity_prompt: v.fidelity_prompt,
      };
      if (additional.length) {
        uploadResponse.datasets = [
          {
            dataset_id: v.dataset_id,
            session_id: v.session_id,
            thread_id: v.thread_id,
            filename: fs.filename,
            alias: fs.filename,
            schema: v.schema,
            samples: v.samples,
            rows_sampled:
              v.rows_sampled ?? (Array.isArray(v.samples) ? v.samples.length : undefined),
            visualization_config: v.visualization_config ?? v.visualization_configs,
            visualization_status: v.visualization_status,
            analysis_fidelity: v.analysis_fidelity,
            selected_sample_name: v.selected_sample_name,
          },
          ...additional,
        ];
      }

      // Create a fresh analyses row for THIS newly-registered dataset so that
      // ?aid= in the URL points at the new analysis — never the previously
      // opened one. Mirrors the file-upload flow (no project attached).
      let newAnalysisId: string | null = null;
      try {
        const created = await persistCreateAnalysis({
          project_id: null,
          name: fs.filename || "New Analysis",

          dataset_id: v.dataset_id ?? null,
          thread_id: v.thread_id ?? null,
          session_id: v.session_id ?? null,
          filename: fs.filename ?? null,
          schema: v.schema ?? null,
          samples: v.samples ?? null,
          viz_config: v.visualization_config ?? v.visualization_configs ?? null,
        });
        newAnalysisId = created.id;
      } catch (err) {
        console.warn("Failed to pre-create analysis row for cloud dataset", err);
      }

      onOpenChange(false);
      navigate({
        to: "/analysis",
        search: (newAnalysisId ? { aid: newAnalysisId } : {}) as any,
        state: {
          uploadResponse,
          filename: fs.filename,
          newAnalysisId,
        } as any,
      });
    } else if (successCount > 0) {
      toast.error("The registered files did not return a populated dataset preview.");
    }
  };

  const handleRegisterFolder = async (override?: RemoteFolder[]) => {
    if (!connection) return;
    const targets: RemoteFolder[] = override?.length
      ? override
      : selectedFolders.length
      ? selectedFolders
      : listPrefix
        ? [
            {
              name: listPrefix.split("/").filter(Boolean).pop() || listPrefix,
              prefix: listPrefix,
            },
          ]
        : [];
    if (!targets.length) return;


    setFolderError("");
    setFolderTableType(null);
    setFolderBusy(true);
    setFolderProgress(
      Object.fromEntries(targets.map((f) => [f.prefix, { state: "pending" as const }])),
    );

    type FolderSuccess = { response: any; name: string };
    const successes: FolderSuccess[] = [];
    const failures: { name: string; message: string }[] = [];

    // SEQUENTIAL — one register call per folder (no batch endpoint).
    for (const folder of targets) {
      const targetPrefix = folder.prefix.replace(/^\/+|\/+$/g, "");
      setFolderProgress((prev) => ({ ...prev, [folder.prefix]: { state: "running" } }));
      try {
        const v: any = await backendApi.registerExistingFolder({
          storage_uri: buildStorageUri(),
          folder: targetPrefix,
          connection_id: connection.id,
        });
        successes.push({ response: v, name: folder.name || targetPrefix });
        if (v?.table_type) setFolderTableType(String(v.table_type));
        setFolderProgress((prev) => ({ ...prev, [folder.prefix]: { state: "done" } }));
      } catch (e) {
        const err = e as Error & { status?: number };
        const is422 =
          err.status === 422 || /not a recognized parquet or iceberg table/i.test(err.message);
        const message = is422
          ? "Not a partitioned Parquet/Iceberg table — open it and select individual files instead."
          : err.message || "Failed to register folder";
        failures.push({ name: folder.name || targetPrefix, message });
        setFolderProgress((prev) => ({
          ...prev,
          [folder.prefix]: { state: "error", message },
        }));
      }
    }

    setFolderBusy(false);

    if (failures.length) {
      setFolderError(
        `${failures.length} folder(s) could not be registered: ${failures
          .map((f) => `${f.name} — ${f.message}`)
          .join(" | ")}`,
      );
    }

    if (!successes.length) {
      toast.error("No folders could be registered");
      return;
    }

    const first = successes[0];
    const v = first.response;
    const toDataset = (s: FolderSuccess) => ({
      dataset_id: s.response.dataset_id,
      session_id: s.response.session_id,
      thread_id: s.response.thread_id,
      filename: s.name,
      alias: s.name,
      schema: s.response.schema,
      samples: s.response.samples,
      rows_sampled:
        s.response.rows_sampled ??
        (Array.isArray(s.response.samples) ? s.response.samples.length : undefined),
      file_size_mb: s.response.file_size_mb,
      file_size_bytes: s.response.file_size_bytes,
      visualization_config: s.response.visualization_config ?? s.response.visualization_configs,
      visualization_status: s.response.visualization_status,
      analysis_fidelity: s.response.analysis_fidelity,
      selected_sample_name: s.response.selected_sample_name,
    });

    const uploadResponse: Record<string, any> = {
      dataset_id: v.dataset_id,
      session_id: v.session_id,
      thread_id: v.thread_id,
      schema: v.schema,
      samples: v.samples,
      rows_sampled:
        v.rows_sampled ?? (Array.isArray(v.samples) ? v.samples.length : undefined),
      file_size_mb: v.file_size_mb,
      file_size_bytes: v.file_size_bytes,
      analysis_fidelity: v.analysis_fidelity,
      selected_sample_name: v.selected_sample_name,
      available_samples: v.available_samples,
      visualization_config: v.visualization_config ?? v.visualization_configs,
      visualization_status: v.visualization_status,
      table_type: v.table_type,
    };
    if (successes.length > 1) {
      uploadResponse.datasets = successes.map(toDataset);
    }

    // Registered datasets go to the same non-project analysis surface as a
    // normal file upload (proanalysis → "Recent"), never into a project.
    let newAnalysisId: string | null = null;
    try {
      const created = await persistCreateAnalysis({
        project_id: null,
        name: first.name || "New Analysis",
        dataset_id: v.dataset_id ?? null,
        thread_id: v.thread_id ?? null,
        session_id: v.session_id ?? null,
        filename: first.name ?? null,
        schema: v.schema ?? null,
        samples: v.samples ?? null,
        viz_config: v.visualization_config ?? v.visualization_configs ?? null,
      });
      newAnalysisId = created.id;
      await queryClient.invalidateQueries({ queryKey: ["analyses"] });
    } catch (err) {
      console.warn("Failed to pre-create analysis row for folder dataset", err);
    }

    toast.success(
      successes.length > 1
        ? `Registered ${successes.length} folders as datasets`
        : v?.table_type
          ? `Registered folder as ${v.table_type} table`
          : "Folder registered",
    );
    onOpenChange(false);
    navigate({
      to: "/analysis",
      search: (newAnalysisId ? { aid: newAnalysisId } : {}) as any,
      state: { uploadResponse, filename: first.name, newAnalysisId, showAutoInsights: true } as any,
    });
  };


  const retry = async (key: string) => {
    if (!connection) return;
    setRowStatus((prev) => ({ ...prev, [key]: { state: "running" } }));
    try {
      const res: any = await backendApi.registerExistingStorage({
        storage_uri: buildStorageUri(),
        key,
        connection_id: connection.id,
      });
      setRowStatus((prev) => ({
        ...prev,
        [key]: {
          state: "success",
          datasetId: res.dataset_id,
          sessionId: res.session_id,
          threadId: res.thread_id,
          largeFidelity: res.requires_fidelity_choice ? res.fidelity_prompt ?? null : null,
        },
      }));
    } catch (e) {
      setRowStatus((prev) => ({
        ...prev,
        [key]: { state: "error", message: e instanceof Error ? e.message : "Retry failed" },
      }));
    }
  };

  if (!connection) return null;

  const folderDone = Object.values(folderProgress).filter((p) => p.state === "done").length;
  const PARTITION_RE = /^[^/=]+=[^/=]+$/;
  const partitionSelected = selectedFolders.filter((f) =>
    PARTITION_RE.test((f.name || f.prefix.replace(/\/+$/, "").split("/").pop() || "").trim()),
  );
  const hasPartitionSelection = partitionSelected.length > 0;
  const partitionParentPrefix = hasPartitionSelection
    ? partitionSelected[0].prefix.replace(/\/+$/, "").split("/").slice(0, -1).join("/")
    : "";
  const partitionParentName = partitionParentPrefix.split("/").filter(Boolean).pop() || "";
  const selectionLabel = selectedFolders.length
    ? folderBusy
      ? `Registering ${folderDone + 1} of ${selectedFolders.length} folder(s)…`
      : `${selectedFolders.length} folder(s) selected`
    : selected.size === 1
      ? fileName(Array.from(selected)[0])
      : `${selected.size} file(s) selected`;




  return (
    <Dialog open={open} onOpenChange={(o) => (o ? onOpenChange(true) : close())}>
      <DialogContent className="max-w-[820px] gap-0 overflow-hidden rounded-xl border border-border p-0 [&>button]:hidden">
        <div className="flex items-center justify-between border-b border-border bg-background px-5 py-4">
          <div className="flex items-center gap-2.5">
            <Database className="h-5 w-5 text-foreground" strokeWidth={1.75} />
            <div>
              <h2 className="text-[15px] font-semibold tracking-tight">
                Datasets in {connection.name}
              </h2>
              <p className="text-xs text-muted-foreground">
                Bucket: {connection.bucket_name} ({PROVIDER_LABEL[connection.provider]})
              </p>
            </div>
          </div>
          <button
            onClick={close}
            className="rounded p-1 text-muted-foreground hover:bg-muted"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex items-center gap-2 border-b border-border bg-background px-5 py-3">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search files by name..."
              className="h-9 bg-background pl-8"
            />
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void loadObjects()}
            disabled={loading}
            className="gap-1"
          >
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
            Refresh
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-1 border-b border-border bg-background px-5 py-2 text-xs">
          <button
            type="button"
            onClick={() => navigateTo(parentPrefix)}
            disabled={loading || crumbs.length === 0}
            className="mr-1 inline-flex items-center gap-1 rounded px-1.5 py-1 text-muted-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
            title="Up one level"
          >
            <CornerLeftUp className="h-3.5 w-3.5" /> Up
          </button>
          <button
            type="button"
            onClick={() => navigateTo("")}
            disabled={loading}
            className={cn(
              "rounded px-1.5 py-1 hover:bg-muted",
              crumbs.length === 0 ? "font-semibold text-foreground" : "text-muted-foreground",
            )}
          >
            {listBucket || bucket || "Bucket"}
          </button>
          {crumbs.map((c, i) => (
            <span key={c.prefix} className="flex items-center gap-1">
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <button
                type="button"
                onClick={() => navigateTo(c.prefix)}
                disabled={loading}
                className={cn(
                  "rounded px-1.5 py-1 hover:bg-muted",
                  i === crumbs.length - 1
                    ? "font-semibold text-foreground"
                    : "text-muted-foreground",
                )}
              >
                {c.label}
              </button>
            </span>
          ))}
        </div>

        <div className="max-h-[52vh] overflow-y-auto bg-background">
          {loading ? (
            <div className="flex items-center justify-center gap-2 px-5 py-16 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading files…
            </div>
          ) : loadError ? (
            <div className="mx-5 my-6 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
              <p className="text-xs text-amber-800">{loadError}</p>
            </div>
          ) : filtered.length === 0 && filteredFolders.length === 0 ? (
            <p className="px-5 py-16 text-center text-sm text-muted-foreground">
              No files or folders found in this bucket/prefix.
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10 border-b border-border bg-muted/40 text-xs text-muted-foreground">
                <tr>
                  <th className="w-10 px-3 py-2">
                    <input
                      type="checkbox"
                      aria-label="Select all files"
                      className="h-4 w-4 cursor-pointer accent-[#1565EF]"
                      checked={allSelected}
                      ref={(el) => {
                        if (el) el.indeterminate = someSelected;
                      }}
                      onChange={toggleAll}
                    />
                  </th>
                  <th className="px-3 py-2 text-left font-medium">File Name</th>
                  <th className="px-3 py-2 text-left font-medium">Size</th>
                  <th className="px-3 py-2 text-left font-medium">Last Modified</th>
                  <th className="px-3 py-2 text-left font-medium">Type</th>
                  <th className="px-3 py-2 text-left font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredFolders.map((f) => {
                  const isFolderSel = selectedFolders.some((sf) => sf.prefix === f.prefix);
                  const fProgress = folderProgress[f.prefix];

                  return (
                    <tr
                      key={`folder:${f.prefix}`}
                      className={cn(
                        "border-b border-border/60 hover:bg-muted/30",
                        isFolderSel && "bg-[#eef4ff]",
                      )}
                    >
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            aria-label={`Select folder ${f.name}`}
                            className="h-4 w-4 cursor-pointer accent-[#1565EF]"
                            checked={isFolderSel}
                            onChange={() => toggleFolder(f)}
                          />
                        </div>
                      </td>
                      <td className="px-3 py-2 font-medium text-foreground">
                        <span className="inline-flex items-center gap-2">
                          <Folder className="h-4 w-4 shrink-0 text-[#1565EF]" />
                          {f.name}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-muted-foreground">—</td>
                      <td className="px-3 py-2 text-muted-foreground">—</td>
                      <td className="px-3 py-2 text-muted-foreground">Folder</td>
                      <td className="px-3 py-2 text-xs">
                        <div className="flex items-center gap-3">
                          <button
                            type="button"
                            onClick={() => navigateTo(f.prefix)}
                            className="text-[#1565EF] underline underline-offset-2 hover:text-[#1257d6]"
                          >
                            Open
                          </button>
                          {fProgress?.state === "running" && (
                            <span className="inline-flex items-center gap-1 text-muted-foreground">
                              <Loader2 className="h-3 w-3 animate-spin" /> Registering…
                            </span>
                          )}
                          {fProgress?.state === "pending" && (
                            <span className="text-muted-foreground">Queued</span>
                          )}
                          {fProgress?.state === "done" && (
                            <span className="text-emerald-700">Registered</span>
                          )}
                          {fProgress?.state === "error" && (
                            <span className="text-destructive" title={fProgress.message}>
                              Failed
                            </span>
                          )}
                        </div>
                      </td>

                    </tr>
                  );
                })}

                {filtered.map((o) => {
                  const isSel = selected.has(o.key);
                  const status = rowStatus[o.key];
                  return (
                    <tr
                      key={o.key}
                      onClick={() => toggleOne(o.key)}
                      className={cn(
                        "cursor-pointer border-b border-border/60 hover:bg-muted/30",
                        isSel && "bg-[#eef4ff]",
                      )}
                    >
                      <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          aria-label={`Select ${o.key}`}
                          className="h-4 w-4 cursor-pointer accent-[#1565EF]"
                          checked={isSel}
                          onChange={() => toggleOne(o.key)}
                        />
                      </td>
                      <td className="px-3 py-2 font-medium text-foreground">{o.key}</td>
                      <td className="px-3 py-2 text-muted-foreground">{o.size || "—"}</td>
                      <td className="px-3 py-2 text-muted-foreground">{o.updated || "—"}</td>
                      <td className="px-3 py-2 text-muted-foreground">{fileType(o.key)}</td>
                      <td className="px-3 py-2">
                        {status?.state === "running" && (
                          <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                            <Loader2 className="h-3 w-3 animate-spin" /> Registering…
                          </span>
                        )}
                        {status?.state === "success" && (
                          <span className="text-xs text-emerald-700">
                            Registered
                            {status.largeFidelity ? " — large, needs fidelity choice" : ""}
                          </span>
                        )}
                        {status?.state === "error" && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              void retry(o.key);
                            }}
                            className="text-xs text-red-600 underline"
                            title={status.message}
                          >
                            Failed — Retry
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {hasPartitionSelection && (
          <div className="border-t border-border bg-background px-5 py-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-500">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                These look like partitions of a table. Go up one level and register the parent
                folder as a single dataset instead.
              </span>
              {partitionParentPrefix ? (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={registering || folderBusy}
                  onClick={() =>
                    void handleRegisterFolder([
                      { name: partitionParentName || partitionParentPrefix, prefix: partitionParentPrefix },
                    ])
                  }
                >
                  <Folder className="mr-2 h-4 w-4" />
                  Register parent folder ({partitionParentName || partitionParentPrefix}) as one
                  dataset
                </Button>
              ) : null}
            </div>
          </div>
        )}

        {(folderError || folderTableType) && (
          <div className="border-t border-border bg-background px-5 py-2">
            {folderTableType && (
              <span className="mr-2 inline-flex items-center rounded-full border border-border bg-muted px-2 py-0.5 text-[11px] font-medium text-foreground">
                {folderTableType} table
              </span>
            )}
            {folderError && (
              <span className="inline-flex items-start gap-1.5 text-xs text-destructive">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                {folderError}
              </span>
            )}
          </div>
        )}

        <div className="flex items-center justify-between border-t border-border bg-background px-5 py-3">
          <p className="text-xs text-muted-foreground">{selectionLabel}</p>
          <div className="flex items-center gap-2">
            {(() => {
              const folderMode = selectedFolders.length > 0;
              const nothingSelected = !folderMode && selected.size === 0;
              const busy = registering || folderBusy;
              return (
                <Button
                  onClick={() => (folderMode ? void handleRegisterFolder() : handleUse())}
                  disabled={nothingSelected || busy || (folderMode && hasPartitionSelection)}
                  className="bg-[#1565EF] text-white hover:bg-[#1257d6]"
                  title={
                    folderMode && hasPartitionSelection
                      ? "Partition folders can't be registered individually — register the parent folder instead"
                      : folderMode
                        ? "Register the selected folder(s) as dataset(s)"
                        : "Register the selected file(s) as dataset(s)"
                  }
                >
                  {busy ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {folderBusy
                        ? selectedFolders.length > 1
                          ? `Registering ${folderDone + 1}/${selectedFolders.length}…`
                          : "Registering folder…"
                        : "Registering…"}
                    </>
                  ) : folderMode ? (
                    <>
                      <Folder className="mr-2 h-4 w-4" />
                      {selectedFolders.length > 1
                        ? `Use ${selectedFolders.length} folders`
                        : "Use folder"}
                    </>
                  ) : selected.size > 1 ? (
                    `Use ${selected.size} datasets`
                  ) : (
                    "Use dataset"
                  )}
                </Button>
              );
            })()}
          </div>
        </div>

      </DialogContent>
    </Dialog>
  );
}
