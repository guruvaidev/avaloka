import { useEffect, useState, type ReactElement } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import {
  Cloud,
  X,
  Check,
  AlertCircle,
  Eye,
  EyeOff,
  HelpCircle,
  Loader2,
  ChevronLeft,
  ChevronDown,
  CloudUpload,
  Trash2,
  ArrowUp,
  Briefcase,
  Search,
  AlertTriangle,
  RefreshCw,
  Link2,
  SquarePen,
} from "lucide-react";
import { PROVIDERS, getProvider, GCP_REGION_OPTIONS } from "./providers";
import { PROVIDER_LOGOS } from "./logos";
import type { Provider } from "@/lib/data-sources";
import { useDataSourceMutations } from "@/lib/data-sources";
import { backendApi } from "@/lib/api/backendApi";
import {
  buildGcsDatasetPath,
  createGcpCloudConnection,
  finalizeGcpCloudConnection,
  parseGcsDatasetPath,
} from "@/lib/gcp-connection";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
const gcsLogo = { url: "/assets/logos/Logo_Type_GCS.png" };
const s3Logo = { url: "/assets/logos/Logo_Type_S3.png" };
const azureLogo = { url: "/assets/logos/Logo_Type_Azure.png" };
import {
  CloudStorageConnectionsListModal,
} from "./CloudStorageConnectionsListModal";
import { CloudStorageBrowserModal } from "./CloudStorageBrowserModal";
import type {
  StorageConnection,
  StorageProvider,
} from "./CloudStorageConnectModal";

const STORAGE_TILES: { id: StorageProvider; name: string; logoUrl: string }[] = [
  { id: "gcp", name: "Google Cloud Storage", logoUrl: gcsLogo.url },
  { id: "aws", name: "Amazon S3", logoUrl: s3Logo.url },
  { id: "azure", name: "Azure Blob Storage", logoUrl: azureLogo.url },
];


type RemoteFile = { id: string; name: string; type: "CSV" | "XLSX"; size: string; columns: number; rows: number };
type UploadFile = {
  id: string;
  name: string;
  type: "CSV" | "XLSX";
  size: string;
  progress: number;
  status: "uploading" | "done" | "failed";
};

type Step =
  | "select-provider"
  | "credentials"
  | "testing"
  | "auth-error"
  | "files"
  | "uploading"
  | "metadata"
  | "preview"
  | "confirm";

const TOTAL_STEPS = 5;

const MOCK_FILES: RemoteFile[] = [
  { id: "doc-01", name: "Doc.01", type: "CSV", size: "120MB", columns: 12, rows: 10 },
  { id: "doc-04", name: "Doc.04", type: "XLSX", size: "80MB", columns: 5, rows: 25 },
  { id: "doc-05", name: "Doc.05", type: "CSV", size: "90MB", columns: 8, rows: 15 },
  { id: "doc-08", name: "Doc.08", type: "XLSX", size: "45MB", columns: 6, rows: 30 },
  { id: "doc-12", name: "Doc.12", type: "CSV", size: "210MB", columns: 18, rows: 42 },
];

const STEP_NUMBER: Record<Step, number> = {
  "select-provider": 1,
  credentials: 2,
  testing: 2,
  "auth-error": 2,
  files: 3,
  uploading: 3,
  metadata: 4,
  preview: 5,
  confirm: 5,
};

const STEP_META: Record<Step, { title: string; subtitle: string }> = {
  "select-provider": { title: "Select Data Source", subtitle: "Choose the cloud database you want to connect" },
  credentials: { title: "Authenticate & Connect", subtitle: "Enter your database credentials to establish a secure connection" },
  testing: { title: "Authenticate & Connect", subtitle: "Verifying your credentials" },
  "auth-error": { title: "Authenticate & Connect", subtitle: "Enter your database credentials to establish a secure connection" },
  files: { title: "Select Data", subtitle: "" },
  uploading: { title: "Uploading", subtitle: "Importing selected files" },
  metadata: { title: "Dataset Basics & Refresh Settings", subtitle: "" },
  preview: { title: "Preview Dataset", subtitle: "Review your data before finalizing the connection" },
  confirm: { title: "Connection Successful", subtitle: "Your data source is ready to use" },
};

const CONNECTION_NAME_PLACEHOLDER: Partial<Record<string, string>> = {
  mysql: "MySQL_Production_DB",
  snowflake: "Snowflake_Finance_Prod",
  postgres: "Postgres Analytics DB",
  mssql: "MSSQL_Production_DB",
  bigquery: "GCP_Revenue_Dataset",
};

function gcsObjectsToRemoteFiles(
  objects: Array<{ key: string; size?: string }>,
): RemoteFile[] {
  return objects
    .map((o) => {
      const name = o.key.split("/").pop() || o.key;
      if (!name || name.endsWith("/")) return null;
      const ext = name.split(".").pop()?.toLowerCase();
      if (!ext || !["csv", "xlsx", "xls"].includes(ext)) return null;
      return {
        id: o.key,
        name,
        type: ext === "csv" ? ("CSV" as const) : ("XLSX" as const),
        size: o.size || "—",
        columns: 0,
        rows: 0,
      };
    })
    .filter(Boolean) as RemoteFile[];
}

export function ConnectCloudModal({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [step, setStep] = useState<Step>("select-provider");
  const [provider, setProvider] = useState<Provider | null>(null);
  const [creds, setCreds] = useState<Record<string, string>>({});
  const [showPw, setShowPw] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [uploads, setUploads] = useState<UploadFile[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [meta, setMeta] = useState({ name: "", description: "", domain: "", use_cases: "" });
  const [errorMsg, setErrorMsg] = useState("");
  const [remoteFiles, setRemoteFiles] = useState<RemoteFile[]>([]);
  const [saFileName, setSaFileName] = useState("");
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);
  const [showConnectConfirm, setShowConnectConfirm] = useState(false);
  const { create, insertFiles } = useDataSourceMutations();
  const queryClient = useQueryClient();

  const reset = () => {
    setStep("select-provider");
    setProvider(null);
    setCreds({});
    setSelectedFiles(new Set());
    setUploads([]);
    setUploadProgress(0);
    setMeta({ name: "", description: "", domain: "", use_cases: "" });
    setErrorMsg("");
    setRemoteFiles([]);
    setSaFileName("");
    setShowPw(false);
    setShowCancelConfirm(false);
    setShowConnectConfirm(false);
  };

  const [storageConnectProvider, setStorageConnectProvider] =
    useState<StorageProvider | null>(null);
  const [storageBrowseConnection, setStorageBrowseConnection] =
    useState<StorageConnection | null>(null);


  const close = () => {
    onOpenChange(false);
    setTimeout(reset, 250);
  };

  const requestClose = () => {
    if (step !== "select-provider" && step !== "confirm") {
      setShowCancelConfirm(true);
    } else {
      close();
    }
  };

  const providerMeta = provider ? getProvider(provider) : null;

  // Auto-close success after a moment
  useEffect(() => {
    if (step !== "confirm") return;
    const t = setTimeout(() => close(), 1800);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  const goBack = () => {
    if (step === "credentials" || step === "auth-error") setStep("select-provider");
    else if (step === "files") setStep("credentials");
    else if (step === "metadata") setStep("files");
    else if (step === "preview") setStep("metadata");
  };

  const handleTestAuth = async () => {
    setErrorMsg("");
    setStep("testing");

    if (provider === "bigquery") {
      try {
        const saJson = creds.service_account_json?.trim();
        if (!saJson) throw new Error("Upload a service account JSON key file.");
        if (!creds.region?.trim()) throw new Error("Region is required.");
        if (!creds.bucket_name?.trim()) throw new Error("Bucket name is required.");
        if (!creds.folder_path?.trim()) throw new Error("Folder path is required.");

        const datasetPath = buildGcsDatasetPath(creds.bucket_name, creds.folder_path);
        const { bucket, prefix } = parseGcsDatasetPath(datasetPath, creds.project_id);
        const connection = await createGcpCloudConnection({
          name: creds.connectionName || `${providerMeta?.name ?? "GCP"} connection`,
          projectId: creds.project_id,
          region: creds.region,
          datasetPath,
          serviceAccountJson: saJson,
        });

        const listing = await backendApi.listBucketObjects({
          backend: "gcs",
          bucket,
          prefix,
          connectionId: connection.id,
        });

        const files = gcsObjectsToRemoteFiles(listing.objects);
        if (!files.length) {
          throw new Error(
            "Connected successfully but no CSV or Excel files were found in that folder.",
          );
        }

        setCreds((prev) => ({
          ...prev,
          connection_id: connection.id,
          gcs_bucket: bucket,
          gcs_prefix: prefix,
        }));
        setRemoteFiles(files);
        setStep("files");
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Authentication failed.";
        setErrorMsg(msg);
        setStep("auth-error");
      }
      return;
    }

    setTimeout(() => {
      const pw = creds.password ?? "";
      if (!pw || pw.toLowerCase().includes("fail")) {
        const msg = "Authentication failed. Check credentials and try again.";
        setErrorMsg(msg);
        setStep("auth-error");
      } else {
        setRemoteFiles([]);
        setStep("files");
      }
    }, 900);
  };

  const handleStartUpload = () => {
    if (selectedFiles.size === 0 && uploads.filter((u) => u.status === "done").length === 0) {
      toast.error("Select at least one file");
      return;
    }
    if (selectedFiles.size === 0) {
      // Local uploads already done — skip to metadata
      setStep("metadata");
      return;
    }
    setStep("uploading");
    setUploadProgress(0);
    let p = 0;
    const t = setInterval(() => {
      p += 12 + Math.random() * 18;
      if (p >= 100) {
        clearInterval(t);
        setUploadProgress(100);
        setTimeout(() => setStep("metadata"), 350);
      } else setUploadProgress(Math.min(99, p));
    }, 220);
  };

  const handleConfirmConnect = async () => {
    if (!provider) return;
    try {
      const name = meta.name || creds.connectionName || `${providerMeta?.name} connection`;
      const selected = Array.from(selectedFiles);

      if (provider === "bigquery") {
        const connectionId = creds.connection_id;
        if (!connectionId) throw new Error("Missing cloud connection. Authenticate again.");
        await finalizeGcpCloudConnection({
          connectionId,
          name,
          description: meta.description || null,
          domain: meta.domain || null,
          use_cases: meta.use_cases || null,
          selectedFiles: selected,
        });
        await queryClient.invalidateQueries({ queryKey: ["data_sources"] });
        setStep("confirm");
        return;
      }

      const {
        password: _pw,
        service_account_json: _sa,
        ...safeConfig
      } = creds;
      const ds = await create.mutateAsync({
        provider,
        name,
        description: meta.description || null,
        domain: meta.domain || null,
        use_cases: meta.use_cases || null,
        config: { ...safeConfig, selected_files: selected },
        status: "connected",
      });
      if (selected.length > 0) {
        await insertFiles.mutateAsync({
          dataSourceId: ds.id,
          files: selected.map((p) => ({ path: p, selected: true, uploaded: true })),
        });
      }
      setStep("confirm");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const stepNum = STEP_NUMBER[step];
  const stepInfo = STEP_META[step];
  const progressPct = (stepNum / TOTAL_STEPS) * 100;
  const canGoBack = step !== "select-provider" && step !== "testing" && step !== "uploading" && step !== "confirm";

  return (
    <>
    <Dialog open={open} onOpenChange={(o) => (o ? onOpenChange(true) : requestClose())}>
      <DialogContent className="max-w-[560px] gap-0 overflow-hidden rounded-xl border border-border p-0 [&>button]:hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border bg-background px-5 py-4">
          <div className="flex items-center gap-2.5">
            {canGoBack ? (
              <button
                onClick={goBack}
                className="rounded p-1 hover:bg-muted"
                aria-label="Back"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
            ) : (
              <Cloud className="h-5 w-5 text-foreground" strokeWidth={1.75} />
            )}
            <h2 className="text-[17px] font-semibold tracking-tight">Cloud Connect</h2>
            {providerMeta && step !== "select-provider" && (
              <span className="ml-1 inline-flex items-center rounded-md border border-border bg-background dark:bg-white px-2 py-1">
                <img
                  src={PROVIDER_LOGOS[providerMeta.id]}
                  alt={providerMeta.name}
                  className="h-3.5 w-auto object-contain"
                />
              </span>
            )}
          </div>
          <button onClick={requestClose} className="rounded p-1 text-muted-foreground hover:bg-muted" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="bg-background px-6 pb-6 pt-5">
          <div className="mb-3 flex items-start justify-between gap-4">
            <div>
              <div className="text-sm font-semibold text-foreground">{stepInfo.title}</div>
              {step !== "select-provider" && (
                <div className="text-xs text-muted-foreground">{stepInfo.subtitle}</div>
              )}
            </div>
            <div className="whitespace-nowrap text-xs font-medium text-muted-foreground">
              Step {stepNum} of {TOTAL_STEPS}
            </div>
          </div>

          <div className="mb-5 h-1 w-full overflow-hidden rounded-full bg-[#eef0f4]">
            <div
              className="h-full rounded-full bg-[#1565EF] transition-all"
              style={{ width: `${progressPct}%` }}
            />
          </div>

          {/* Error notification banner */}
          {step === "auth-error" && errorMsg && (
            <div className="mb-4 flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
              <div className="flex-1">
                <p className="text-sm font-medium text-amber-800">Connection failed</p>
                <p className="text-xs text-amber-700">{errorMsg}</p>
              </div>
              <button
                onClick={() => setErrorMsg("")}
                className="text-amber-600 hover:text-amber-800"
                aria-label="Dismiss error"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          <div className="min-h-[280px]">
            {step === "select-provider" && (
              <div className="grid grid-cols-3 gap-3">
                {PROVIDERS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setProvider(p.id)}
                    className={cn(
                      "flex h-[78px] items-center justify-center rounded-lg border bg-background dark:bg-white px-3 transition hover:border-[#1565EF] hover:shadow-sm",
                      provider === p.id ? "border-[#1565EF] ring-1 ring-[#1565EF]/30" : "border-border",
                    )}
                  >
                    <img
                      src={PROVIDER_LOGOS[p.id]}
                      alt={p.name}
                      className="max-h-9 max-w-[120px] object-contain"
                    />
                  </button>
                ))}
                {STORAGE_TILES.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setStorageConnectProvider(t.id)}
                    className={cn(
                      "flex h-[78px] items-center justify-center rounded-lg border border-border bg-background dark:bg-white px-3 transition hover:border-[#1565EF] hover:shadow-sm",
                    )}
                  >
                    <img
                      src={t.logoUrl}
                      alt={t.name}
                      className="max-h-9 max-w-[120px] object-contain"
                    />
                  </button>
                ))}
              </div>
            )}


            {(step === "credentials" || step === "auth-error") && providerMeta && (
              <div className="space-y-3">
                <FormField label="Connection Name" required>
                  <Input
                    placeholder={CONNECTION_NAME_PLACEHOLDER[providerMeta.id] ?? `${providerMeta.name} connection`}
                    value={creds.connectionName ?? ""}
                    onChange={(e) => setCreds({ ...creds, connectionName: e.target.value })}
                    className="h-10 bg-background"
                  />
                </FormField>

                {providerMeta.hasConnectionType && (
                  <FormField label="Connection Type" required>
                    <div className="space-y-2 pt-0.5">
                      {[
                        { id: "live", title: "Live Connection", desc: "Query data directly from source" },
                        { id: "synced", title: "Synced dataset", desc: "Sync dataset into Avaloka for faster analysis" },
                      ].map((opt) => {
                        const selected = (creds.connectionType ?? "live") === opt.id;
                        return (
                          <label key={opt.id} className="flex cursor-pointer items-start gap-3">
                            <span
                              className={cn(
                                "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                                selected ? "border-[#1565EF]" : "border-border",
                              )}
                            >
                              {selected && <span className="h-2 w-2 rounded-full bg-[#1565EF]" />}
                            </span>
                            <input
                              type="radio"
                              name="connectionType"
                              className="sr-only"
                              checked={selected}
                              onChange={() => setCreds({ ...creds, connectionType: opt.id })}
                            />
                            <span className="leading-tight">
                              <span className="block text-sm font-medium text-foreground">{opt.title}</span>
                              <span className="block text-xs text-muted-foreground">{opt.desc}</span>
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </FormField>
                )}

                {providerMeta.fields
                  .filter((f) => f.key !== "password" && f.type !== "file")
                  .reduce<ReactElement[]>((acc, f, idx, arr) => {
                    if (f.key === "host" && arr[idx + 1]?.key === "port") {
                      const portField = arr[idx + 1];
                      acc.push(
                        <div key="host-port" className="grid grid-cols-[1fr_140px] gap-3">
                          <FormField label={f.label} required={f.required}>
                            <Input
                              placeholder={f.placeholder ?? "Enter Domain"}
                              value={creds[f.key] ?? ""}
                              onChange={(e) => setCreds({ ...creds, [f.key]: e.target.value })}
                              className="h-10 bg-background"
                            />
                          </FormField>
                          <FormField
                            label={portField.label}
                            required={portField.required}
                            tooltip={portField.tooltip ?? "Default ports vary by provider"}
                          >
                            <div className="relative">
                              <Input
                                type="number"
                                placeholder={portField.placeholder ?? "Port"}
                                value={creds[portField.key] ?? ""}
                                onChange={(e) =>
                                  setCreds({ ...creds, [portField.key]: e.target.value })
                                }
                                className="h-10 bg-background pr-9"
                              />
                              <HelpCircle className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                            </div>
                          </FormField>
                        </div>,
                      );
                      return acc;
                    }
                    if (f.key === "port" && arr[idx - 1]?.key === "host") return acc;
                    if (f.type === "select" && f.options?.length) {
                      acc.push(
                        <FormField key={f.key} label={f.label} required={f.required} tooltip={f.tooltip}>
                          <select
                            value={creds[f.key] ?? ""}
                            onChange={(e) => setCreds({ ...creds, [f.key]: e.target.value })}
                            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-[#1565EF]/30"
                          >
                            <option value="" disabled>
                              Select {f.label.toLowerCase()}
                            </option>
                            {f.options.map((opt) => (
                              <option key={opt.value} value={opt.value}>
                                {opt.label}
                              </option>
                            ))}
                          </select>
                        </FormField>,
                      );
                      return acc;
                    }
                    acc.push(
                      <FormField key={f.key} label={f.label} required={f.required} tooltip={f.tooltip}>
                        <Input
                          type={f.type === "number" ? "number" : "text"}
                          placeholder={f.placeholder ?? `Enter ${f.label}`}
                          value={creds[f.key] ?? ""}
                          onChange={(e) => setCreds({ ...creds, [f.key]: e.target.value })}
                          className="h-10 bg-background"
                        />
                      </FormField>,
                    );
                    return acc;
                  }, [])
                  .flatMap((node, idx, arr) => {
                    // For ClickHouse, insert Protocol + Authentication divider between Host/Port and Username
                    if (
                      providerMeta.id === "clickhouse" &&
                      (node as ReactElement).key === "host-port"
                    ) {
                      return [
                        node,
                        <FormField key="protocol" label="Protocol" required>
                          <div className="flex items-center gap-10 pt-1">
                            {[
                              { id: "http", label: "HTTP" },
                              { id: "https", label: "HTTPS" },
                            ].map((opt) => {
                              const selected = (creds.protocol ?? "http") === opt.id;
                              return (
                                <label key={opt.id} className="flex cursor-pointer items-center gap-2">
                                  <span
                                    className={cn(
                                      "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                                      selected ? "border-[#1565EF]" : "border-border",
                                    )}
                                  >
                                    {selected && <span className="h-2 w-2 rounded-full bg-[#1565EF]" />}
                                  </span>
                                  <input
                                    type="radio"
                                    name="ch-protocol"
                                    className="sr-only"
                                    checked={selected}
                                    onChange={() => setCreds({ ...creds, protocol: opt.id })}
                                  />
                                  <span className="text-sm text-foreground">{opt.label}</span>
                                </label>
                              );
                            })}
                          </div>
                        </FormField>,
                        <div key="auth-divider" className="relative py-1">
                          <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
                          <div className="relative mx-auto w-fit bg-background px-3 text-xs font-medium text-muted-foreground">
                            Authentication
                          </div>
                        </div>,
                      ];
                    }
                    return [node];
                  })}

                {providerMeta.id === "bigquery" && (
                  <FormField
                    label="Service Account Key (JSON)"
                    required
                    tooltip="GCP Console → IAM & Admin → Service Accounts → Keys → Add key → JSON"
                    error={
                      step === "auth-error" && !creds.service_account_json
                        ? "Upload a valid JSON key file."
                        : undefined
                    }
                  >
                    <label className="flex cursor-pointer flex-col gap-2 rounded-lg border border-dashed border-border bg-background px-4 py-3 transition hover:border-[#1565EF]">
                      <input
                        type="file"
                        accept=".json,application/json"
                        className="sr-only"
                        onChange={async (e) => {
                          const file = e.target.files?.[0];
                          if (!file) return;
                          try {
                            const text = await file.text();
                            JSON.parse(text);
                            setSaFileName(file.name);
                            setCreds((prev) => ({ ...prev, service_account_json: text }));
                          } catch {
                            toast.error("Invalid JSON file. Upload the key downloaded from GCP.");
                            setSaFileName("");
                            setCreds((prev) => ({ ...prev, service_account_json: "" }));
                          }
                        }}
                      />
                      <span className="inline-flex items-center gap-2 text-sm font-medium text-[#1565EF]">
                        <CloudUpload className="h-4 w-4" />
                        {saFileName ? "Replace JSON key" : "Upload JSON key file"}
                      </span>
                      {saFileName ? (
                        <span className="truncate text-xs text-muted-foreground">{saFileName}</span>
                      ) : (
                        <span className="text-xs text-muted-foreground">
                          Paste-free upload from GCP service account keys
                        </span>
                      )}
                    </label>
                  </FormField>
                )}

                {providerMeta.id !== "bigquery" && (
                <FormField label="Password" required error={step === "auth-error" ? "Incorrect password. Please try again." : undefined}>
                  <div className="relative">
                    <Input
                      type={showPw ? "text" : "password"}
                      placeholder="Enter Password"
                      value={creds.password ?? ""}
                      onChange={(e) => setCreds({ ...creds, password: e.target.value })}
                      className={cn(
                        "h-10 bg-background pr-10",
                        step === "auth-error" && "border-destructive focus-visible:ring-destructive/30",
                      )}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPw((v) => !v)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      aria-label="Toggle password visibility"
                    >
                      {showPw ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                    </button>
                  </div>
                </FormField>
                )}
              </div>
            )}

            {step === "testing" && (
              <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
                <Loader2 className="h-8 w-8 animate-spin text-[#1565EF]" />
                <div className="text-sm text-muted-foreground">
                  Authenticating with {providerMeta?.name}…
                </div>
              </div>
            )}

            {step === "files" && (
              <FilesStep
                files={remoteFiles.length ? remoteFiles : MOCK_FILES}
                selected={selectedFiles}
                setSelected={setSelectedFiles}
                uploads={uploads}
                setUploads={setUploads}
              />
            )}

            {step === "uploading" && (
              <div className="space-y-3 py-10">
                <div className="text-sm">Uploading {selectedFiles.size} file(s)…</div>
                <Progress value={uploadProgress} className="h-2" />
                <div className="text-xs text-muted-foreground">{Math.round(uploadProgress)}%</div>
              </div>
            )}

            {step === "metadata" && (
              <MetadataStep meta={meta} setMeta={setMeta} />
            )}

            {step === "preview" && providerMeta && (
              <PreviewStep
                provider={providerMeta}
                creds={creds}
                files={(remoteFiles.length ? remoteFiles : MOCK_FILES).filter((f) => selectedFiles.has(f.id))}
                uploads={uploads.filter((u) => u.status === "done")}
                meta={meta}
                saFileName={saFileName}
              />
            )}

            {step === "confirm" && (
              <div className="flex flex-col items-center gap-3 py-12 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100">
                  <Check className="h-7 w-7 text-emerald-700" strokeWidth={2.5} />
                </div>
                <div className="mt-2 text-[17px] font-semibold text-foreground">Successfully Connected</div>
                <div className="text-sm text-muted-foreground">
                  {meta.name || `${providerMeta?.name} Data browser`}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer (hidden on success step) */}
        {step !== "confirm" && (
        <div className="grid grid-cols-2 gap-3 border-t border-border bg-background px-5 py-4">
          <FooterButtons
            step={step}
            isPending={create.isPending || insertFiles.isPending}
            onCancel={requestClose}
            onContinueProvider={() => provider && setStep("credentials")}
            providerSelected={!!provider}
            onAuth={handleTestAuth}
            onContinueFiles={handleStartUpload}
            onConfirm={() => setShowConnectConfirm(true)}
            onDone={close}
            onMetadataContinue={() => setStep("preview")}
            onEditFromPreview={() => setStep("metadata")}
            metaValid={!!meta.name.trim()}
          />
        </div>
        )}
      </DialogContent>
    </Dialog>

    {/* Cancel confirmation */}
    <Dialog open={showCancelConfirm} onOpenChange={setShowCancelConfirm}>
      <DialogContent className="max-w-[400px] gap-0 overflow-hidden rounded-2xl border border-border p-0 text-center [&>button]:hidden">
        <button
          onClick={() => setShowCancelConfirm(false)}
          className="absolute right-4 top-4 rounded p-1 text-muted-foreground hover:bg-muted"
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </button>
        <div className="flex flex-col items-center px-6 pb-6 pt-8">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-red-50">
            <Cloud className="h-6 w-6 text-red-500" strokeWidth={1.75} />
          </div>
          <h3 className="mt-4 text-base font-semibold text-foreground">Are you you want to Cancel</h3>
          <p className="mt-1 text-sm text-muted-foreground">Cloud Connect</p>
          <div className="mt-6 grid w-full grid-cols-2 gap-3">
            <Button
              variant="outline"
              onClick={() => setShowCancelConfirm(false)}
              className="h-11 w-full justify-center rounded-lg border border-border bg-background text-sm font-semibold text-foreground hover:bg-muted"
            >
              Cancel
            </Button>
            <Button
              onClick={close}
              className="h-11 w-full justify-center rounded-lg bg-[#dc2626] text-sm font-semibold text-white hover:bg-[#b91d1d]"
            >
              Confirm
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>

    {/* Connect confirmation */}
    <Dialog open={showConnectConfirm} onOpenChange={setShowConnectConfirm}>
      <DialogContent className="max-w-[400px] gap-0 overflow-hidden rounded-2xl border border-border p-0 text-center [&>button]:hidden">
        <button
          onClick={() => setShowConnectConfirm(false)}
          className="absolute right-4 top-4 rounded p-1 text-muted-foreground hover:bg-muted"
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </button>
        <div className="flex flex-col items-center px-6 pb-6 pt-8">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-[#eef3ff] dark:bg-[#1565EF]/15">
            <HelpCircle className="h-6 w-6 text-[#1565EF]" strokeWidth={1.75} />
          </div>
          <h3 className="mt-4 text-base font-semibold text-foreground">Are you you want to Connect</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            {meta.name || `${providerMeta?.name ?? ""} Data browser`}
          </p>
          <div className="mt-6 grid w-full grid-cols-2 gap-3">
            <Button
              variant="outline"
              onClick={() => setShowConnectConfirm(false)}
              className="h-11 w-full justify-center rounded-lg border border-border bg-background text-sm font-semibold text-foreground hover:bg-muted"
            >
              Cancel
            </Button>
            <Button
              onClick={async () => {
                setShowConnectConfirm(false);
                await handleConfirmConnect();
              }}
              disabled={create.isPending || insertFiles.isPending}
              className="h-11 w-full justify-center rounded-lg bg-[#1565EF] text-sm font-semibold text-white hover:bg-[#1257cf]"
            >
              Confirm
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
    {storageConnectProvider && (
      <CloudStorageConnectionsListModal
        open={!!storageConnectProvider}
        provider={storageConnectProvider}
        onOpenChange={(o) => {
          if (!o) setStorageConnectProvider(null);
        }}
        onBrowse={(conn) => {
          setStorageConnectProvider(null);
          setStorageBrowseConnection(conn);
          onOpenChange(false);
        }}
      />
    )}
    <CloudStorageBrowserModal
      open={!!storageBrowseConnection}
      connection={storageBrowseConnection}
      onOpenChange={(o) => {
        if (!o) setStorageBrowseConnection(null);
      }}
    />
  </>);
}

function FooterButtons({
  step,
  isPending,
  onCancel,
  onContinueProvider,
  providerSelected,
  onAuth,
  onContinueFiles,
  onConfirm,
  onDone,
  onMetadataContinue,
  onEditFromPreview,
  metaValid,
}: {
  step: Step;
  isPending: boolean;
  onCancel: () => void;
  onContinueProvider: () => void;
  providerSelected: boolean;
  onAuth: () => void;
  onContinueFiles: () => void;
  onConfirm: () => void;
  onDone: () => void;
  onMetadataContinue: () => void;
  onEditFromPreview: () => void;
  metaValid: boolean;
}) {
  const outline = "h-11 w-full justify-center gap-2 rounded-lg border border-border bg-background text-sm font-semibold text-foreground hover:bg-muted";
  const primary = "h-11 w-full justify-center gap-2 rounded-lg bg-[#1565EF] text-sm font-semibold text-white hover:bg-[#1257cf]";
  const cancel = (
    <Button variant="outline" onClick={onCancel} className={outline}>
      <X className="h-4 w-4" /> Cancel
    </Button>
  );

  switch (step) {
    case "select-provider":
      return (
        <>
          {cancel}
          <Button onClick={onContinueProvider} disabled={!providerSelected} className={outline}>
            <Check className="h-4 w-4" /> Continue
          </Button>
        </>
      );
    case "credentials":
    case "auth-error":
      return (
        <>
          {cancel}
          <Button onClick={onAuth} className={primary}>
            {step === "auth-error" ? (
              <>
                <RefreshCw className="h-4 w-4" /> Retry
              </>
            ) : (
              <>
                <Check className="h-4 w-4" /> Authenticate
              </>
            )}
          </Button>
        </>
      );
    case "testing":
      return <>{cancel}<Button disabled className={primary}>Authenticating…</Button></>;
    case "files":
      return (
        <>
          {cancel}
          <Button onClick={onContinueFiles} className={primary}>
            <Check className="h-4 w-4" /> Continue
          </Button>
        </>
      );
    case "uploading":
      return <>{cancel}<Button disabled className={primary}>Uploading…</Button></>;
    case "metadata":
      return (
        <>
          {cancel}
          <Button onClick={onMetadataContinue} disabled={!metaValid} className={primary}>
            <Check className="h-4 w-4" /> Continue
          </Button>
        </>
      );
    case "preview":
      return (
        <>
          <Button onClick={onEditFromPreview} className={outline}>
            <SquarePen className="h-4 w-4" /> Edit
          </Button>
          <Button onClick={onConfirm} disabled={isPending} className={primary}>
            <Link2 className="h-4 w-4" /> {isPending ? "Connecting…" : "Connect"}
          </Button>
        </>
      );
    case "confirm":
      return (
        <>
          {cancel}
          <Button onClick={onDone} className={primary}>
            <Check className="h-4 w-4" /> Done
          </Button>
        </>
      );
  }
}

function FormField({
  label,
  required,
  tooltip,
  error,
  children,
}: {
  label: string;
  required?: boolean;
  tooltip?: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <Label className="flex items-center gap-1 text-xs font-medium text-foreground">
        {label}
        {required && <span className="text-destructive">*</span>}
        {tooltip && (
          <span title={tooltip} className="ml-0.5 inline-flex">
            <HelpCircle className="h-3 w-3 text-muted-foreground" />
          </span>
        )}
      </Label>
      {children}
      {error && <p className="text-[11px] font-medium text-destructive">{error}</p>}
    </div>
  );
}


function FileTypeBadge({ type }: { type: "CSV" | "XLSX" }) {
  return (
    <div className="relative flex h-10 w-9 shrink-0 items-center justify-center rounded-[3px] bg-[#16a34a] text-[9px] font-bold uppercase tracking-wide text-white shadow-sm">
      <span className="absolute right-0 top-0 h-2 w-2 rounded-bl-[3px] bg-white/30" />
      {type}
    </div>
  );
}

function FilesStep({
  files,
  selected,
  setSelected,
  uploads,
  setUploads,
}: {
  files: RemoteFile[];
  selected: Set<string>;
  setSelected: React.Dispatch<React.SetStateAction<Set<string>>>;
  uploads: UploadFile[];
  setUploads: React.Dispatch<React.SetStateAction<UploadFile[]>>;
}) {
  const [open, setOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const removeChip = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  };

  const startUpload = (name: string, type: "CSV" | "XLSX", sizeKb: number) => {
    const id = `up-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const willFail = /fail/i.test(name);
    setUploads((u) => [
      ...u,
      { id, name, type, size: `${sizeKb} KB`, progress: 0, status: "uploading" },
    ]);
    let p = 0;
    const t = setInterval(() => {
      p += 10 + Math.random() * 15;
      setUploads((u) =>
        u.map((f) =>
          f.id === id ? { ...f, progress: Math.min(99, p) } : f,
        ),
      );
      if (p >= 100) {
        clearInterval(t);
        setUploads((u) =>
          u.map((f) =>
            f.id === id
              ? { ...f, progress: willFail ? 70 : 100, status: willFail ? "failed" : "done" }
              : f,
          ),
        );
      }
    }, 220);
  };

  const onPick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv,.xlsx";
    input.multiple = true;
    input.onchange = () => {
      Array.from(input.files ?? []).forEach((f) => {
        const ext = f.name.toLowerCase().endsWith(".xlsx") ? "XLSX" : "CSV";
        startUpload(f.name, ext as "CSV" | "XLSX", Math.max(1, Math.round(f.size / 1024)));
      });
    };
    input.click();
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    Array.from(e.dataTransfer.files).forEach((f) => {
      const ext = f.name.toLowerCase().endsWith(".xlsx") ? "XLSX" : "CSV";
      startUpload(f.name, ext as "CSV" | "XLSX", Math.max(1, Math.round(f.size / 1024)));
    });
  };

  const selectedFiles = files.filter((f) => selected.has(f.id));

  return (
    <div className="space-y-4">
      <div className="grid gap-1.5">
        <Label className="flex items-center gap-1 text-xs font-medium text-foreground">
          Files
          <span className="text-destructive">*</span>
          <HelpCircle className="ml-0.5 h-3 w-3 text-muted-foreground" />
        </Label>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              className={cn(
                "flex min-h-10 w-full items-center justify-between gap-2 rounded-md border bg-background px-3 py-1.5 text-left text-sm",
                open ? "border-[#1565EF] ring-1 ring-[#1565EF]/30" : "border-border",
              )}
            >
              <div className="flex flex-1 flex-wrap items-center gap-1.5">
                {selectedFiles.length === 0 ? (
                  <span className="text-muted-foreground">Choose File</span>
                ) : (
                  selectedFiles.map((f) => (
                    <span
                      key={f.id}
                      className="inline-flex items-center gap-1.5 rounded border border-border bg-background px-1.5 py-0.5 text-xs"
                    >
                      <span className="flex h-4 w-3.5 items-center justify-center rounded-[2px] bg-[#16a34a] text-[7px] font-bold text-white">
                        {f.type === "XLSX" ? "X" : "C"}
                      </span>
                      {f.name}
                      <X
                        className="h-3 w-3 cursor-pointer text-muted-foreground hover:text-foreground"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeChip(f.id);
                        }}
                      />
                    </span>
                  ))
                )}
              </div>
              <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition", open && "rotate-180")} />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            sideOffset={6}
            className="w-[var(--radix-popover-trigger-width)] p-1.5"
          >
            <div className="max-h-64 space-y-0.5 overflow-y-auto">
              {files.map((f) => {
                const checked = selected.has(f.id);
                return (
                  <button
                    type="button"
                    key={f.id}
                    onClick={() => toggle(f.id)}
                    aria-pressed={checked}
                    className="flex w-full cursor-pointer items-center gap-3 rounded-md p-2 text-left outline-none hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-[#1565EF]/30"
                  >
                    <FileTypeBadge type={f.type} />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-foreground">{f.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {f.size} <span className="mx-1">•</span> {f.columns} Columns, {f.rows} rows
                      </div>
                    </div>
                    <span
                      aria-hidden="true"
                      className={cn(
                        "grid h-4 w-4 shrink-0 place-items-center rounded-sm border shadow-sm transition-colors",
                        checked ? "border-[#1565EF] bg-[#1565EF] text-white" : "border-border bg-background text-transparent",
                      )}
                    >
                      <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
                    </span>
                  </button>
                );
              })}
            </div>
          </PopoverContent>
        </Popover>
      </div>

      <div className="flex items-center justify-center text-xs text-muted-foreground">Or</div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={onPick}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border bg-background px-4 py-6 text-center transition",
          dragOver ? "border-[#1565EF] ring-2 ring-[#1565EF]/30" : "border-border hover:border-[#1565EF]",
        )}
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-md border border-border bg-background dark:bg-white">
          <CloudUpload className="h-5 w-5 text-foreground" strokeWidth={1.75} />
        </div>
        <div className="text-sm">
          <span className="font-semibold text-[#1565EF]">Select</span>{" "}
          <span className="text-foreground">or drag and drop file</span>
        </div>
        <div className="text-xs text-muted-foreground">.CSV or .xlsx</div>
      </div>

      {uploads.length > 0 && (
        <div className="space-y-2">
          {uploads.map((u) => (
            <div
              key={u.id}
              className={cn(
                "rounded-lg border bg-background p-3",
                u.status === "failed" ? "border-[#dc2626]" : "border-border",
              )}
            >
              <div className="flex items-start gap-3">
                <FileTypeBadge type={u.type} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-foreground">{u.name}</div>
                  {u.status === "failed" ? (
                    <>
                      <div className="text-xs text-muted-foreground">Upload failed, please try again.</div>
                      <button
                        type="button"
                        onClick={() => {
                          setUploads((list) => list.filter((x) => x.id !== u.id));
                          startUpload(u.name.replace(/fail/i, "retry"), u.type, 200);
                        }}
                        className="mt-0.5 text-xs font-semibold text-[#dc2626] hover:underline"
                      >
                        Try again
                      </button>
                    </>
                  ) : (
                    <div className="text-xs text-muted-foreground">
                      {u.size} <span className="mx-1">|</span>{" "}
                      <ArrowUp className="-mt-0.5 inline h-3 w-3" /> {Math.round(u.progress)}%
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setUploads((list) => list.filter((x) => x.id !== u.id))}
                  className="text-muted-foreground hover:text-foreground"
                  aria-label="Remove"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const DOMAIN_OPTIONS = ["Finance", "Sales", "Marketing", "Product", "Operations", "Human Resource"];

function MetadataStep({
  meta,
  setMeta,
}: {
  meta: { name: string; description: string; domain: string; use_cases: string };
  setMeta: React.Dispatch<
    React.SetStateAction<{ name: string; description: string; domain: string; use_cases: string }>
  >;
}) {
  const [domainOpen, setDomainOpen] = useState(false);
  const [tagInput, setTagInput] = useState("");
  const tags = meta.use_cases
    ? meta.use_cases.split(",").map((t) => t.trim()).filter(Boolean)
    : [];

  const setTags = (next: string[]) =>
    setMeta((m) => ({ ...m, use_cases: next.join(", ") }));

  const addTag = (raw: string) => {
    const t = raw.trim().replace(/,$/, "").trim();
    if (!t || tags.includes(t)) return;
    setTags([...tags, t]);
  };

  return (
    <div className="space-y-4">
      <FormField label="Dataset Name" required>
        <Input
          placeholder="Doc 01"
          className="h-10 bg-background"
          value={meta.name}
          onChange={(e) => setMeta((m) => ({ ...m, name: e.target.value }))}
        />
      </FormField>

      <FormField label="Description">
        <Textarea
          rows={4}
          placeholder="Enter a description..."
          className="resize-none bg-background"
          value={meta.description}
          onChange={(e) => setMeta((m) => ({ ...m, description: e.target.value }))}
        />
      </FormField>

      <FormField label="Domain" required>
        <Popover open={domainOpen} onOpenChange={setDomainOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              className={cn(
                "flex h-10 w-full items-center gap-2 rounded-md border bg-background px-3 text-left text-sm",
                domainOpen ? "border-[#1565EF] ring-1 ring-[#1565EF]/30" : "border-border",
              )}
            >
              <Briefcase className="h-4 w-4 text-muted-foreground" strokeWidth={1.75} />
              <span className={cn("flex-1", !meta.domain && "text-muted-foreground")}>
                {meta.domain || "Select a domain"}
              </span>
              <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition", domainOpen && "rotate-180")} />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            sideOffset={6}
            className="w-[var(--radix-popover-trigger-width)] p-1"
          >
            {DOMAIN_OPTIONS.map((d) => (
              <button
                type="button"
                key={d}
                onClick={() => {
                  setMeta((m) => ({ ...m, domain: d }));
                  setDomainOpen(false);
                }}
                className={cn(
                  "flex w-full items-center gap-2 rounded p-2 text-left text-sm hover:bg-muted/60",
                  meta.domain === d && "bg-muted/60 font-medium",
                )}
              >
                <Briefcase className="h-4 w-4 text-muted-foreground" strokeWidth={1.75} />
                {d}
              </button>
            ))}
          </PopoverContent>
        </Popover>
      </FormField>

      <FormField label="Use Case" tooltip="Add the use cases this dataset supports">
        <div className="flex min-h-10 flex-wrap items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5">
          <Search className="h-4 w-4 text-muted-foreground" />
          {tags.map((t) => (
            <span
              key={t}
              className="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-xs"
            >
              {t}
              <X
                className="h-3 w-3 cursor-pointer text-muted-foreground hover:text-foreground"
                onClick={() => setTags(tags.filter((x) => x !== t))}
              />
            </span>
          ))}
          <input
            value={tagInput}
            onChange={(e) => {
              const v = e.target.value;
              if (v.endsWith(",")) {
                addTag(v);
                setTagInput("");
              } else setTagInput(v);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && tagInput.trim()) {
                e.preventDefault();
                addTag(tagInput);
                setTagInput("");
              } else if (e.key === "Backspace" && !tagInput && tags.length) {
                setTags(tags.slice(0, -1));
              }
            }}
            placeholder={tags.length === 0 ? "Add a use case" : ""}
            className="flex-1 min-w-[80px] border-0 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
        </div>
      </FormField>

      <div className="relative pt-1">
        <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
        <div className="relative mx-auto w-fit bg-background px-3 text-xs font-medium text-muted-foreground">
          Refresh Settings
        </div>
      </div>
    </div>
  );
}

/* ---------------- Preview Step ---------------- */
import type { ProviderMeta } from "./providers";

function PreviewStep({
  provider,
  creds,
  files,
  uploads,
  meta,
  saFileName = "",
}: {
  provider: ProviderMeta;
  creds: Record<string, string>;
  files: RemoteFile[];
  uploads: UploadFile[];
  meta: { name: string; description: string; domain: string; use_cases: string };
  saFileName?: string;
}) {
  const allFiles: Array<{ id: string; name: string; type: "CSV" | "XLSX" }> = [
    ...files.map((f) => ({ id: f.id, name: f.name, type: f.type })),
    ...uploads.map((u) => ({ id: u.id, name: u.name.replace(/\.(csv|xlsx)$/i, ""), type: u.type })),
  ];
  const visible = allFiles.slice(0, 2);
  const more = allFiles.length - visible.length;

  // Build the credential rows we want to render as read-only summary.
  const cnName = creds.connectionName || `${provider.name}_Connection`;
  const password = creds.password ?? "";
  const passwordDots = "•".repeat(Math.max(8, Math.min(password.length || 12, 14)));

  return (
    <div className="space-y-5">
      {/* Data Source */}
      <SectionDivider label={null}>
        <div className="space-y-1">
          <div className="text-xs text-muted-foreground">Data Source</div>
          <div className="text-sm font-semibold text-foreground">{provider.name}</div>
        </div>
      </SectionDivider>

      {/* Authenticate & Connect */}
      <div className="space-y-3">
        <DividerLabel>Authenticate &amp; Connect</DividerLabel>
        {provider.id === "clickhouse" && (
          <div className="-mb-1">
            <span className="inline-flex items-center gap-1 rounded-md bg-[#1565EF] px-2 py-0.5 text-[11px] font-medium text-white">
              <svg width="10" height="10" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="3" width="12" height="3" rx="0.5" stroke="currentColor"/><rect x="2" y="10" width="12" height="3" rx="0.5" stroke="currentColor"/></svg>
              Container
            </span>
          </div>
        )}
        <div className={cn(
          "rounded-lg border bg-background p-4",
          provider.id === "clickhouse" ? "border-[#1565EF]" : "border-border",
        )}>
          <SummaryField label="Connection Name" value={cnName} />
          <AuthBody provider={provider} creds={creds} passwordDots={passwordDots} saFileName={saFileName} />
        </div>
      </div>


      {/* Data */}
      <div className="space-y-3">
        <DividerLabel>Data</DividerLabel>
        <div className="flex flex-wrap items-center gap-2">
          {visible.length === 0 ? (
            <span className="text-xs text-muted-foreground">No files selected</span>
          ) : (
            visible.map((f) => (
              <div
                key={f.id}
                className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-2.5 py-1.5"
              >
                <span className="flex h-5 w-5 items-center justify-center rounded-[3px] bg-[#16a34a] text-[8px] font-bold text-white">
                  {f.type === "XLSX" ? "X" : "C"}
                </span>
                <span className="max-w-[120px] truncate text-xs font-medium text-foreground">
                  {f.name.length > 10 ? `${f.name.slice(0, 10)}...` : f.name}
                </span>
              </div>
            ))
          )}
          {more > 0 && (
            <span className="text-xs font-medium text-muted-foreground">{more} More</span>
          )}
        </div>
      </div>

      {/* Dataset Basics */}
      <div className="space-y-3">
        <DividerLabel>Dataset Basics</DividerLabel>
        <div className="rounded-lg border border-border bg-background p-4">
          <SummaryField label="Dataset Name" value={meta.name || "—"} />
          {meta.description && (
            <div className="mt-3">
              <SummaryField label="Description" value={meta.description} />
            </div>
          )}
          {meta.domain && (
            <div className="mt-3">
              <SummaryField label="Domain" value={meta.domain} />
            </div>
          )}
          {meta.use_cases && (
            <div className="mt-3">
              <SummaryField label="Use Case" value={meta.use_cases} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DividerLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative">
      <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
      <div className="relative mx-auto w-fit bg-background px-3 text-xs font-medium text-muted-foreground">
        {children}
      </div>
    </div>
  );
}

function SectionDivider({
  label,
  children,
}: {
  label: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      {label && <DividerLabel>{label}</DividerLabel>}
      {children}
    </div>
  );
}

function SummaryField({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-0.5">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold text-foreground break-words">{value}</div>
    </div>
  );
}

function SslToggleRow({
  title,
  desc,
  on: defaultOn = true,
}: {
  title: string;
  desc: string;
  on?: boolean;
}) {
  const [on, setOn] = useState(defaultOn);
  return (
    <div className="mt-4 flex items-start justify-between gap-4 border-t border-border pt-4">
      <div className="space-y-0.5">
        <div className="text-sm font-medium text-foreground">{title}</div>
        <div className="text-xs text-muted-foreground">{desc}</div>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={on}
        onClick={() => setOn((v) => !v)}
        className={cn(
          "relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full transition",
          on ? "bg-[#1565EF]" : "bg-muted",
        )}
      >
        <span
          className={cn(
            "inline-block h-4 w-4 transform rounded-full bg-background shadow transition",
            on ? "translate-x-[18px]" : "translate-x-0.5",
          )}
        />
      </button>
    </div>
  );
}

function SslFileRow({ filename = "Tech design requirements.pdf" }: { filename?: string }) {
  return (
    <div className="mt-4 space-y-1.5">
      <div className="text-xs text-muted-foreground">SSL</div>
      <div className="flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-[3px] bg-[#dc2626] text-[8px] font-bold text-white">
          PDF
        </span>
        <span className="truncate text-sm text-foreground">{filename}</span>
      </div>
    </div>
  );
}

function AuthBody({
  provider,
  creds,
  passwordDots,
  saFileName = "",
}: {
  provider: ProviderMeta;
  creds: Record<string, string>;
  passwordDots: string;
  saFileName?: string;
}) {
  const userPass = (
    <>
      {creds.username && (
        <div className="mt-3">
          <SummaryField label="Username" value={creds.username} />
        </div>
      )}
      <div className="mt-3">
        <SummaryField label="Password" value={passwordDots} />
      </div>
    </>
  );

  // Snowflake: Host (account, full width), user/pass, Warehouse+Role grid, Database
  if (provider.id === "snowflake") {
    return (
      <>
        <div className="mt-3">
          <SummaryField label="Host" value={creds.account || creds.host || "—"} />
        </div>
        {userPass}
        {(creds.warehouse || creds.role) && (
          <div className="mt-3 grid grid-cols-2 gap-4">
            <SummaryField label="Warehouse" value={creds.warehouse || "—"} />
            <SummaryField label="Role" value={creds.role || "—"} />
          </div>
        )}
        {creds.database && (
          <div className="mt-3">
            <SummaryField label="Database" value={creds.database} />
          </div>
        )}
      </>
    );
  }

  // BigQuery / GCP: project, region, dataset path, JSON key (no password)
  if (provider.id === "bigquery") {
    const regionLabel =
      GCP_REGION_OPTIONS.find((o) => o.value === creds.region)?.label ?? creds.region ?? "—";
    return (
      <>
        <div className="mt-3 space-y-3">
          <SummaryField label="Project ID" value={creds.project_id || "—"} />
          <SummaryField label="Region" value={regionLabel} />
          <SummaryField label="Bucket" value={creds.bucket_name || "—"} />
          <SummaryField label="Folder" value={creds.folder_path || "—"} />
          <SummaryField
            label="Service Account Key"
            value={saFileName || (creds.service_account_json ? "JSON key uploaded" : "—")}
          />
        </div>
      </>
    );
  }

  // Databricks: workspace URL + http path + token
  if (provider.id === "databricks") {
    return (
      <>
        {creds.host && (
          <div className="mt-3">
            <SummaryField label="Workspace URL" value={creds.host} />
          </div>
        )}
        {creds.http_path && (
          <div className="mt-3">
            <SummaryField label="HTTP Path" value={creds.http_path} />
          </div>
        )}
        <div className="mt-3">
          <SummaryField label="Access Token" value={passwordDots} />
        </div>
      </>
    );
  }

  // MongoDB: single Host (no port)
  if (provider.id === "mongodb") {
    return (
      <>
        {creds.host && (
          <div className="mt-3">
            <SummaryField label="Host" value={creds.host} />
          </div>
        )}
        {userPass}
      </>
    );
  }

  // Default: Host + Port grid, user/pass, then provider-specific extras
  return (
    <>
      {(creds.host || creds.port) && (
        <div className="mt-3 grid grid-cols-2 gap-4">
          <SummaryField label="Host" value={creds.host || "—"} />
          <SummaryField label="Port" value={creds.port || "—"} />
        </div>
      )}
      {userPass}
      <ProviderExtras provider={provider} creds={creds} />
    </>
  );
}

function ProviderExtras({
  provider,
  creds,
}: {
  provider: ProviderMeta;
  creds: Record<string, string>;
}) {
  switch (provider.id) {
    case "postgres":
    case "mariadb":
      return <SslFileRow filename={creds.sslFile || "Tech design requirements.pdf"} />;
    case "mssql":
      return (
        <>
          <SslToggleRow
            title="Encrypt Connection"
            desc="Encrypts traffic between Avaloka and SQL Server."
          />
          <SslToggleRow
            title="Trust Server Certificate (Optional)"
            desc="Allows SSL without CA validation (useful in dev/test; not ideal for production)"
          />
        </>
      );
    case "clickhouse":
      return (
        <SslToggleRow
          title="SSL Mode"
          desc="Ensures encrypted communication with the ClickHouse server."
        />
      );
    default:
      return null;
  }
}







