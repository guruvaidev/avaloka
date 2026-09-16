import { useMemo, useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { AlertTriangle, Cloud, Loader2, X } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { backendApi } from "@/lib/api/backendApi";
import { PROVIDER_LOGOS } from "./logos";

export type StorageProvider = "aws" | "azure" | "gcp";

export type StorageConnection = {
  id: string;
  name: string;
  provider: StorageProvider;
  bucket_name: string;
};

const AWS_REGIONS = [
  "us-east-1",
  "us-east-2",
  "us-west-1",
  "us-west-2",
  "eu-west-1",
  "eu-central-1",
  "ap-south-1",
  "ap-southeast-1",
  "ap-southeast-2",
];
const AZURE_REGIONS = [
  "eastus",
  "eastus2",
  "westus",
  "westus2",
  "westeurope",
  "northeurope",
  "centralindia",
  "southeastasia",
];
const GCP_REGIONS = [
  "us",
  "us-central1",
  "us-east1",
  "us-west1",
  "europe-west1",
  "europe-west2",
  "asia-south1",
  "asia-southeast1",
];

const PROVIDER_META: Record<
  StorageProvider,
  {
    label: string;
    logoKey: keyof typeof PROVIDER_LOGOS | string;
    regions: string[];
    bucketLabel: string;
  }
> = {
  aws: { label: "Amazon S3", logoKey: "s3", regions: AWS_REGIONS, bucketLabel: "Bucket Name" },
  azure: { label: "Azure Blob Storage", logoKey: "azure", regions: AZURE_REGIONS, bucketLabel: "Container Name" },
  gcp: { label: "Google Cloud Storage", logoKey: "gcs", regions: GCP_REGIONS, bucketLabel: "Bucket Name" },
};

type State = {
  name: string;
  region: string;
  endpoint_url: string;
  project_id: string;
  access_key: string;
  secret_key: string;
  bucket_name: string;
};

const emptyState = (provider: StorageProvider): State => ({
  name: "",
  region: PROVIDER_META[provider].regions[0] ?? "",
  endpoint_url: "",
  project_id: "",
  access_key: "",
  secret_key: "",
  bucket_name: "",
});

export function CloudStorageConnectModal({
  open,
  provider,
  onOpenChange,
  onConnected,
}: {
  open: boolean;
  provider: StorageProvider;
  onOpenChange: (open: boolean) => void;
  onConnected: (connection: StorageConnection) => void;
}) {
  const [form, setForm] = useState<State>(() => emptyState(provider));
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string>("");

  const meta = PROVIDER_META[provider];
  const logo = (PROVIDER_LOGOS as Record<string, string>)[meta.logoKey as string];

  const set = <K extends keyof State>(k: K, v: State[K]) => setForm((s) => ({ ...s, [k]: v }));

  const canSubmit = useMemo(() => {
    if (!form.name.trim() || !form.bucket_name.trim() || !form.secret_key.trim()) return false;
    if (provider === "aws" && !form.access_key.trim()) return false;
    if (provider === "azure" && !form.access_key.trim()) return false;
    return true;
  }, [form, provider]);

  const close = () => {
    if (testing) return;
    onOpenChange(false);
    setTimeout(() => {
      setForm(emptyState(provider));
      setError("");
    }, 200);
  };

  const handleSubmit = async () => {
    setError("");
    if (!canSubmit) {
      setError("Please fill required fields.");
      return;
    }
    if (provider === "gcp") {
      try {
        JSON.parse(form.secret_key);
      } catch {
        setError("Service account key must be valid JSON.");
        return;
      }
    }
    setTesting(true);
    try {
      const payload = {
        name: form.name.trim(),
        provider,
        region: form.region || undefined,
        endpoint_url: form.endpoint_url.trim() || undefined,
        project_id: provider === "gcp" ? form.project_id.trim() || undefined : undefined,
        access_key:
          provider === "aws" || provider === "azure"
            ? form.access_key.trim()
            : undefined,
        secret_key: form.secret_key,
        bucket_name: form.bucket_name.trim(),
      };
      const conn = await backendApi.createCloudConnection(payload);
      toast.success("Connection saved");
      onConnected({
        id: conn.id,
        name: conn.name || payload.name,
        provider,
        bucket_name: conn.bucket_name || payload.bucket_name,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save connection");
    } finally {
      setTesting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => (o ? onOpenChange(true) : close())}>
      <DialogContent className="max-w-[560px] gap-0 overflow-hidden rounded-xl border border-border p-0 [&>button]:hidden">
        <div className="flex items-center justify-between border-b border-border bg-background px-5 py-4">
          <div className="flex items-center gap-2.5">
            <Cloud className="h-5 w-5 text-foreground" strokeWidth={1.75} />
            <h2 className="text-[17px] font-semibold tracking-tight">
              Add New Cloud Storage Connection
            </h2>
          </div>
          <button
            onClick={close}
            className="rounded p-1 text-muted-foreground hover:bg-muted"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="bg-background px-6 pb-4 pt-5">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-foreground">
                Configure your cloud storage connection parameters
              </div>
              <div className="text-xs text-muted-foreground">
                Provider is fixed to <span className="font-medium">{meta.label}</span>
              </div>
            </div>
            <span className="inline-flex items-center rounded-md border border-border bg-background dark:bg-white px-2 py-1">
              {logo ? (
                <img src={logo} alt={meta.label} className="h-4 w-auto object-contain" />
              ) : (
                <span className="text-xs font-medium">{meta.label}</span>
              )}
            </span>
          </div>

          {error && (
            <div className="mb-3 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
              <p className="text-xs text-amber-800">{error}</p>
            </div>
          )}

          <div className="grid max-h-[60vh] gap-3 overflow-y-auto pr-1">
            <Field label="Connection Name" required helper="Display name for this connection">
              <Input
                value={form.name}
                onChange={(e) => set("name", e.target.value)}
                placeholder="My cloud connection"
                className="h-10 bg-background"
              />
            </Field>

            <Field label="Region">
              <select
                value={form.region}
                onChange={(e) => set("region", e.target.value)}
                className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              >
                {meta.regions.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label="Custom Endpoint URL"
              helper="Leave empty to use default endpoint for the selected provider"
            >
              <Input
                value={form.endpoint_url}
                onChange={(e) => set("endpoint_url", e.target.value)}
                placeholder="https://…"
                className="h-10 bg-background"
              />
            </Field>

            {provider === "gcp" && (
              <>
                <Field label="Project ID (Optional)">
                  <Input
                    value={form.project_id}
                    onChange={(e) => set("project_id", e.target.value)}
                    placeholder="my-project-id"
                    className="h-10 bg-background"
                  />
                </Field>
                <Field
                  label="Secret Key"
                  required
                  helper="Paste entire JSON service account key"
                  subHelper="Download from: GCP Console → IAM & Admin → Service Accounts → Keys"
                >
                  <Textarea
                    value={form.secret_key}
                    onChange={(e) => set("secret_key", e.target.value)}
                    placeholder='{"type":"service_account","project_id":"..."}'
                    className="min-h-[120px] bg-background font-mono text-xs"
                  />
                </Field>
                <Field
                  label={meta.bucketLabel}
                  required
                  helper="Must match the exact bucket/container name from your cloud provider"
                >
                  <Input
                    value={form.bucket_name}
                    onChange={(e) => set("bucket_name", e.target.value)}
                    className="h-10 bg-background"
                  />
                </Field>
              </>
            )}

            {provider === "aws" && (
              <>
                <Field
                  label="Access Key"
                  required
                  helper="Found in: AWS Console → IAM → Users → Security credentials → Access keys"
                >
                  <Input
                    value={form.access_key}
                    onChange={(e) => set("access_key", e.target.value)}
                    className="h-10 bg-background"
                  />
                </Field>
                <Field
                  label="Secret Key"
                  required
                  helper="Generated when creating Access Key. Store securely - it's only shown once!"
                >
                  <Input
                    type="password"
                    value={form.secret_key}
                    onChange={(e) => set("secret_key", e.target.value)}
                    className="h-10 bg-background"
                  />
                </Field>
                <Field label={meta.bucketLabel} required>
                  <Input
                    value={form.bucket_name}
                    onChange={(e) => set("bucket_name", e.target.value)}
                    className="h-10 bg-background"
                  />
                </Field>
              </>
            )}

            {provider === "azure" && (
              <>
                <Field
                  label="Storage Account Name"
                  required
                  helper="Found in: Azure Portal → Storage Account → Overview"
                >
                  <Input
                    value={form.access_key}
                    onChange={(e) => set("access_key", e.target.value)}
                    className="h-10 bg-background"
                  />
                </Field>
                <Field
                  label="Account Key or SAS Token"
                  required
                  helper="Option 1 - SAS Token: Azure Portal → Storage Account → Security + networking → Shared access signature"
                  subHelper="Option 2 - Account Key: Azure Portal → Storage Account → Security + networking → Access keys. SAS Tokens are recommended for better security (time-limited, scoped permissions)."
                >
                  <Input
                    type="password"
                    value={form.secret_key}
                    onChange={(e) => set("secret_key", e.target.value)}
                    className="h-10 bg-background"
                  />
                </Field>
                <Field
                  label={meta.bucketLabel}
                  required
                  helper="The name of your Azure Blob Storage container"
                >
                  <Input
                    value={form.bucket_name}
                    onChange={(e) => set("bucket_name", e.target.value)}
                    className="h-10 bg-background"
                  />
                </Field>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-border bg-muted/30 px-5 py-3">
          <Button variant="ghost" onClick={close} disabled={testing}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!canSubmit || testing}
            className={cn("bg-[#1565EF] text-white hover:bg-[#1257d6]")}
          >
            {testing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Testing…
              </>
            ) : (
              "Test & Save Connection"
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  required,
  helper,
  subHelper,
  children,
}: {
  label: string;
  required?: boolean;
  helper?: string;
  subHelper?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-xs font-medium">
        {label} {required && <span className="text-red-500">*</span>}
      </Label>
      {children}
      {helper && <p className="text-[11px] text-muted-foreground">{helper}</p>}
      {subHelper && <p className="text-[11px] text-muted-foreground">{subHelper}</p>}
    </div>
  );
}
