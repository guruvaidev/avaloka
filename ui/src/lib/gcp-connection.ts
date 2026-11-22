import { supabase } from "@/integrations/supabase/client";

export type GcsPath = { bucket: string; prefix: string };

/** Parse `bucket/folder/subfolder` from a combined GCS path. */
export function parseGcsDatasetPath(raw: string, projectId?: string): GcsPath {
  let trimmed = raw.trim().replace(/^gs:\/\//, "").replace(/\/+$/, "");
  const pid = projectId?.trim();
  if (pid) {
    const withSlash = `${pid}/`;
    if (trimmed.startsWith(withSlash)) trimmed = trimmed.slice(withSlash.length);
  }
  const parts = trimmed.split("/").filter(Boolean);
  if (!parts.length) {
    throw new Error("Dataset path is required (e.g. avaloka-test-user-filestore/test-input-data).");
  }
  return {
    bucket: parts[0],
    prefix: parts.slice(1).join("/"),
  };
}

/** Validate bucket-only input (no slashes, no embedded project id). */
export function validateGcsBucketName(bucket: string, projectId?: string): string {
  const name = bucket.trim().replace(/^gs:\/\//, "").split("/")[0]?.trim() ?? "";
  if (!name) throw new Error("Bucket name is required.");
  if (bucket.includes("/")) {
    throw new Error("Enter only the bucket name — use Folder path for subfolders.");
  }
  const pid = projectId?.trim();
  if (pid && name.includes(pid)) {
    throw new Error(
      `Bucket name must not include the Project ID. Example: avaloka-test-user-filestore`,
    );
  }
  if (pid && /playground-\d+/i.test(name) && name.includes("avaloka")) {
    throw new Error(
      "Bucket name looks merged with Project ID. Enter only avaloka-test-user-filestore (see Cloud Storage → Buckets → Name).",
    );
  }
  return name;
}

export function buildGcsDatasetPath(bucket: string, folderPath: string): string {
  const b = validateGcsBucketName(bucket);
  const folder = folderPath.trim().replace(/^\/+|\/+$/g, "");
  return folder ? `${b}/${folder}` : b;
}

/** `gs://bucket/prefix` from a stored `bucket_name` path (bucket/folder/...). */
export function buildGcsStorageUri(bucketPath: string): string {
  const path = bucketPath.trim().replace(/^gs:\/\//, "").replace(/\/+$/, "");
  if (!path) throw new Error("Missing GCS bucket path.");
  return `gs://${path}`;
}

export type GcpCloudConnection = {
  id: string;
  name: string;
  provider: string;
  bucket_name: string | null;
};

async function userId(): Promise<string> {
  const { data } = await supabase.auth.getUser();
  const id = data.user?.id;
  if (!id) throw new Error("Not signed in");
  return id;
}

/**
 * Persist a GCP connection. The service-account key goes to the backend,
 * which seals it (AES-GCM under DB_ENCRYPTION_KEY) and writes only the
 * *_ciphertext / *_iv columns: `cloud_datasets` refuses plaintext keys since
 * migration 20261116130000. The backend reads them via get_cloud_connection
 * (cloud_connections.py), as before.
 */
export async function createGcpCloudConnection(input: {
  name: string;
  projectId?: string;
  region: string;
  datasetPath: string;
  serviceAccountJson: string;
}): Promise<GcpCloudConnection> {
  await userId(); // signed in, or the backend call cannot be made
  const projectId = input.projectId?.trim() ?? "";
  const region = input.region.trim();
  const datasetPath = input.datasetPath.trim().replace(/\/+$/, "");
  const saJson = input.serviceAccountJson.trim();

  if (!region) throw new Error("Region is required.");
  if (!datasetPath) throw new Error("Dataset path is required.");
  if (!saJson) throw new Error("Service account JSON key is required.");

  const { backendApi } = await import("@/lib/api/backendApi");
  const data = await backendApi.createCloudConnection({
    name: input.name.trim() || "GCP connection",
    provider: "gcp",
    region,
    project_id: projectId || undefined,
    bucket_name: datasetPath,
    secret_key: saJson,
  });
  return { ...data, bucket_name: data.bucket_name ?? datasetPath } as GcpCloudConnection;
}

/** Mark an existing GCP cloud_datasets row as connected after the wizard confirm step. */
export async function finalizeGcpCloudConnection(input: {
  connectionId: string;
  name: string;
  description?: string | null;
  domain?: string | null;
  use_cases?: string | null;
  selectedFiles?: string[];
}): Promise<GcpCloudConnection> {
  const meta = {
    description: input.description ?? null,
    domain: input.domain ?? null,
    use_cases: input.use_cases ?? null,
    selected_files: input.selectedFiles ?? [],
  };

  const { data, error } = await supabase
    .from("cloud_datasets")
    .update({
      name: input.name.trim() || "GCP connection",
      connection_status: "connected",
      last_tested_at: new Date().toISOString(),
      schema: meta,
    })
    .eq("id", input.connectionId)
    .select("id,name,provider,bucket_name")
    .single();

  if (error) throw new Error(error.message);
  return data as GcpCloudConnection;
}
