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
 * Persist GCP credentials to Supabase `cloud_datasets`.
 * Matches the existing production schema + `check_credentials_format`:
 *   - access_key  → GCP Project ID
 *   - secret_key  → service account JSON (full key file contents)
 * Backend reads these via get_cloud_connection (cloud_connections.py).
 */
export async function createGcpCloudConnection(input: {
  name: string;
  projectId?: string;
  region: string;
  datasetPath: string;
  serviceAccountJson: string;
}): Promise<GcpCloudConnection> {
  const uid = await userId();
  const projectId = input.projectId?.trim() ?? "";
  const region = input.region.trim();
  const datasetPath = input.datasetPath.trim().replace(/\/+$/, "");
  let saJson = input.serviceAccountJson.trim();

  if (!region) throw new Error("Region is required.");
  if (!datasetPath) throw new Error("Dataset path is required.");
  if (!saJson) throw new Error("Service account JSON key is required.");

  try {
    const parsed = JSON.parse(saJson) as Record<string, unknown>;
    if (projectId && !parsed.project_id) parsed.project_id = projectId;
    saJson = JSON.stringify(parsed);
  } catch {
    throw new Error("Service account key must be valid JSON.");
  }

  const { data, error } = await supabase
    .from("cloud_datasets")
    .insert({
      user_id: uid,
      name: input.name.trim() || "GCP connection",
      provider: "gcp",
      region,
      bucket_name: datasetPath,
      access_key: projectId,
      secret_key: saJson,
    })
    .select("id,name,provider,bucket_name")
    .single();

  if (error) throw new Error(error.message);

  // Do not call encrypt_cloud_credentials here: that RPC clears access_key/secret_key
  // and stores values in *_ciphertext without IVs. The backend reads plaintext secret_key
  // (or decrypts ciphertext+iv when DB_ENCRYPTION_KEY is set). Leaving plaintext lets
  // /buckets/list initialize the GCS client with the service account JSON.

  return data as GcpCloudConnection;
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
