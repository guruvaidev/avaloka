import { supabase } from "@/integrations/supabase/client";
import { backendApi } from "@/lib/api/backendApi";


/** Scheme per provider — mirrors CloudStorageBrowserModal. */
function schemeFor(provider?: string): "gs" | "s3" | "az" {
  const p = (provider ?? "").toLowerCase();
  if (p === "aws" || p === "s3") return "s3";
  if (p === "azure" || p === "azure_blob") return "az";
  return "gs";
}

/**
 * Stored `bucket_name` may include a prefix (bucket/folder/sub). Selected file
 * keys are already bucket-root-relative and repeat that prefix, so the
 * storage_uri must be anchored to the bucket ROOT only — same as the browse flow.
 */
function buildRootStorageUri(bucketPath: string, provider?: string): string {
  const path = bucketPath.trim().replace(/^[a-z0-9]+:\/\//i, "").replace(/^\/+|\/+$/g, "");
  const rootBucket = path.split("/")[0] ?? "";
  if (!rootBucket) throw new Error("Missing bucket path on this connection.");
  return `${schemeFor(provider)}://${rootBucket}`;
}
import type { DataSource } from "@/lib/data-sources";

type CloudUploadResponse = Awaited<ReturnType<typeof backendApi.registerExistingStorage>>;

type StoredRegistration = {
  file_key: string;
  filename: string;
  upload: CloudUploadResponse;
};

const browserCacheKey = (connectionId: string) => `cloud-analysis:${connectionId}`;

function seedAnalysisSession(uploadResponse: CloudUploadResponse, filename: string) {
  try {
    sessionStorage.setItem(
      "analysis:dataset",
      JSON.stringify({
        filename,
        schema: uploadResponse.schema,
        samples: uploadResponse.samples,
        rows_sampled: uploadResponse.rows_sampled,
        visualization_config: uploadResponse.visualization_config,
        visualization_status: uploadResponse.visualization_status,
      }),
    );
    const chips = uploadResponse.dataset_id
      ? [{ id: uploadResponse.dataset_id, name: filename }]
      : [];
    sessionStorage.setItem(
      "analysis:session",
      JSON.stringify({
        threadId: uploadResponse.thread_id,
        sessionId: uploadResponse.session_id,
        datasetChips: chips,
      }),
    );
  } catch {
    /* sessionStorage may be unavailable */
  }
}

function readBrowserCache(connectionId: string, fileKey: string): StoredRegistration | null {
  try {
    const raw = sessionStorage.getItem(browserCacheKey(connectionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredRegistration;
    if (parsed.file_key !== fileKey || !parsed.upload?.dataset_id) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeBrowserCache(connectionId: string, stored: StoredRegistration) {
  try {
    sessionStorage.setItem(browserCacheKey(connectionId), JSON.stringify(stored));
  } catch {
    /* ignore */
  }
}

/** Reuse a prior analysis — no backend register call. */
function readStoredRegistration(
  row: Record<string, unknown>,
  fileKey: string,
): StoredRegistration | null {
  const browser = readBrowserCache(row.id as string, fileKey);
  if (browser) return browser;

  const schema = (row.schema as Record<string, unknown>) ?? {};
  const reg = schema.registration as
    | { file_key?: string; filename?: string; upload?: CloudUploadResponse }
    | undefined;

  if (reg?.upload?.dataset_id && reg.file_key === fileKey) {
    return {
      file_key: fileKey,
      filename: reg.filename ?? fileKey.split("/").pop() ?? "Dataset",
      upload: reg.upload,
    };
  }

  // Legacy rows: ids on cloud_datasets after a prior analysis (before full cache was saved)
  const datasetId = row.dataset_id as string | null;
  const sessionId = row.session_id as string | null;
  const rowFileKey = row.file_key as string | null;
  if (datasetId && sessionId && rowFileKey === fileKey) {
    const cachedUpload = reg?.upload;
    return {
      file_key: fileKey,
      filename: reg?.filename ?? fileKey.split("/").pop() ?? "Dataset",
      upload: {
        dataset_id: datasetId,
        session_id: sessionId,
        thread_id: (row.thread_id as string) ?? cachedUpload?.thread_id ?? "",
        schema: cachedUpload?.schema ?? [],
        samples: cachedUpload?.samples ?? [],
        ddl_schema: cachedUpload?.ddl_schema ?? "",
        rows_sampled: cachedUpload?.rows_sampled ?? 0,
        visualization_config: cachedUpload?.visualization_config,
        visualization_status: cachedUpload?.visualization_status,
        portfolio_samples: cachedUpload?.portfolio_samples,
        available_samples: cachedUpload?.available_samples,
      },

    };
  }

  return null;
}

async function persistRegistration(
  connectionId: string,
  fileKey: string,
  filename: string,
  uploadResponse: CloudUploadResponse,
  existingSchema: Record<string, unknown>,
) {
  const stored: StoredRegistration = { file_key: fileKey, filename, upload: uploadResponse };
  writeBrowserCache(connectionId, stored);

  await supabase
    .from("cloud_datasets")
    .update({
      dataset_id: uploadResponse.dataset_id,
      session_id: uploadResponse.session_id,
      thread_id: uploadResponse.thread_id,
      file_key: fileKey,
      schema: {
        ...existingSchema,
        registration: stored,
      },
    })
    .eq("id", connectionId);
}

/**
 * Open /analysis for a cloud connection.
 * First click: registers the GCS file with the backend (one-time analysis setup).
 * Later clicks: reuses the saved session and navigates immediately — no re-analysis.
 */
export async function openCloudDatasetInAnalysis(
  source: DataSource,
  navigate: any,
): Promise<void> {
  const selectedFiles = (source.config.selected_files as string[]) ?? [];
  if (!selectedFiles.length) {
    throw new Error("No files linked to this connection. Connect again and select data files.");
  }

  const { data: row, error } = await supabase
    .from("cloud_datasets")
    .select("*")
    .eq("id", source.id)
    .single();
  if (error || !row) throw new Error(error?.message ?? "Connection not found");

  const fileKey =
    (typeof row.file_key === "string" && row.file_key && selectedFiles.includes(row.file_key)
      ? row.file_key
      : selectedFiles[0]) ?? "";
  const bucketPath = row.bucket_name as string | null;
  if (!bucketPath) throw new Error("Missing bucket path on this connection.");

  const filename = fileKey.split("/").pop() || source.name;
  const existingSchema = ((row.schema as Record<string, unknown>) ?? {}) as Record<string, unknown>;

  const stored = readStoredRegistration(row as Record<string, unknown>, fileKey);
  if (stored) {
    seedAnalysisSession(stored.upload, stored.filename);
    navigate({
      to: "/analysis",
      state: { uploadResponse: stored.upload, filename: stored.filename },
    });
    return;
  }

  const uploadResponse = await backendApi.registerExistingStorage({
    storage_uri: buildRootStorageUri(bucketPath, (row as any).provider as string | undefined),
    key: fileKey,
    connection_id: source.id,
  });

  await persistRegistration(source.id, fileKey, filename, uploadResponse, existingSchema);

  seedAnalysisSession(uploadResponse, filename);
  navigate({
    to: "/analysis",
    state: { uploadResponse, filename },
  });
}
