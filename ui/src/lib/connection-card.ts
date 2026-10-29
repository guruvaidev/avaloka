import type { DataSource } from "@/lib/data-sources";
import type { StorageConnection } from "@/components/database/CloudStorageConnectModal";

/**
 * Builds the bucket-browser target for a cloud connection card from that card's
 * own row only. Previously selected files are ignored so a card click can never
 * reopen an earlier dataset.
 */
export function storageConnectionFromSource(source: DataSource): StorageConnection | null {
  if (source.connectionType !== "storage" || !source.storageProvider) return null;
  const bucket = typeof source.config?.bucket_name === "string" ? source.config.bucket_name : "";
  return { id: source.id, name: source.name, provider: source.storageProvider, bucket_name: bucket };
}
