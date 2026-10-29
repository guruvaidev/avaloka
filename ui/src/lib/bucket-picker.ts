export type BucketObject = { key: string; size?: string | number; updated?: string; content_type?: string; contentType?: string };
export type BucketFolder = { name: string; prefix: string; size?: string | number | null; content_type?: string; contentType?: string; key?: string; updated?: string };

export function folderHint(name: string): string | undefined {
  return /\.(csv|parquet|json)$/i.test(name.replace(/\/+$/, ""))
    ? "Folder of files (written by a job)"
    : undefined;
}

export function selectionLabel(folders: number, files: number): string {
  const parts: string[] = [];
  if (folders) parts.push(`${folders} folder${folders === 1 ? "" : "s"}`);
  if (files || !folders) parts.push(`${files} file${files === 1 ? "" : "s"}`);
  return `${parts.join(" and ")} selected`;
}

export function useSelectionLabel(folders: number, files: number): string {
  if (folders) return folders === 1 ? "Use folder" : "Use folders";
  return files === 1 ? "Use file" : "Use files";
}

/** Classify only metadata already returned by the existing listing endpoint. */
export function presentBucketListing(objects: BucketObject[], folders: BucketFolder[]) {
  const files = new Map(objects.map((object) => [object.key, object]));
  const directories: BucketFolder[] = [];
  for (const folder of folders) {
    const hasSize = folder.size !== undefined && folder.size !== null && folder.size !== "";
    if (hasSize || folder.content_type || folder.contentType) {
      const key = folder.key || folder.prefix.replace(/\/+$/, "");
      if (!files.has(key)) files.set(key, { key, size: folder.size ?? undefined, updated: folder.updated, content_type: folder.content_type || folder.contentType });
    } else {
      directories.push(folder);
    }
  }
  return { objects: [...files.values()], folders: directories };
}