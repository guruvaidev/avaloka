// The dataset chips of an upload, kept in sessionStorage so a reload or a
// return to the page restores the whole batch. What comes back is a JSON copy:
// nothing restored here is the same object as what was stored, so code that
// recognises a chip's config must compare it by content, not identity
// (deferred-profile.chipsWithFinishedProfile).

export const BATCH_KEY = "analysis:batch";
export type StoredBatch<C> = { chips: C[]; selectedId: string | null };

export function storeDatasetBatch<C>(analysisId: string | null | undefined, batch: StoredBatch<C>) {
  try {
    const raw = JSON.stringify(batch);
    sessionStorage.setItem(BATCH_KEY, raw);
    if (analysisId) sessionStorage.setItem(`${BATCH_KEY}:${analysisId}`, raw);
  } catch {
    /* ignore */
  }
}

export function readDatasetBatch<C>(analysisId: string | null | undefined): StoredBatch<C> | null {
  try {
    const raw =
      (analysisId ? sessionStorage.getItem(`${BATCH_KEY}:${analysisId}`) : null) ??
      sessionStorage.getItem(BATCH_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredBatch<C>;
    return Array.isArray(parsed?.chips) && parsed.chips.length ? parsed : null;
  } catch {
    return null;
  }
}
