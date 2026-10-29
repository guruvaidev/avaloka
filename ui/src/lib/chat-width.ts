/** Resizable chat panel width: pure clamp rules plus try/catch persistence. */
export const CHAT_WIDTH_STORAGE_KEY = "avaloka.chatWidth";
export const CHAT_WIDTH_MIN = 320;
export const CHAT_WIDTH_MAX = 720;
export const CHAT_WIDTH_STEP = 16;
/** The data area always keeps at least this many pixels. */
export const DATA_AREA_MIN = 480;
/** Icon rail plus workspace gaps that sit beside the data area and chat. */
export const WORKSPACE_CHROME = 112;

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function defaultStorage(): StorageLike | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

/** The pre-existing CSS default: clamp(320px, 26vw, 400px). */
export function defaultChatWidth(viewport: number): number {
  return Math.round(Math.min(400, Math.max(CHAT_WIDTH_MIN, viewport * 0.26)));
}

/** min(720px, 55% of the window), never squeezing the data area below 480px, never below the minimum. */
export function maxChatWidth(viewport: number): number {
  return Math.round(
    Math.max(CHAT_WIDTH_MIN, Math.min(CHAT_WIDTH_MAX, viewport * 0.55, viewport - WORKSPACE_CHROME - DATA_AREA_MIN)),
  );
}

export function clampChatWidth(width: number, viewport: number): number {
  if (!Number.isFinite(width)) return defaultChatWidth(viewport);
  return Math.round(Math.min(maxChatWidth(viewport), Math.max(CHAT_WIDTH_MIN, width)));
}

/** Remembered width clamped to the current limits, or null when nothing valid is stored. */
export function readChatWidth(viewport: number, storage?: StorageLike | null): number | null {
  try {
    const raw = (storage ?? defaultStorage())?.getItem(CHAT_WIDTH_STORAGE_KEY);
    if (raw == null) return null;
    const value = Number(raw);
    return Number.isFinite(value) ? clampChatWidth(value, viewport) : null;
  } catch {
    return null;
  }
}

export function writeChatWidth(width: number, storage?: StorageLike | null): void {
  try {
    (storage ?? defaultStorage())?.setItem(CHAT_WIDTH_STORAGE_KEY, String(Math.round(width)));
  } catch {
    /* storage unavailable: keep the in-memory width */
  }
}

/** Double-click reset: forget the custom width so the default applies again. */
export function clearChatWidth(storage?: StorageLike | null): void {
  try {
    (storage ?? defaultStorage())?.removeItem(CHAT_WIDTH_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
