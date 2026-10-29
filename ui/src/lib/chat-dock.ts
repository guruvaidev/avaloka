export type ChatDockSide = "left" | "right" | "float";

export const CHAT_DOCK_STORAGE_KEY = "avaloka.chatDock";

type StorageLike = Pick<Storage, "getItem" | "setItem">;

function defaultStorage(): StorageLike | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

/** Reads the remembered chat dock mode; defaults to "right" on any failure or unknown value. */
export function readChatDock(storage?: StorageLike | null): ChatDockSide {
  try {
    const value = (storage ?? defaultStorage())?.getItem(CHAT_DOCK_STORAGE_KEY);
    return value === "left" || value === "float" ? value : "right";
  } catch {
    return "right";
  }
}

/** Persists the chat dock mode; storage failures are ignored. */
export function writeChatDock(side: ChatDockSide, storage?: StorageLike | null): void {
  try {
    (storage ?? defaultStorage())?.setItem(CHAT_DOCK_STORAGE_KEY, side);
  } catch {
    /* storage unavailable: keep the in-memory choice only */
  }
}
