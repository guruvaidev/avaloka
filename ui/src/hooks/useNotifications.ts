import { useCallback, useSyncExternalStore } from "react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { getCurrentProfileId } from "@/lib/current-profile";
import {
  listNotifications,
  markAllRead as markAllReadRemote,
  markRead as markReadRemote,
  deleteNotification as deleteNotificationRemote,
  deleteAllNotifications as deleteAllRemote,
  type NotificationRow,
} from "@/lib/notifications";

/**
 * Single shared notifications store. However many components use the hook,
 * there is exactly one fetch loop (15s, paused while the tab is hidden), one
 * realtime channel and one list in memory. Started on first subscriber,
 * torn down when the last one unmounts.
 */
const POLL_MS = 15_000;

type State = { items: NotificationRow[]; profileId: string | null; loading: boolean };
let state: State = { items: [], profileId: null, loading: true };
const subs = new Set<() => void>();
let refCount = 0;
let timer: ReturnType<typeof setInterval> | null = null;
let channel: ReturnType<typeof supabase.channel> | null = null;
let inFlight: Promise<void> | null = null;
let generation = 0;

function set(next: Partial<State>) {
  state = { ...state, ...next };
  subs.forEach((s) => s());
}
function setItems(fn: (prev: NotificationRow[]) => NotificationRow[]) {
  set({ items: fn(state.items) });
}

function refresh(): Promise<void> {
  if (inFlight) return inFlight;
  inFlight = (async () => {
    try {
      const rows = await listNotifications(30);
      set({ items: rows, loading: false });
    } finally {
      inFlight = null;
    }
  })();
  return inFlight;
}

const startTimer = () => {
  if (!timer) timer = setInterval(() => !document.hidden && void refresh(), POLL_MS);
};
const stopTimer = () => {
  if (timer) clearInterval(timer);
  timer = null;
};
const onVisibility = () => {
  if (document.hidden) stopTimer();
  else {
    void refresh();
    startTimer();
  }
};
const onRefreshEvent = () => void refresh();

async function start() {
  const gen = ++generation;
  const pid = await getCurrentProfileId();
  if (gen !== generation) return;
  set({ profileId: pid });
  if (!pid) {
    set({ loading: false });
    return;
  }
  await refresh();
  if (gen !== generation) return;
  channel = supabase
    .channel(`notifications:${pid}`)
    .on(
      "postgres_changes",
      { event: "*", schema: "public", table: "notifications", filter: `recipient_id=eq.${pid}` },
      (payload) => {
        if (payload.eventType === "INSERT") {
          const row = payload.new as NotificationRow;
          setItems((prev) => [row, ...prev.filter((n) => n.id !== row.id)].slice(0, 30));
          toast(row.title, { description: row.body ?? undefined, position: "top-right", duration: 5000, closeButton: true });
        } else if (payload.eventType === "UPDATE") {
          const updated = payload.new as NotificationRow;
          setItems((prev) => prev.map((n) => (n.id === updated.id ? updated : n)));
        } else if (payload.eventType === "DELETE") {
          const oldRow = payload.old as { id: string };
          setItems((prev) => prev.filter((n) => n.id !== oldRow.id));
        }
      },
    )
    .subscribe();
  if (!document.hidden) startTimer();
  document.addEventListener("visibilitychange", onVisibility);
  window.addEventListener("app-notifications:refresh", onRefreshEvent);
}

function stop() {
  generation++;
  stopTimer();
  document.removeEventListener("visibilitychange", onVisibility);
  window.removeEventListener("app-notifications:refresh", onRefreshEvent);
  if (channel) supabase.removeChannel(channel);
  channel = null;
}

function subscribe(cb: () => void) {
  subs.add(cb);
  if (++refCount === 1) void start();
  return () => {
    subs.delete(cb);
    if (--refCount === 0) stop();
  };
}

const getSnapshot = () => state;

export function useNotifications() {
  const s = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  const markOne = useCallback(async (id: string) => {
    setItems((prev) => prev.map((n) => (n.id === id && !n.read_at ? { ...n, read_at: new Date().toISOString() } : n)));
    await markReadRemote(id);
  }, []);
  const markAll = useCallback(async () => {
    const now = new Date().toISOString();
    setItems((prev) => prev.map((n) => (n.read_at ? n : { ...n, read_at: now })));
    await markAllReadRemote();
  }, []);
  const removeOne = useCallback(async (id: string) => {
    setItems((prev) => prev.filter((n) => n.id !== id));
    await deleteNotificationRemote(id);
  }, []);
  const clearAll = useCallback(async () => {
    setItems(() => []);
    await deleteAllRemote();
  }, []);

  const unreadCount = s.items.filter((n) => !n.read_at).length;
  return { items: s.items, unreadCount, loading: s.loading, profileId: s.profileId, markOne, markAll, removeOne, clearAll, refresh };
}
