import { useCallback, useEffect, useRef, useState } from "react";
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

export function useNotifications() {
  const [items, setItems] = useState<NotificationRow[]>([]);
  const [profileId, setProfileId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const channelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);

  const refresh = useCallback(async () => {
    const rows = await listNotifications(30);
    setItems(rows);
    setLoading(false);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const pid = await getCurrentProfileId();
      if (cancelled) return;
      setProfileId(pid);
      if (!pid) {
        setLoading(false);
        return;
      }
      await refresh();

      const channel = supabase
        .channel(`notifications:${pid}`)
        .on(
          "postgres_changes",
          {
            event: "*",
            schema: "public",
            table: "notifications",
            filter: `recipient_id=eq.${pid}`,
          },
          (payload) => {
            if (payload.eventType === "INSERT") {
              const row = payload.new as NotificationRow;
              setItems((prev) => [row, ...prev].slice(0, 30));
              toast(row.title, {
                description: row.body ?? undefined,
                position: "top-right",
                duration: 5000,
                closeButton: true,
              });
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
      channelRef.current = channel;
    })();

    return () => {
      cancelled = true;
      if (channelRef.current) supabase.removeChannel(channelRef.current);
      channelRef.current = null;
    };
  }, [refresh]);

  const unreadCount = items.filter((n) => !n.read_at).length;

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
    setItems([]);
    await deleteAllRemote();
  }, []);

  return { items, unreadCount, loading, profileId, markOne, markAll, removeOne, clearAll, refresh };
}
