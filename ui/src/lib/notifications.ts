// Global in-app notifications. Persistent per profile, realtime-enabled.

import { supabase } from "@/integrations/supabase/client";
import { getCurrentProfileId } from "@/lib/current-profile";

export type NotificationType =
  | "analysis.completed"
  | "comment.created"
  | "collaborator.invited"
  | "share.created"
  | "dataset.uploaded"
  | "generic";

export type NotificationRow = {
  id: string;
  recipient_id: string;
  actor_id: string | null;
  type: string;
  title: string;
  body: string | null;
  resource_type: string | null;
  resource_id: string | null;
  link: string | null;
  read_at: string | null;
  created_at: string;
};

const TABLE = "notifications" as never;

export async function listNotifications(limit = 30): Promise<NotificationRow[]> {
  const { data, error } = await supabase
    .from(TABLE)
    .select("*")
    .order("created_at", { ascending: false })
    .limit(limit);
  if (error) {
    console.warn("[notifications] list failed", error.message);
    return [];
  }
  return (data ?? []) as unknown as NotificationRow[];
}

export async function markRead(id: string): Promise<void> {
  const { error } = await supabase
    .from(TABLE)
    .update({ read_at: new Date().toISOString() } as never)
    .eq("id", id)
    .is("read_at", null);
  if (error) console.warn("[notifications] markRead", error.message);
}

export async function markAllRead(): Promise<void> {
  const profileId = await getCurrentProfileId();
  if (!profileId) return;
  const { error } = await supabase
    .from(TABLE)
    .update({ read_at: new Date().toISOString() } as never)
    .eq("recipient_id", profileId)
    .is("read_at", null);
  if (error) console.warn("[notifications] markAllRead", error.message);
}

export async function deleteNotification(id: string): Promise<void> {
  const { error } = await supabase.from(TABLE).delete().eq("id", id);
  if (error) console.warn("[notifications] delete", error.message);
}

export async function deleteAllNotifications(): Promise<void> {
  const profileId = await getCurrentProfileId();
  if (!profileId) return;
  const { error } = await supabase.from(TABLE).delete().eq("recipient_id", profileId);
  if (error) console.warn("[notifications] deleteAll", error.message);
}

export type NotifyInput = {
  recipientId: string;
  type: NotificationType | string;
  title: string;
  body?: string;
  resourceType?: string;
  resourceId?: string;
  link?: string;
};

/** Insert a single notification. Actor is the current profile (if signed in). */
export async function notify(input: NotifyInput): Promise<void> {
  const actor = await getCurrentProfileId();
  const row = {
    recipient_id: input.recipientId,
    actor_id: actor,
    type: input.type,
    title: input.title,
    body: input.body ?? null,
    resource_type: input.resourceType ?? null,
    resource_id: input.resourceId ?? null,
    link: input.link ?? null,
  };
  const { error } = await supabase.from(TABLE).insert(row as never);
  if (error) console.warn("[notifications] notify failed", error.message);
}

/** Insert notifications for many recipients at once. */
export async function notifyMany(
  recipientIds: string[],
  base: Omit<NotifyInput, "recipientId">,
): Promise<void> {
  const unique = [...new Set(recipientIds.filter(Boolean))];
  if (!unique.length) return;
  const actor = await getCurrentProfileId();
  const rows = unique.map((rid) => ({
    recipient_id: rid,
    actor_id: actor,
    type: base.type,
    title: base.title,
    body: base.body ?? null,
    resource_type: base.resourceType ?? null,
    resource_id: base.resourceId ?? null,
    link: base.link ?? null,
  }));
  const { error } = await supabase.from(TABLE).insert(rows as never);
  if (error) {
    console.warn("[notifications] notifyMany failed", error.message);
    throw error;
  }
}
