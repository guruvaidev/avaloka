// Filters notification recipients by their saved notification preferences.
// "comment" maps to profiles.notification_settings.comments.push,
// "mention" maps to profiles.notification_settings.tags.push.

type Admin = {
  from: (table: string) => {
    select: (cols: string) => {
      in: (col: string, values: string[]) => Promise<{ data: unknown }>;
    };
  };
};

type Prefs = {
  comments?: { push?: boolean };
  tags?: { push?: boolean };
};

export async function filterRecipientsByPreference(
  admin: unknown,
  profileIds: string[],
  kind: "comment" | "mention",
): Promise<string[]> {
  const ids = [...new Set(profileIds)].filter(Boolean);
  if (!ids.length) return [];

  const { data } = await (admin as Admin)
    .from("profiles")
    .select("id,notification_settings")
    .in("id", ids);

  const rows = (data ?? []) as { id: string; notification_settings: Prefs | null }[];
  const byId = new Map(rows.map((r) => [r.id, r.notification_settings ?? null]));

  return ids.filter((id) => {
    const prefs = byId.get(id);
    if (!prefs) return true; // default: notify
    const enabled = kind === "comment" ? prefs.comments?.push : prefs.tags?.push;
    return enabled !== false;
  });
}
