// Other people's names and avatars.
//
// Another user's profile row is not readable directly: it holds their phone,
// company and recovery email and billing ids (migration
// 20261103120000_admin_no_cross_tenant.sql). public.get_display_profiles(ids)
// returns display fields only (id, user_id, names, avatar), and only for people
// related to the caller: the same organisation, or someone on a report, analysis
// or project the caller takes part in. It accepts profile ids or auth user ids.

export type DisplayProfile = {
  id: string;
  user_id: string | null;
  full_name: string | null;
  first_name: string | null;
  last_name: string | null;
  avatar_url: string | null;
};

// The function reads at most this many ids per call.
const CHUNK = 500;

export async function fetchDisplayProfiles(
  supabase: any,
  ids: ReadonlyArray<string | null | undefined>,
): Promise<DisplayProfile[]> {
  const unique = Array.from(new Set(ids.filter((v): v is string => !!v)));
  const out: DisplayProfile[] = [];
  for (let i = 0; i < unique.length; i += CHUNK) {
    const { data, error } = await supabase.rpc("get_display_profiles", {
      p_ids: unique.slice(i, i + CHUNK),
    });
    if (error) {
      console.warn("[profiles] display lookup failed", error.message);
      continue;
    }
    out.push(...((data ?? []) as DisplayProfile[]));
  }
  return out;
}

/** The rows keyed by profile id and by auth user id. */
export function indexDisplayProfiles(rows: ReadonlyArray<DisplayProfile>): Map<string, DisplayProfile> {
  const map = new Map<string, DisplayProfile>();
  for (const p of rows) {
    map.set(p.id, p);
    if (p.user_id) map.set(p.user_id, p);
  }
  return map;
}

/** Full name, else first and last, else "". */
export function displayNameOf(p: DisplayProfile | null | undefined): string {
  if (!p) return "";
  return (p.full_name || `${p.first_name ?? ""} ${p.last_name ?? ""}`).trim();
}

/**
 * Sets `author` on each row from its `author_id` (a profile id), using display
 * fields only. A comment author on a shared report or analysis may be in
 * another organisation, where a profiles embed would come back empty.
 */
export async function attachDisplayAuthors<T extends { author_id?: string | null }>(
  supabase: any,
  rows: ReadonlyArray<T>,
): Promise<Array<T & { author: { id: string; full_name: string | null; avatar_url: string | null } | null }>> {
  const index = indexDisplayProfiles(await fetchDisplayProfiles(supabase, rows.map((r) => r.author_id)));
  return rows.map((r) => {
    const p = r.author_id ? index.get(r.author_id) : undefined;
    return { ...r, author: p ? { id: p.id, full_name: displayNameOf(p) || null, avatar_url: p.avatar_url } : null };
  });
}
