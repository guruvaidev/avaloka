// Server function: list people the current user can @mention.
import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

export type MentionCandidate = {
  profileId: string;
  name: string;
  email: string | null;
  avatarUrl: string | null;
};

export const listMentionCandidates = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }): Promise<MentionCandidate[]> => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const authUserId = context.userId;

    // Resolve caller profile + organization.
    const { data: myProfiles } = await supabaseAdmin
      .from("profiles")
      .select("id,user_id,organization_id")
      .or(`id.eq.${authUserId},user_id.eq.${authUserId}`);
    const me = ((myProfiles ?? []) as { id: string; organization_id: string | null }[])[0];
    if (!me) return [];

    const orgId = me.organization_id;
    if (!orgId) return [];

    const { data: appUsers } = await supabaseAdmin
      .from("app_users")
      .select("email,auth_user_id,profile_id,status")
      .eq("organization_id", orgId);

    const rows = (appUsers ?? []) as {
      email: string | null;
      auth_user_id: string | null;
      profile_id: string | null;
      status: string | null;
    }[];

    const authIds = rows.map((r) => r.auth_user_id).filter((v): v is string => !!v);
    const profileIds = rows.map((r) => r.profile_id).filter((v): v is string => !!v);

    const [{ data: byUserId }, { data: byId }] = await Promise.all([
      authIds.length
        ? supabaseAdmin.from("profiles").select("id,user_id,full_name,avatar_url").in("user_id", authIds)
        : Promise.resolve({ data: [] as never[] }),
      supabaseAdmin
        .from("profiles")
        .select("id,user_id,full_name,avatar_url")
        .in("id", [...new Set([...authIds, ...profileIds])].length ? [...new Set([...authIds, ...profileIds])] : ["00000000-0000-0000-0000-000000000000"]),
    ]);

    type Prof = { id: string; user_id: string | null; full_name: string | null; avatar_url: string | null };
    const profs = [...((byUserId ?? []) as Prof[]), ...((byId ?? []) as Prof[])];

    const out = new Map<string, MentionCandidate>();
    for (const row of rows) {
      const match = profs.find(
        (p) =>
          (row.profile_id && p.id === row.profile_id) ||
          (row.auth_user_id && (p.user_id === row.auth_user_id || p.id === row.auth_user_id)),
      );
      if (!match) continue;
      if (match.id === me.id) continue;
      const name = match.full_name?.trim() || row.email?.split("@")[0] || "Unknown user";
      out.set(match.id, {
        profileId: match.id,
        name,
        email: row.email,
        avatarUrl: match.avatar_url,
      });
    }

    return [...out.values()].sort((a, b) => a.name.localeCompare(b.name));
  });
