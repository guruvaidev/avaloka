// Resolves the current user's organization id on the client.
//
// The database trigger `set_row_organization()` fills `organization_id` only
// when it exists on the target project; to make ownership stamping reliable we
// also resolve it here and send it explicitly on insert.
//
// Resolution order (mirrors the DB `current_org_id()` function):
//   1. profiles.organization_id
//   2. organization owned by this profile (organizations.owner_profile_id)
//   3. app_users membership row (profile_id / auth_user_id / email match)

import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";
import { getCurrentProfileId } from "@/lib/current-profile";

let cache: { profileId: string; orgId: string | null } | null = null;

export function clearCurrentOrgCache() {
  cache = null;
}

supabase.auth.onAuthStateChange((event) => {
  if (event === "SIGNED_OUT" || event === "SIGNED_IN" || event === "USER_UPDATED") {
    clearCurrentOrgCache();
  }
});

export async function getCurrentOrgId(): Promise<string | null> {
  const profileId = await getCurrentProfileId();
  if (!profileId) return null;
  if (cache && cache.profileId === profileId && cache.orgId) return cache.orgId;

  let orgId: string | null = null;

  // 1. profile link
  const prof = await supabase
    .from("profiles")
    .select("organization_id, company_email")
    .eq("id", profileId)
    .maybeSingle();
  orgId = (prof.data?.organization_id as string | null) ?? null;

  // 2. organization owned by this profile
  if (!orgId) {
    const owned = await supabase
      .from("organizations")
      .select("id")
      .eq("owner_profile_id", profileId)
      .limit(1)
      .maybeSingle();
    orgId = (owned.data?.id as string | null) ?? null;
  }

  // 3. membership row in app_users
  if (!orgId) {
    const authUser = await getCachedAuthUser();
    const email = (authUser?.email ?? prof.data?.company_email ?? "").toLowerCase();
    const filters = [`profile_id.eq.${profileId}`];
    if (authUser?.id) filters.push(`auth_user_id.eq.${authUser.id}`);
    if (email) filters.push(`email.eq.${email}`);
    const member = await supabase
      .from("app_users")
      .select("organization_id")
      .or(filters.join(","))
      .not("organization_id", "is", null)
      .limit(1)
      .maybeSingle();
    orgId = (member.data?.organization_id as string | null) ?? null;
  }

  // 4. Server-side fallback (service role) — RLS can hide the membership rows
  //    above for invited collaborators / admins.
  if (!orgId) {
    try {
      const { hasSupabaseSession } = await import("@/lib/has-session");
      if (await hasSupabaseSession()) {
        const { resolveMyOrgId } = await import("@/lib/current-org.functions");
        orgId = ((await resolveMyOrgId()) as string | null) ?? null;
      }
    } catch {
      orgId = null;
    }
  }

  cache = { profileId, orgId };
  return orgId;
}

