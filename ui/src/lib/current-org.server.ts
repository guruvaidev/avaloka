// Server-only: resolve an auth user's organization id, bypassing RLS.
//
// Order: profiles.organization_id -> organization owned by the profile ->
// app_users membership (profile_id / auth_user_id / email match).

export async function resolveOrgIdForAuthUser(
  admin: any,
  authUserId: string,
): Promise<string | null> {
  if (!authUserId) return null;

  const { data: profile } = await admin
    .from("profiles")
    .select("id,organization_id,company_email")
    .or(`id.eq.${authUserId},user_id.eq.${authUserId}`)
    .maybeSingle();

  if (profile?.organization_id) return profile.organization_id as string;

  const profileId: string | null = profile?.id ?? null;

  if (profileId) {
    const { data: owned } = await admin
      .from("organizations")
      .select("id")
      .eq("owner_profile_id", profileId)
      .limit(1)
      .maybeSingle();
    if (owned?.id) return owned.id as string;
  }

  let email: string | null = (profile?.company_email as string | null) ?? null;
  try {
    const { data: authUser } = await admin.auth.admin.getUserById(authUserId);
    email = authUser?.user?.email ?? email;
  } catch {
    /* ignore */
  }

  const filters = [`auth_user_id.eq.${authUserId}`];
  if (profileId) filters.push(`profile_id.eq.${profileId}`);
  if (email) filters.push(`email.ilike.${email}`);

  const { data: member } = await admin
    .from("app_users")
    .select("organization_id")
    .or(filters.join(","))
    .not("organization_id", "is", null)
    .limit(1)
    .maybeSingle();

  return (member?.organization_id as string | null) ?? null;
}
