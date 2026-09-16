// Server-only helper: organization owners implicitly see everything created by
// members of their organization (projects, analyses, reports) — no invite needed.

export type OrgOwnerScope = {
  profileId: string | null;
  orgIds: string[];
  /** Profile ids of org members (excluding the owner's own profile). */
  memberProfileIds: string[];
  /** Auth user ids matching those member profiles (reports.created_by is an auth id). */
  memberAuthUserIds: string[];
};

const EMPTY: OrgOwnerScope = {
  profileId: null,
  orgIds: [],
  memberProfileIds: [],
  memberAuthUserIds: [],
};

export async function resolveOrgOwnerScope(
  admin: any,
  authUserId: string,
): Promise<OrgOwnerScope> {
  if (!authUserId) return EMPTY;

  const { data: profile } = await admin
    .from("profiles")
    .select("id")
    .or(`id.eq.${authUserId},user_id.eq.${authUserId}`)
    .maybeSingle();

  const profileId: string | null = profile?.id ?? null;
  if (!profileId) return EMPTY;

  const { data: orgs } = await admin
    .from("organizations")
    .select("id")
    .eq("owner_profile_id", profileId);

  const orgIds = ((orgs ?? []) as { id: string }[]).map((o) => o.id);
  if (!orgIds.length) return { ...EMPTY, profileId };

  const { data: members } = await admin
    .from("profiles")
    .select("id,user_id")
    .in("organization_id", orgIds);

  const rows = (members ?? []) as { id: string; user_id: string | null }[];
  const memberProfileIds = rows.map((r) => r.id).filter((id) => id !== profileId);
  const memberAuthUserIds = Array.from(
    new Set(
      rows
        .flatMap((r) => [r.user_id, r.id])
        .filter((v): v is string => !!v && v !== profileId && v !== authUserId),
    ),
  );

  return { profileId, orgIds, memberProfileIds, memberAuthUserIds };
}
