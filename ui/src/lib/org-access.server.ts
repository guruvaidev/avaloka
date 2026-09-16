// Server-only helper: decide whether an organization member may sign in.
//
// Team members (non-owner app_users linked to an organization) only exist on
// the Enterprise plan. When an org downgrades to Professional (single user),
// invited members lose login access until the org is back on Enterprise.
// Nothing is deleted — access is restored automatically on re-upgrade.

export const ORG_DOWNGRADED_MESSAGE =
  "Your organization's plan no longer includes team access. Please contact your organization administrator.";

type AppUserRow = {
  id: string;
  status: string | null;
  role: string | null;
  organization_id: string | null;
} | null;

export async function findAppUser(
  admin: any,
  opts: { authUserId?: string | null; email?: string | null },
): Promise<AppUserRow> {
  const cols = "id,status,role,organization_id";
  if (opts.authUserId) {
    const { data } = await admin
      .from("app_users")
      .select(cols)
      .eq("auth_user_id", opts.authUserId)
      .limit(1)
      .maybeSingle();
    if (data) return data as AppUserRow;
  }
  const email = String(opts.email ?? "").trim().toLowerCase();
  if (email) {
    const { data } = await admin
      .from("app_users")
      .select(cols)
      .ilike("email", email)
      .limit(1)
      .maybeSingle();
    if (data) return data as AppUserRow;
  }
  return null;
}

export async function orgHasActiveEnterprise(
  admin: any,
  organizationId: string,
): Promise<boolean> {
  const { data } = await admin
    .from("subscriptions")
    .select("status, trial_ends_at, plan:plans(plan_type,name)")
    .eq("organization_id", organizationId)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (!data) return false;

  const status = String((data as any).status ?? "").toLowerCase();
  const trialEnds = (data as any).trial_ends_at
    ? new Date((data as any).trial_ends_at).getTime()
    : 0;
  const trialActive =
    trialEnds > Date.now() &&
    !["canceled", "cancelled", "unpaid", "incomplete_expired"].includes(status);
  const active =
    ["trialing", "active", "trial"].includes(status) || trialActive;
  if (!active) return false;

  const plan = (data as any).plan ?? {};
  const type = String(plan.plan_type ?? "").toLowerCase();
  const name = String(plan.name ?? "").toLowerCase();
  return type === "enterprise" || name.includes("enterprise");
}

/**
 * Returns a blocking message when the account may not sign in, else null.
 */
export async function loginBlockReason(
  admin: any,
  opts: { authUserId?: string | null; email?: string | null },
): Promise<string | null> {
  const appUser = await findAppUser(admin, opts);
  if (!appUser) return null;

  if (String(appUser.status ?? "").toLowerCase() === "inactive") {
    return "Your account has been deactivated. Please contact your organization administrator.";
  }

  const orgId = appUser.organization_id;
  if (!orgId) return null;

  // Org owners always keep access to manage/upgrade their plan.
  if (String(appUser.role ?? "").toLowerCase() === "owner") return null;

  const { data: org } = await admin
    .from("organizations")
    .select("owner_profile_id")
    .eq("id", orgId)
    .maybeSingle();
  if (org?.owner_profile_id && opts.authUserId) {
    const { data: profile } = await admin
      .from("profiles")
      .select("id")
      .or(`id.eq.${opts.authUserId},user_id.eq.${opts.authUserId}`)
      .maybeSingle();
    if (profile?.id && profile.id === org.owner_profile_id) return null;
  }

  const enterprise = await orgHasActiveEnterprise(admin, orgId);
  return enterprise ? null : ORG_DOWNGRADED_MESSAGE;
}
