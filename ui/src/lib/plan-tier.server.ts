// The plan tier of ANOTHER user, for the team-member rule. Same rules as the
// canonical resolver getMyActivePlan (ui/src/lib/plan.functions.ts) and as the
// database's avaloka_meta.plan_tier_of: the latest subscription of the user's
// organisation, else the latest one their profile owns; active when
// trialing/active/trial or inside its trial window; enterprise/professional by
// plan_type or name; anything else free. Server-only (admin client); callers
// expose a boolean at most, never the tier.
import { ACTIVE_SUB_STATUSES, type PlanTier } from "@/lib/plan.functions";

export async function planTierForAuthUser(admin: any, authUserId: string | null | undefined): Promise<PlanTier> {
  if (!authUserId) return "free";
  try {
    const { data: profile } = await admin
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${authUserId},user_id.eq.${authUserId}`)
      .maybeSingle();
    const profileId = (profile?.id ?? authUserId) as string;
    const orgId = (profile?.organization_id ?? null) as string | null;
    const latest = () =>
      admin
        .from("subscriptions")
        .select("status, trial_ends_at, plan:plans(plan_type, name)")
        .order("created_at", { ascending: false })
        .limit(1);
    let row: any = null;
    if (orgId) row = (await latest().eq("organization_id", orgId).maybeSingle()).data ?? null;
    if (!row) row = (await latest().eq("owner_profile_id", profileId).maybeSingle()).data ?? null;
    if (!row) return "free";
    const status = String(row.status ?? "").toLowerCase();
    const trialActive =
      !!row.trial_ends_at &&
      new Date(row.trial_ends_at).getTime() > Date.now() &&
      !["canceled", "cancelled", "unpaid", "incomplete_expired"].includes(status);
    if (!(ACTIVE_SUB_STATUSES.has(status) || status === "trial" || trialActive)) return "free";
    const planType = String(row.plan?.plan_type ?? "").toLowerCase();
    const planName = String(row.plan?.name ?? "").toLowerCase();
    if (planType === "enterprise" || planName.includes("enterprise")) return "enterprise";
    if (planType === "professional" || planType === "pro" || planName.includes("professional")) return "professional";
    return "free";
  } catch {
    return "free";
  }
}

export function isProfessionalOrHigher(tier: PlanTier): boolean {
  return tier === "professional" || tier === "enterprise";
}
