import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

// Statuses that count as an active/paid subscription.
// NOTE: "past_due" is intentionally excluded — a failed payment revokes access
// immediately (no grace period).
export const ACTIVE_SUB_STATUSES = new Set([
  "trialing",
  "active",
]);

export type PlanTier = "free" | "professional" | "enterprise";
export type PlanScope = "personal" | "organization";

export type ActivePlan = {
  plan: PlanTier;
  scope: PlanScope;
  organizationId: string | null;
  ownerProfileId: string;
  /** True when this user owns the personal/organization subscription scope. */
  isOrgOwner: boolean;
  subscription: {
    id: string;
    status: string;
    current_period_end: string | null;
    trial_ends_at: string | null;
    payment_provider: string | null;
  } | null;
  planRow: {
    id: string;
    plan_type: string;
    name: string | null;
    billing_interval: string | null;
    price: number | null;
  } | null;
};

/**
 * Canonical plan resolver.
 *
 * Rules:
 * 1. profiles.organization_id IS NULL → look up latest subscriptions row
 *    where owner_profile_id = profile.id AND organization_id IS NULL.
 *    Active + plan_type='professional' → Professional, else Free.
 * 2. profiles.organization_id IS NOT NULL → look up latest subscriptions
 *    row for that organization_id. Active + plan_type='enterprise' →
 *    Enterprise, else Free.
 */
export const getMyActivePlan = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }): Promise<ActivePlan> => {
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/client.server"
    );
    const supabase = supabaseAdmin as any;
    const userId = context.userId as string;

    // Resolve profile (owner_profile_id + organization_id).
    const { data: profile } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();

    const ownerProfileId = (profile?.id ?? userId) as string;
    const organizationId = (profile?.organization_id ?? null) as string | null;
    const scope: PlanScope = organizationId ? "organization" : "personal";

    // Query the latest matching subscription, joining plans for tier info.
    const select = () =>
      supabase
        .from("subscriptions")
        .select(
          "id, status, current_period_end, trial_ends_at, payment_provider, plan:plans(id, plan_type, name, billing_interval, price)",
        )
        .order("created_at", { ascending: false })
        .limit(1);

    // Server-side ownership truth: personal scope is always self-owned;
    // org scope compares organizations.owner_profile_id to this profile.
    let isOrgOwner = !organizationId;
    if (organizationId) {
      const { data: org } = await supabase
        .from("organizations")
        .select("owner_profile_id")
        .eq("id", organizationId)
        .maybeSingle();
      const ownerId = (org?.owner_profile_id ?? null) as string | null;
      isOrgOwner = !!ownerId && (ownerId === ownerProfileId || ownerId === userId);
    }

    let row: any = null;
    if (organizationId) {
      const { data, error } = await select()
        .eq("organization_id", organizationId)
        .maybeSingle();
      if (error) console.warn("[plan] org subscription lookup failed", error.message);
      row = data ?? null;
    }
    // Fallback: personal-scope subscriptions (e.g. Professional) are owned by
    // the profile and may carry no organization_id even when the profile is
    // linked to an org. Mirrors the billing page lookup.
    if (!row) {
      const { data, error } = await select()
        .eq("owner_profile_id", ownerProfileId)
        .maybeSingle();
      if (error) console.warn("[plan] owner subscription lookup failed", error.message);
      row = data ?? null;
    }

    const subscription = row
      ? {
          id: row.id,
          status: row.status,
          current_period_end: row.current_period_end ?? null,
          trial_ends_at: row.trial_ends_at ?? null,
          payment_provider: row.payment_provider ?? null,
        }
      : null;
    const planRow = (row?.plan ?? null) as ActivePlan["planRow"];

    // A subscription still inside its trial window counts as active even when
    // the provider status hasn't flipped to "trialing" yet.
    const trialActive =
      !!subscription?.trial_ends_at &&
      new Date(subscription.trial_ends_at).getTime() > Date.now() &&
      !["canceled", "cancelled", "unpaid", "incomplete_expired"].includes(
        String(subscription?.status ?? "").toLowerCase(),
      );
    const isActive = subscription
      ? ACTIVE_SUB_STATUSES.has(subscription.status) ||
        String(subscription.status).toLowerCase() === "trial" ||
        trialActive
      : false;
    const planType = String(planRow?.plan_type ?? "").toLowerCase();
    const planName = String(planRow?.name ?? "").toLowerCase();

    let plan: PlanTier = "free";
    if (isActive) {
      if (planType === "enterprise" || planName.includes("enterprise")) {
        plan = "enterprise";
      } else if (
        planType === "professional" ||
        planType === "pro" ||
        planName.includes("professional")
      ) {
        plan = "professional";
      }
    }

    // The support desk account is always treated as Enterprise and never
    // requires a subscription. Scoped strictly to that single email.
    const claimEmail = String(
      (context.claims as any)?.email ?? (context.claims as any)?.user_metadata?.email ?? "",
    ).toLowerCase();
    if (claimEmail === "support@avaloka.ai") {
      plan = "enterprise";
    }

    return {


      plan,
      scope,
      organizationId,
      ownerProfileId,
      isOrgOwner,
      subscription,
      planRow,
    };
  });
