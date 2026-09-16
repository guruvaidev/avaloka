/**
 * Plan-change history.
 *
 * Every time a subscription is created or moved to a different plan
 * (Free -> Professional, Professional -> Enterprise, Enterprise ->
 * Professional, ...) we append a row to `plan_change_log` so the Billing page
 * can show a human-readable history. Never throws — billing flows must not
 * fail because logging failed.
 */

export type PlanChangeArgs = {
  ownerProfileId?: string | null;
  organizationId?: string | null;
  planId?: string | null;
  status?: string | null;
  paymentProvider?: string | null;
  subscriptionId?: string | null;
};

function tierOf(planType?: string | null, planName?: string | null): string {
  const t = String(planType ?? "").toLowerCase();
  const n = String(planName ?? "").toLowerCase();
  if (t.includes("enterprise") || n.includes("enterprise")) return "enterprise";
  if (t.includes("pro") || n.includes("professional")) return "professional";
  return t || n || "free";
}

export async function logPlanChange(
  supabase: any,
  args: PlanChangeArgs,
): Promise<void> {
  try {
    const ownerProfileId = args.ownerProfileId ?? null;
    const organizationId = args.organizationId ?? null;
    if (!ownerProfileId && !organizationId) return;
    if (!args.planId) return;

    const { data: plan } = await supabase
      .from("plans")
      .select("id, plan_type, name")
      .eq("id", args.planId)
      .maybeSingle();
    const toPlan = tierOf(plan?.plan_type, plan?.name);
    const toPlanName = (plan?.name ?? null) as string | null;

    // Last logged entry for this account decides whether anything changed.
    let q = supabase
      .from("plan_change_log")
      .select("to_plan, status")
      .order("created_at", { ascending: false })
      .limit(1);
    q = organizationId
      ? q.eq("organization_id", organizationId)
      : q.eq("owner_profile_id", ownerProfileId);
    const { data: last } = await q.maybeSingle();

    const status = (args.status ?? null) as string | null;
    const canceled = ["canceled", "cancelled", "expired"].includes(
      String(status ?? "").toLowerCase(),
    );
    const effectiveTo = canceled ? "free" : toPlan;

    if (last && last.to_plan === effectiveTo) return;

    const { error } = await supabase.from("plan_change_log").insert({
      owner_profile_id: ownerProfileId,
      organization_id: organizationId,
      from_plan: (last?.to_plan ?? "free") as string,
      to_plan: effectiveTo,
      to_plan_name: canceled ? null : toPlanName,
      status,
      payment_provider: args.paymentProvider ?? null,
      subscription_id: args.subscriptionId ?? null,
    });
    if (error) console.warn("[plan-change-log] insert failed", error.message);
  } catch (err) {
    console.warn(
      "[plan-change-log] unexpected failure",
      err instanceof Error ? err.message : String(err),
    );
  }
}
