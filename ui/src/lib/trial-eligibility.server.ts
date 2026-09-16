/**
 * Free-trial eligibility.
 *
 * The 15-day free trial is a one-time offer per customer (organization when
 * present, else owner profile). Once ANY subscription row has been created
 * for them — regardless of its current status, plan or payment provider —
 * later plan changes, re-subscribes or payment-method switches must start
 * billing immediately with no trial.
 */
export async function isTrialEligible(
  supabase: any,
  args: { ownerProfileId: string; organizationId?: string | null },
): Promise<boolean> {
  const { ownerProfileId, organizationId } = args;
  try {
    const base = supabase.from("subscriptions").select("id").limit(1);
    const query = organizationId
      ? base.eq("organization_id", organizationId)
      : base.eq("owner_profile_id", ownerProfileId).is("organization_id", null);
    const { data, error } = await query;
    if (error) {
      console.warn("[trial] eligibility lookup failed", error.message);
      // Fail closed: never hand out a second trial when unsure.
      return false;
    }
    return !Array.isArray(data) || data.length === 0;
  } catch (err) {
    console.warn("[trial] eligibility check threw", err);
    return false;
  }
}
