import { paypalFetch } from "./paypal-helpers";

/**
 * When a subscriber switches gateway (Stripe -> PayPal, PayPal -> Stripe) or
 * re-subscribes on a new plan, a BRAND NEW gateway subscription is created.
 * The previous one keeps billing unless we explicitly cancel it, which would
 * double-charge the customer.
 *
 * This helper cancels every *other* live gateway subscription belonging to the
 * same account and flags the local rows as canceled. It is idempotent and
 * never throws — billing finalization must not fail because a cleanup call
 * failed.
 */
export type SupersedeArgs = {
  ownerProfileId?: string | null;
  organizationId?: string | null;
  /** Subscription that must survive (the freshly activated one). */
  keepStripeSubscriptionId?: string | null;
  keepPaypalSubscriptionId?: string | null;
};

const LIVE_STATUSES = ["active", "trialing", "past_due", "approval_pending", "incomplete"];

async function cancelPayPal(subscriptionId: string): Promise<boolean> {
  try {
    await paypalFetch(`/v1/billing/subscriptions/${subscriptionId}/cancel`, {
      method: "POST",
      body: JSON.stringify({ reason: "Replaced by a newer subscription" }),
    });
    return true;
  } catch (err: any) {
    // 422 UNPROCESSABLE == already cancelled/expired; treat as success.
    if (err?.status === 422 || err?.status === 404) return true;
    console.warn("[supersede] paypal cancel failed", subscriptionId, err?.message);
    return false;
  }
}

async function cancelStripe(subscriptionId: string): Promise<boolean> {
  try {
    const { getStripe } = await import("./stripe-cards.server");
    const stripe = getStripe();
    await stripe.subscriptions.cancel(subscriptionId);
    return true;
  } catch (err: any) {
    const code = err?.rawType || err?.code;
    if (code === "resource_missing") return true;
    console.warn("[supersede] stripe cancel failed", subscriptionId, err?.message);
    return false;
  }
}

export async function cancelSupersededSubscriptions(
  supabase: any,
  args: SupersedeArgs,
): Promise<void> {
  const keepStripe = args.keepStripeSubscriptionId ?? null;
  const keepPaypal = args.keepPaypalSubscriptionId ?? null;
  if (!keepStripe && !keepPaypal) return;
  if (!args.organizationId && !args.ownerProfileId) return;

  try {
    let query = supabase
      .from("subscriptions")
      .select(
        "id, status, payment_provider, stripe_subscription_id, paypal_subscription_id",
      )
      .in("status", LIVE_STATUSES);
    query = args.organizationId
      ? query.eq("organization_id", args.organizationId)
      : query.eq("owner_profile_id", args.ownerProfileId);

    const { data: rows, error } = await query;
    if (error) {
      console.warn("[supersede] subscriptions lookup failed", error.message);
      return;
    }

    for (const row of (rows ?? []) as any[]) {
      const stripeId = row.stripe_subscription_id as string | null;
      const paypalId = row.paypal_subscription_id as string | null;
      if (keepStripe && stripeId === keepStripe) continue;
      if (keepPaypal && paypalId === keepPaypal) continue;
      if (!stripeId && !paypalId) continue;

      let canceled = false;
      if (paypalId) canceled = (await cancelPayPal(paypalId)) || canceled;
      if (stripeId) canceled = (await cancelStripe(stripeId)) || canceled;
      if (!canceled) continue;

      const { error: updErr } = await supabase
        .from("subscriptions")
        .update({
          status: "canceled",
          canceled_at: new Date().toISOString(),
          cancel_at_period_end: false,
        })
        .eq("id", row.id);
      if (updErr) {
        console.warn("[supersede] local status update failed", updErr.message);
      } else {
        console.log("[supersede] cancelled superseded subscription", {
          id: row.id,
          stripeId,
          paypalId,
        });
      }
    }
  } catch (err) {
    console.warn(
      "[supersede] unexpected failure",
      err instanceof Error ? err.message : String(err),
    );
  }
}
