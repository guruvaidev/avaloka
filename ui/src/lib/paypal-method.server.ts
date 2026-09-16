import { paypalFetch } from "./paypal-helpers";
import { getAppOrigin } from "./app-origin";
import { resolvePaymentProfile } from "./payment-profile.server";

/**
 * Starts a PayPal re-authorization for an EXISTING subscriber who just wants to
 * switch the payment method. Billing details (company / address / VAT) were
 * captured on the first purchase and are reused from customer_accounts — the
 * user is never asked for them again.
 */
export async function startPayPalMethodChange(
  supabase: any,
  userId: string,
  claims?: Record<string, any> | null,
): Promise<{ approveUrl: string; subscriptionId: string }> {
  const { ownerProfileId, organizationId } = await resolvePaymentProfile(
    supabase,
    userId,
    claims,
  );

  // Current subscription → the plan it is billed on.
  const base = supabase
    .from("subscriptions")
    .select(
      "id, status, current_period_end, trial_ends_at, plan:plans(id, name, paypal_plan_id)",
    )
    .order("created_at", { ascending: false })
    .limit(1);
  const q = organizationId
    ? base.eq("organization_id", organizationId)
    : base.eq("owner_profile_id", ownerProfileId).is("organization_id", null);
  const { data: sub } = await q.maybeSingle();

  const planId: string | null =
    (sub?.plan?.paypal_plan_id as string | undefined) ||
    process.env.PAYPAL_PLAN_ID ||
    null;

  if (!planId) {
    throw new Error(
      "PayPal is not configured for your current plan. Please contact support or choose Stripe.",
    );
  }

  // The period the customer already paid for (or the running free trial) must
  // be honoured: start billing the replacement subscription only when the
  // current period ends, so switching method never charges twice.
  const periodEndRaw = (sub?.trial_ends_at ?? sub?.current_period_end) as
    | string
    | null
    | undefined;
  const periodEnd = periodEndRaw ? new Date(periodEndRaw) : null;
  // PayPal requires start_time to be in the future (allow a small buffer).
  const minStart = Date.now() + 10 * 60 * 1000;
  const startTime =
    periodEnd && periodEnd.getTime() > minStart ? periodEnd.toISOString() : null;


  // Reuse the billing details captured at first purchase.
  const { data: account } = await supabase
    .from("customer_accounts")
    .select("company_name, billing_address, vat_number, billing_email")
    .eq("owner_user_id", ownerProfileId)
    .maybeSingle();

  const email =
    (account?.billing_email as string | undefined) ||
    (claims?.email as string | undefined) ||
    undefined;

  const displayName =
    ((claims?.full_name as string | undefined) ||
      (claims?.name as string | undefined) ||
      (account?.company_name as string | undefined) ||
      "").trim();
  const parts = displayName ? displayName.split(/\s+/) : [];
  const givenName = parts[0] || "Avaloka";
  const surname = parts.slice(1).join(" ") || "Customer";

  const origin = getAppOrigin();
  const returnUrl = `${origin}/settings?tab=billing&billing=success&provider=paypal&change=1`;
  const cancelUrl = `${origin}/settings?tab=billing&billing=cancelled&provider=paypal`;

  let created: any;
  try {
    created = await paypalFetch("/v1/billing/subscriptions", {
      method: "POST",
      body: JSON.stringify({
        plan_id: planId,
        custom_id: ownerProfileId,
        ...(startTime ? { start_time: startTime } : {}),

        subscriber: {
          name: { given_name: givenName, surname },
          ...(email ? { email_address: email } : {}),
        },
        application_context: {
          brand_name: "Avaloka",
          locale: "en-US",
          user_action: "SUBSCRIBE_NOW",
          shipping_preference: "NO_SHIPPING",
          payment_method: {
            payer_selected: "PAYPAL",
            payee_preferred: "IMMEDIATE_PAYMENT_REQUIRED",
          },
          return_url: returnUrl,
          cancel_url: cancelUrl,
        },
      }),
    });
  } catch (err: any) {
    if (err?.status === 404 || err?.paypal?.name === "RESOURCE_NOT_FOUND") {
      throw new Error(
        `The PayPal plan configured for your subscription (${planId}) does not exist in this PayPal environment. Please contact support.`,
      );
    }
    throw err;
  }

  const approveUrl: string | undefined = (created?.links ?? []).find(
    (l: any) => l.rel === "approve",
  )?.href;
  if (!approveUrl) throw new Error("PayPal did not return an approve URL");

  // Stash the pending subscription so the finalizer can pick it up.
  try {
    if (organizationId && created?.id) {
      await supabase
        .from("organizations")
        .update({
          payment_provider: "paypal",
          paypal_subscription_id: created.id,
          subscription_status: "approval_pending",
        })
        .eq("id", organizationId);
    }
  } catch (err) {
    console.warn("[paypal] failed to stash pending method-change subscription", err);
  }

  return { approveUrl, subscriptionId: created.id as string };
}
