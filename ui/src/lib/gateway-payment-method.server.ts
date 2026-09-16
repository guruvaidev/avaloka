import Stripe from "stripe";
import { paypalFetch } from "./paypal-helpers";

export type GatewayPaymentMethod = {
  provider: "stripe" | "paypal";
  brand: string | null;
  last4: string | null;
  exp_month: number | null;
  exp_year: number | null;
  cardholder: string | null;
  label: string | null;
  subscription_status: string | null;
};

function getStripe(): Stripe {
  const key = process.env.STRIPE_TEST_API_KEY || process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("STRIPE_SECRET_KEY is not configured");
  return new Stripe(key, { httpClient: Stripe.createFetchHttpClient() } as any);
}

/**
 * Reads the payment method attached to the caller's ACTIVE subscription
 * directly from the gateway (Stripe / PayPal). Nothing is persisted locally —
 * the gateway is the source of truth.
 */
export async function loadGatewayPaymentMethod(
  supabase: any,
  args: { ownerProfileId: string; organizationId: string | null },
): Promise<GatewayPaymentMethod | null> {
  const select = () =>
    supabase
      .from("subscriptions")
      .select(
        "id, status, payment_provider, stripe_customer_id, stripe_subscription_id, paypal_subscription_id, paypal_payer_id",
      )
      .order("created_at", { ascending: false })
      .limit(1);

  let sub: any = null;
  if (args.organizationId) {
    const { data } = await select().eq("organization_id", args.organizationId).maybeSingle();
    sub = data ?? null;
  }
  // Fallback: profile may not be linked to the org that owns the newest
  // subscription — use the latest subscription owned by this profile.
  if (!sub && args.ownerProfileId) {
    const { data } = await select().eq("owner_profile_id", args.ownerProfileId).maybeSingle();
    sub = data ?? null;
  }

  if (!sub) return null;

  const provider = (sub.payment_provider ??
    (sub.paypal_subscription_id ? "paypal" : sub.stripe_subscription_id ? "stripe" : null)) as
    | "stripe"
    | "paypal"
    | null;
  if (!provider) return null;

  if (provider === "stripe") {
    try {
      const stripe = getStripe();
      let pmId: string | null = null;
      if (sub.stripe_subscription_id) {
        const s = (await stripe.subscriptions.retrieve(sub.stripe_subscription_id)) as any;
        pmId =
          typeof s.default_payment_method === "string"
            ? s.default_payment_method
            : (s.default_payment_method?.id ?? null);
      }
      if (!pmId && sub.stripe_customer_id) {
        const cust = (await stripe.customers.retrieve(sub.stripe_customer_id)) as any;
        const def = cust?.invoice_settings?.default_payment_method;
        pmId = typeof def === "string" ? def : (def?.id ?? null);
        if (!pmId) {
          const list = await stripe.paymentMethods.list({
            customer: sub.stripe_customer_id,
            type: "card",
            limit: 1,
          });
          pmId = list.data[0]?.id ?? null;
        }
      }
      if (!pmId) return null;
      const pm = await stripe.paymentMethods.retrieve(pmId);
      const card = pm.card;
      return {
        provider: "stripe",
        brand: card?.brand ?? null,
        last4: card?.last4 ?? null,
        exp_month: card?.exp_month ?? null,
        exp_year: card?.exp_year ?? null,
        cardholder: pm.billing_details?.name ?? null,
        label: null,
        subscription_status: (sub.status as string) ?? null,
      };
    } catch (err) {
      console.warn("[billing] stripe payment method lookup failed", err);
      return null;
    }
  }

  try {
    if (!sub.paypal_subscription_id) return null;
    const ps = await paypalFetch(`/v1/billing/subscriptions/${sub.paypal_subscription_id}`);
    const payer = ps?.subscriber ?? {};
    const email = payer?.email_address ?? null;
    const name = [payer?.name?.given_name, payer?.name?.surname].filter(Boolean).join(" ") || null;
    const payerId = payer?.payer_id ?? sub.paypal_payer_id ?? null;
    return {
      provider: "paypal",
      brand: "paypal",
      last4: payerId ? String(payerId).slice(-4) : null,
      exp_month: null,
      exp_year: null,
      cardholder: name,
      label: email,
      subscription_status: (ps?.status as string) ?? (sub.status as string) ?? null,
    };
  } catch (err) {
    // The mandate exists even if the PayPal API call fails (token/env issues,
    // rate limits). Still surface PayPal as the saved payment method.
    console.warn("[billing] paypal payment method lookup failed", err);
    return {
      provider: "paypal",
      brand: "paypal",
      last4: sub.paypal_payer_id ? String(sub.paypal_payer_id).slice(-4) : null,
      exp_month: null,
      exp_year: null,
      cardholder: "PayPal",
      label: null,
      subscription_status: (sub.status as string) ?? null,
    };
  }
}
