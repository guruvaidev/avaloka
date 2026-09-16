import { createServerFn } from "@tanstack/react-start";
import Stripe from "stripe";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { assertBillingDetails } from "./billing-details";
import { getAppOrigin } from "./app-origin";
import { paypalFetch } from "./paypal-helpers";



export type BillingInterval = "month" | "year";

export type EnterpriseCheckoutInput = {
  price_id?: string;
  company_name: string;
  billing_address: string;
  vat_number: string;
  interval?: BillingInterval;
  /** Explicit mandate for recurring auto-payments (required). */
  auto_pay_authorized?: boolean;
};


function getStripe(): Stripe {
  const key = process.env.STRIPE_TEST_API_KEY || process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("STRIPE_SECRET_KEY is not configured");
  return new Stripe(key, {
    httpClient: Stripe.createFetchHttpClient(),
  } as any);
}

function resolvePriceId(interval: BillingInterval): string {
  if (interval === "year") {
    const annual = process.env.STRIPE_ENTERPRISE_PRICE_ID_ANNUAL;
    if (!annual) throw new Error("STRIPE_ENTERPRISE_PRICE_ID_ANNUAL is not configured");
    return annual;
  }
  const monthly =
    process.env.STRIPE_ENTERPRISE_PRICE_ID_MONTHLY ||
    process.env.STRIPE_ENTERPRISE_PRICE_ID;
  if (!monthly) throw new Error("STRIPE_ENTERPRISE_PRICE_ID_MONTHLY is not configured");
  return monthly;
}

/**
 * Latest subscription row for the caller. Prefers the org-scoped subscription
 * when the profile has an organization_id, else falls back to the
 * owner_profile-scoped one.
 */
async function loadLatestSubscription(
  supabase: any,
  args: { ownerProfileId: string; organizationId: string | null },
): Promise<Record<string, any> | null> {
  const { ownerProfileId, organizationId } = args;
  const base = supabase
    .from("subscriptions")
    .select(
      "id, owner_profile_id, organization_id, stripe_customer_id, stripe_subscription_id, paypal_subscription_id, paypal_payer_id, status, trial_ends_at, current_period_end, payment_provider",
    )
    .order("created_at", { ascending: false })
    .limit(1);
  const q = organizationId
    ? base.eq("organization_id", organizationId)
    : base.eq("owner_profile_id", ownerProfileId).is("organization_id", null);
  const { data } = await q.maybeSingle();
  return (data ?? null) as Record<string, any> | null;
}


export const createEnterpriseCheckoutSession = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: EnterpriseCheckoutInput) => {
    if (!data || typeof data !== "object") throw new Error("Missing billing info");
    const { company_name, billing_address, vat_number } = assertBillingDetails(data);
    const interval: BillingInterval = data.interval === "year" ? "year" : "month";
    const price_id_raw = typeof data.price_id === "string" ? data.price_id.trim() : "";
    const price_id = price_id_raw.startsWith("price_") ? price_id_raw : undefined;
    return { company_name, billing_address, vat_number, interval, price_id };
  })

  .handler(async ({ context, data }) => {
    const priceId = data.price_id ?? resolvePriceId(data.interval);
    const stripe = getStripe();
    const userId = context.userId;
    const email =
      (context.claims as { email?: string } | undefined)?.email || undefined;

    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;

    // Resolve owner profile + organization (identity only — no billing on profiles/orgs).
    const { data: profile } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();

    const ownerProfileId = (profile?.id ?? userId) as string;
    const organizationId = (profile?.organization_id ?? null) as string | null;

    // Read existing subscription (source of truth for stripe ids).
    const existingSub = await loadLatestSubscription(supabase, {
      ownerProfileId,
      organizationId,
    });
    const existingCustomerId = existingSub?.stripe_customer_id ?? null;
    const existingSubscriptionId = existingSub?.stripe_subscription_id ?? null;

    // If a live subscription is already attached, send them to the Billing Portal
    // instead of double-charging.
    if (existingSubscriptionId) {
      try {
        const existing = await stripe.subscriptions.retrieve(existingSubscriptionId);
        if (
          existing.status === "active" ||
          existing.status === "trialing" ||
          existing.status === "past_due"
        ) {
          if (existingCustomerId) {
            const origin = getAppOrigin();
            const portal = await stripe.billingPortal.sessions.create({
              customer: existingCustomerId,
              return_url: `${origin}/settings?tab=billing`,
            });
            return { url: portal.url, reused: true as const };
          }
        }
      } catch (err) {
        console.warn("[billing] existing subscription lookup failed", err);
      }
    }

    let customerId = existingCustomerId;
    if (!customerId) {
      const customer = await stripe.customers.create({
        email,
        name: data.company_name,
        metadata: { user_id: userId, vat_number: data.vat_number },
      });
      customerId = customer.id;
    } else {
      try {
        await stripe.customers.update(customerId, {
          email,
          name: data.company_name,
          metadata: { user_id: userId, vat_number: data.vat_number },
        });
      } catch (err) {
        console.warn("[billing] stripe customer update failed", err);
      }
    }

    // Note: we intentionally do NOT persist stripe_customer_id here.
    // The Stripe webhook is the sole writer for the subscriptions row.

    const origin = getAppOrigin();
    // The 15-day trial is a one-time offer: only grant it when this customer
    // has never had a subscription before.
    const { isTrialEligible } = await import("./trial-eligibility.server");
    const trialEligible = await isTrialEligible(supabase, {
      ownerProfileId,
      organizationId,
    });
    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      line_items: [{ price: priceId, quantity: 1 }],
      payment_method_collection: "always",
      subscription_data: {
        ...(trialEligible
          ? {
              trial_period_days: 15,
              trial_settings: {
                end_behavior: { missing_payment_method: "cancel" as const },
              },
            }
          : {}),
        metadata: {
          user_id: userId,
          owner_profile_id: ownerProfileId,
          organization_id: organizationId ?? "",
          company_name: data.company_name,
          billing_address: data.billing_address,
          vat_number: data.vat_number,
          plan: "enterprise",
          interval: data.interval,
        },
      },
      metadata: {
        user_id: userId,
        owner_profile_id: ownerProfileId,
        organization_id: organizationId ?? "",
        company_name: data.company_name,
        billing_address: data.billing_address,
        vat_number: data.vat_number,
        plan: "enterprise",
        interval: data.interval,
      },
      success_url: `${origin}/settings?tab=billing&billing=success&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${origin}/settings?tab=billing&billing=cancelled`,
      allow_promotion_codes: true,
    });

    if (!session.url) throw new Error("Stripe did not return a checkout URL");
    return { url: session.url, reused: false as const };
  });

/**
 * "Pay Now during trial" — opens the Stripe Billing Portal for the user's
 * existing subscription so they can attach a payment method WITHOUT ending
 * their trial. Reads the Stripe customer id from `subscriptions`.
 */
export const openBillingPortal = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const stripe = getStripe();
    const userId = context.userId;
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;

    const { data: profile } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();

    const sub = await loadLatestSubscription(supabase, {
      ownerProfileId: (profile?.id ?? userId) as string,
      organizationId: (profile?.organization_id ?? null) as string | null,
    });
    const customerId = sub?.stripe_customer_id ?? null;
    if (!customerId) throw new Error("No Stripe customer on file");

    const origin = getAppOrigin();
    const portal = await stripe.billingPortal.sessions.create({
      customer: customerId,
      return_url: `${origin}/settings?tab=billing`,
    });
    return { url: portal.url };
  });

/**
 * "Setup Payment Method" during an application-managed free trial.
 *
 * Preconditions (app-managed trial):
 *   - subscriptions.status = 'trialing'
 *   - subscriptions.stripe_subscription_id IS NULL (Stripe hasn't taken over yet)
 *
 * Creates a Stripe Customer (once) and a Checkout Session in subscription
 * mode with trial_end pinned to subscriptions.trial_ends_at, so the user is
 * NOT charged until the free trial actually ends.
 */
export const setupTrialPaymentMethod = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: { interval?: BillingInterval; price_id?: string } | undefined) => {
    const interval: BillingInterval = data?.interval === "year" ? "year" : "month";
    const raw = typeof data?.price_id === "string" ? data.price_id.trim() : "";
    const price_id = raw.startsWith("price_") ? raw : undefined;
    return { interval, price_id };
  })
  .handler(async ({ context, data }) => {
    const stripe = getStripe();
    const priceId = data.price_id ?? resolvePriceId(data.interval);
    const userId = context.userId;
    const email =
      (context.claims as { email?: string } | undefined)?.email || undefined;

    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;

    const { data: profile } = await supabase
      .from("profiles")
      .select("id, organization_id, full_name")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();
    if (!profile?.organization_id) throw new Error("No organization found for user");

    const { data: org } = await supabase
      .from("organizations")
      .select("id, name")
      .eq("id", profile.organization_id)
      .maybeSingle();
    if (!org) throw new Error("Organization not found");

    // Latest subscription — must be an app-managed trial (no Stripe subscription yet).
    const sub = await loadLatestSubscription(supabase, {
      ownerProfileId: profile.id as string,
      organizationId: profile.organization_id as string,
    });
    if (!sub) throw new Error("No trial subscription found for this organization");
    if (sub.stripe_subscription_id) {
      throw new Error("Payment method already set up for this organization");
    }

    const { data: customerAccount } = await supabase
      .from("customer_accounts")
      .select("company_name, billing_address, vat_number, billing_email")
      .eq("owner_user_id", profile.id)
      .maybeSingle();

    const companyName = org.name || customerAccount?.company_name || undefined;
    const billingAddress = customerAccount?.billing_address ?? "";
    const vatNumber = customerAccount?.vat_number ?? "";
    const billingEmail = customerAccount?.billing_email || email;

    // Create or reuse the Stripe Customer stashed on the subscription row.
    let customerId: string | null = (sub.stripe_customer_id ?? null) as string | null;
    if (!customerId) {
      const customer = await stripe.customers.create({
        email: billingEmail,
        name: companyName,
        metadata: {
          user_id: userId,
          organization_id: org.id,
          vat_number: vatNumber,
        },
      });
      customerId = customer.id;
      // Persist the customer id on the subscription so Portal reuse works
      // even before Stripe webhooks fire for the new checkout session.
      try {
        await supabase
          .from("subscriptions")
          .update({ stripe_customer_id: customerId })
          .eq("id", sub.id);
      } catch {
        /* non-fatal */
      }
    }

    // Pin trial_end to subscriptions.trial_ends_at so the user is NOT charged
    // until the trial ends. Fall back to 15 days from now if missing.
    const trialEndsAt = sub.trial_ends_at ? new Date(sub.trial_ends_at) : null;
    const nowSec = Math.floor(Date.now() / 1000);
    const minTrialEnd = nowSec + 48 * 60 * 60; // Stripe requires >=48h in future
    let trialEndSec = trialEndsAt
      ? Math.floor(trialEndsAt.getTime() / 1000)
      : nowSec + 15 * 24 * 60 * 60;
    if (trialEndSec < minTrialEnd) trialEndSec = minTrialEnd;

    const origin = getAppOrigin();
    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      line_items: [{ price: priceId, quantity: 1 }],
      payment_method_collection: "always",
      subscription_data: {
        trial_end: trialEndSec,
        trial_settings: {
          end_behavior: { missing_payment_method: "cancel" },
        },
        metadata: {
          user_id: userId,
          owner_profile_id: profile.id,
          organization_id: org.id,
          company_name: companyName ?? "",
          billing_address: billingAddress,
          vat_number: vatNumber,
          plan: "enterprise",
          interval: data.interval,
          flow: "trial_setup_payment_method",
        },
      },
      metadata: {
        user_id: userId,
        owner_profile_id: profile.id,
        organization_id: org.id,
        plan: "enterprise",
        interval: data.interval,
        flow: "trial_setup_payment_method",
      },
      success_url: `${origin}/settings?tab=billing&billing=success&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${origin}/settings?tab=billing&billing=cancelled`,
      allow_promotion_codes: true,
    });

    if (!session.url) throw new Error("Stripe did not return a checkout URL");
    return { url: session.url };
  });

/**
 * Schedule cancellation at the end of the current billing period (or trial).
 * For Stripe: sets `cancel_at_period_end = true` on the Stripe subscription.
 * For PayPal: suspends the subscription so no more payments are taken, and marks
 * the local row for end-of-period cancellation.
 * The user keeps access until the end of the current period. The webhook then
 * finalizes the local status once the period ends.
 */
export const cancelSubscriptionAtPeriodEnd = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const userId = context.userId;
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;

    const { data: profile } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();

    const ownerProfileId = (profile?.id ?? userId) as string;
    const organizationId = (profile?.organization_id ?? null) as string | null;

    const sub = await loadLatestSubscription(supabase, {
      ownerProfileId,
      organizationId,
    });

    const stripeSubId = sub?.stripe_subscription_id ?? null;
    const paypalSubId = sub?.paypal_subscription_id ?? null;
    // Fall back to the stored gateway ids when payment_provider is missing so
    // cancellation still routes to the right gateway.
    const provider = (String(sub?.payment_provider ?? "").toLowerCase() ||
      (paypalSubId ? "paypal" : stripeSubId ? "stripe" : "")) as "stripe" | "paypal" | "";

    if (!sub || (!stripeSubId && !paypalSubId)) {
      throw new Error("No active subscription to cancel");
    }

    const nowIso = new Date().toISOString();
    const canceledAt = sub?.canceled_at ? new Date(sub.canceled_at as string) : null;
    const periodEnd = (sub?.current_period_end ?? sub?.trial_ends_at) as string | null;
    const periodEndMs = periodEnd ? new Date(periodEnd).getTime() : 0;
    const alreadyCanceledAtPeriodEnd = Boolean(sub?.cancel_at_period_end);

    // Common local DB mirroring payload.
    const mirrorUpdate = {
      cancel_at_period_end: true,
      canceled_at: nowIso,
    };

    if (provider === "paypal" && paypalSubId) {
      // PayPal has no native "cancel at period end". Suspend billing now and
      // finalize the actual cancellation in the webhook/cron once the period ends.
      try {
        await paypalFetch(`/v1/billing/subscriptions/${paypalSubId}/suspend`, {
          method: "POST",
          body: JSON.stringify({ reason: "Customer requested cancellation" }),
        });
      } catch (err: any) {
        // Already suspended/cancelled/expired in PayPal is acceptable.
        if (![404, 422].includes(err?.status)) {
          throw new Error(
            `PayPal could not suspend the subscription: ${err?.message ?? "Unknown error"}`,
          );
        }
      }

      // Mark the local row as scheduled for cancellation. Keep status active so
      // the user retains access until the period end.
      try {
        await supabase
          .from("subscriptions")
          .update({ ...mirrorUpdate, status: sub.status ?? "active" })
          .eq("id", sub.id);
      } catch (err) {
        console.warn("[billing] failed to mirror PayPal cancel schedule locally", err);
      }

      if (organizationId) {
        await supabase
          .from("organizations")
          .update({
            canceled_at: nowIso,
          })
          .eq("id", organizationId);
      }


      return {
        cancel_at_period_end: true,
        status: sub.status ?? "active",
        current_period_end: periodEnd,
        payment_provider: "paypal",
      };
    }

    if (provider !== "stripe" || !stripeSubId) {
      throw new Error("No active Stripe or PayPal subscription to cancel");
    }

    const stripe = getStripe();
    const updated = await stripe.subscriptions.update(stripeSubId, {
      cancel_at_period_end: true,
    });

    // Reflect immediately in our table so the UI updates without waiting for
    // the webhook. The webhook remains the source of truth for the final
    // `status = 'canceled'` transition.
    try {
      await supabase
        .from("subscriptions")
        .update({
          ...mirrorUpdate,
          canceled_at: (updated as any).canceled_at
            ? new Date((updated as any).canceled_at * 1000).toISOString()
            : nowIso,
        })
        .eq("stripe_subscription_id", stripeSubId);
    } catch (err) {
      console.warn("[billing] failed to mirror cancel_at_period_end locally", err);
    }

    if (organizationId) {
      await supabase
        .from("organizations")
        .update({
          canceled_at: (updated as any).canceled_at
            ? new Date((updated as any).canceled_at * 1000).toISOString()
            : nowIso,
        })
        .eq("id", organizationId);
    }


    const updatedAny = updated as any;
    return {
      cancel_at_period_end: true,
      status: updatedAny.status,
      current_period_end: updatedAny.current_period_end
        ? new Date(updatedAny.current_period_end * 1000).toISOString()
        : null,
      payment_provider: "stripe",
    };
  });

