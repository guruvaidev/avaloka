import { createFileRoute } from "@tanstack/react-router";
import Stripe from "stripe";

const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Stripe-Signature",
};

function textResp(body: string, status = 200) {
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/plain", ...CORS },
  });
}

function getStripe(): Stripe {
  const key = process.env.STRIPE_TEST_API_KEY || process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("STRIPE_SECRET_KEY missing");
  return new Stripe(key, {
    httpClient: Stripe.createFetchHttpClient(),
  } as any);
}

const ACTIVE_STATUSES = new Set(["trialing", "active"]);

function isoOrNull(ts: number | null | undefined): string | null {
  return typeof ts === "number" && ts > 0 ? new Date(ts * 1000).toISOString() : null;
}

function addInterval(fromMs: number, interval: string | undefined, count: number): string {
  const d = new Date(fromMs);
  const n = count || 1;
  switch (interval) {
    case "day": d.setUTCDate(d.getUTCDate() + n); break;
    case "week": d.setUTCDate(d.getUTCDate() + 7 * n); break;
    case "month": d.setUTCMonth(d.getUTCMonth() + n); break;
    case "year": d.setUTCFullYear(d.getUTCFullYear() + n); break;
    default: d.setUTCDate(d.getUTCDate() + 30);
  }
  return d.toISOString();
}

function resolveLicenseExpiresAt(subscription: Stripe.Subscription): {
  value: string;
  source: string;
} {
  const subAny = subscription as any;
  const item = subscription.items?.data?.[0] as any;

  const topCpe = subAny.current_period_end;
  if (typeof topCpe === "number" && topCpe > 0) {
    return { value: new Date(topCpe * 1000).toISOString(), source: "top" };
  }
  const itemCpe = item?.current_period_end;
  if (typeof itemCpe === "number" && itemCpe > 0) {
    return { value: new Date(itemCpe * 1000).toISOString(), source: "items" };
  }
  if (typeof subscription.trial_end === "number" && subscription.trial_end > 0) {
    return { value: new Date(subscription.trial_end * 1000).toISOString(), source: "trial_end" };
  }
  const anchor = subAny.billing_cycle_anchor;
  const recurring = item?.price?.recurring;
  if (typeof anchor === "number" && anchor > 0 && recurring?.interval) {
    return {
      value: addInterval(anchor * 1000, recurring.interval, recurring.interval_count ?? 1),
      source: "computed",
    };
  }
  return {
    value: addInterval(Date.now(), recurring?.interval, recurring?.interval_count ?? 1),
    source: "fallback",
  };
}

/**
 * Resolve the `plans.id` corresponding to a Stripe price id.
 */
async function resolvePlanId(
  supabase: any,
  stripePriceId: string | null | undefined,
): Promise<string | null> {
  if (!stripePriceId) return null;
  const { data } = await supabase
    .from("plans")
    .select("id")
    .eq("stripe_price_id", stripePriceId)
    .limit(1)
    .maybeSingle();
  return (data?.id ?? null) as string | null;
}

/**
 * Resolve owner_profile_id + organization_id from Stripe subscription metadata.
 * Falls back to looking up existing subscriptions or the profiles table.
 */
async function resolveOwners(
  supabase: any,
  metaSources: Array<Record<string, any> | null | undefined>,
  fallbackSubscriptionId: string | null,
): Promise<{ ownerProfileId: string | null; organizationId: string | null }> {
  const merged: Record<string, any> = {};
  for (const m of metaSources) if (m) Object.assign(merged, m);

  let ownerProfileId = (merged.owner_profile_id as string | undefined) || null;
  let organizationId = (merged.organization_id as string | undefined) || null;
  const userId = (merged.user_id as string | undefined) || null;

  if (organizationId === "") organizationId = null;
  if (ownerProfileId === "") ownerProfileId = null;

  // If owner_profile_id wasn't stamped explicitly, derive it from user_id.
  if (!ownerProfileId && userId) {
    const { data: profile } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();
    if (profile?.id) {
      ownerProfileId = profile.id as string;
      if (!organizationId) organizationId = (profile.organization_id as string | null) ?? null;
    }
  }

  // As a last resort, if the row already exists (updated events), keep prior owners.
  if ((!ownerProfileId || !organizationId) && fallbackSubscriptionId) {
    const { data: existing } = await supabase
      .from("subscriptions")
      .select("owner_profile_id, organization_id")
      .eq("stripe_subscription_id", fallbackSubscriptionId)
      .maybeSingle();
    if (existing) {
      ownerProfileId = ownerProfileId ?? (existing.owner_profile_id as string | null);
      organizationId = organizationId ?? (existing.organization_id as string | null);
    }
  }

  return { ownerProfileId, organizationId };
}

/**
 * Build a subscriptions upsert payload from a Stripe Subscription.
 */
async function buildSubscriptionPayload(
  supabase: any,
  subscription: Stripe.Subscription,
  extraMeta?: Record<string, any> | null,
): Promise<Record<string, any> | null> {
  const subAny = subscription as any;
  const item = subscription.items?.data?.[0] as any;
  const priceId = (item?.price?.id ?? null) as string | null;
  const planId = await resolvePlanId(supabase, priceId);

  const { ownerProfileId, organizationId } = await resolveOwners(
    supabase,
    [subscription.metadata as any, extraMeta ?? null],
    subscription.id,
  );

  if (!ownerProfileId) {
    console.warn(
      "[stripe-webhook] cannot resolve owner_profile_id for subscription",
      subscription.id,
    );
    return null;
  }
  if (!planId) {
    console.warn(
      "[stripe-webhook] cannot resolve plan_id for stripe_price_id",
      priceId,
    );
    return null;
  }

  const trialEnd = isoOrNull(subscription.trial_end);
  const trialStart = isoOrNull(subscription.trial_start);
  const cpe =
    isoOrNull(subAny.current_period_end) ||
    isoOrNull(item?.current_period_end) ||
    trialEnd ||
    null;
  const cps =
    isoOrNull(subAny.current_period_start) ||
    isoOrNull(item?.current_period_start) ||
    isoOrNull(subAny.start_date) ||
    null;

  const customerId =
    typeof subscription.customer === "string"
      ? subscription.customer
      : (subscription.customer as any)?.id ?? null;

  return {
    owner_profile_id: ownerProfileId,
    organization_id: organizationId,
    plan_id: planId,
    payment_provider: "stripe",
    stripe_customer_id: customerId,
    stripe_subscription_id: subscription.id,
    status: subscription.status,
    trial_starts_at: trialStart,
    trial_ends_at: trialEnd,
    current_period_start: cps,
    current_period_end: cpe,
    cancel_at_period_end: Boolean(subscription.cancel_at_period_end),
    canceled_at: isoOrNull(subscription.canceled_at as any),
  };
}

async function upsertSubscription(
  supabase: any,
  payload: Record<string, any>,
): Promise<void> {
  const { attachSubscriptionSerial } = await import("@/lib/subscription-serial.server");
  await attachSubscriptionSerial(supabase, payload, {
    column: "stripe_subscription_id",
    value: payload.stripe_subscription_id,
  });

  let { error } = await supabase
    .from("subscriptions")
    .upsert(payload, { onConflict: "stripe_subscription_id" });
  if (error && /serial_number/i.test(error.message ?? "")) {
    // Column not present yet — retry without it.
    delete payload.serial_number;
    ({ error } = await supabase
      .from("subscriptions")
      .upsert(payload, { onConflict: "stripe_subscription_id" }));
  }
  if (error) {
    console.error("[stripe-webhook] subscriptions upsert failed:", error.message);
    throw new Error("subscriptions upsert failed: " + error.message);
  }

  {
    const { logPlanChange } = await import("@/lib/plan-change-log.server");
    await logPlanChange(supabase, {
      ownerProfileId: payload.owner_profile_id ?? null,
      organizationId: payload.organization_id ?? null,
      planId: payload.plan_id ?? null,
      status: payload.status ?? null,
      paymentProvider: "stripe",
      subscriptionId: payload.stripe_subscription_id ?? null,
    });
  }


  if (
    payload.organization_id &&
    (payload.status === "active" || payload.status === "trialing")
  ) {
    const { seedOrganizationDefaults } = await import(
      "@/lib/org-defaults.server"
    );
    await seedOrganizationDefaults(supabase, payload.organization_id as string);
  }

  // A newly live Stripe subscription supersedes any previous gateway
  // subscription (older Stripe sub or a PayPal one) — cancel it so the
  // customer is never billed twice.
  if (payload.status === "active" || payload.status === "trialing") {
    const { cancelSupersededSubscriptions } = await import(
      "@/lib/subscription-supersede.server"
    );
    await cancelSupersededSubscriptions(supabase, {
      ownerProfileId: payload.owner_profile_id ?? null,
      organizationId: payload.organization_id ?? null,
      keepStripeSubscriptionId: payload.stripe_subscription_id ?? null,
    });
  }
}


async function handleCheckoutCompleted(
  supabase: any,
  stripe: Stripe,
  session: Stripe.Checkout.Session,
) {
  if (session.mode !== "subscription") return;
  const subscriptionId =
    typeof session.subscription === "string"
      ? session.subscription
      : session.subscription?.id;
  if (!subscriptionId) return;

  const subscription = await stripe.subscriptions.retrieve(subscriptionId);
  const meta = { ...(session.metadata ?? {}), ...(subscription.metadata ?? {}) };
  const userId = meta.user_id as string | undefined;
  if (!userId) {
    console.error("[stripe-webhook] missing user_id in metadata");
    throw new Error("missing user_id metadata");
  }

  const companyName = (meta.company_name as string) || "Enterprise Organization";
  const billingAddress = (meta.billing_address as string) || null;
  const vatNumber = (meta.vat_number as string) || null;

  // Resolve/ensure profile row (auto-create if the handle_new_user trigger
  // hasn't populated one yet).
  let profile: { id: string; organization_id: string | null } | null = null;
  {
    const { data } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();
    profile = (data as any) ?? null;
  }
  if (!profile?.id) {
    const { data } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .eq("id", userId)
      .maybeSingle();
    profile = (data as any) ?? null;
  }
  if (!profile?.id) {
    console.warn("[stripe-webhook] no profile for user_id, creating", userId);
    const { data: created, error: profErr } = await supabase
      .from("profiles")
      .insert({ id: userId })
      .select("id, organization_id")
      .single();
    if (profErr) throw new Error("profile create failed: " + profErr.message);
    profile = created as any;
  }
  const ownerProfileId = profile!.id as string;

  // customer_accounts (unchanged)
  let customerAccountId: string | undefined;
  {
    const { data: ca } = await supabase
      .from("customer_accounts")
      .select("id")
      .eq("owner_user_id", ownerProfileId)
      .maybeSingle();
    if (ca?.id) {
      customerAccountId = ca.id;
      await supabase
        .from("customer_accounts")
        .update({
          company_name: companyName,
          billing_address: billingAddress,
          vat_number: vatNumber,
        })
        .eq("id", ca.id);
    } else {
      const { data: created, error } = await supabase
        .from("customer_accounts")
        .insert({
          owner_user_id: ownerProfileId,
          company_name: companyName,
          billing_address: billingAddress,
          vat_number: vatNumber,
          plan_type: "enterprise",
        })
        .select("id")
        .single();
      if (error) throw new Error("customer_accounts insert failed: " + error.message);
      customerAccountId = created.id as string;
    }
  }

  // organizations (identity only — name, no billing writes)
  let organizationId = profile!.organization_id as string | null;
  if (organizationId) {
    const { data: orgRow } = await supabase
      .from("organizations")
      .select("id")
      .eq("id", organizationId)
      .maybeSingle();
    if (!orgRow) organizationId = null;
  }
  if (!organizationId) {
    const { data: orgCreated, error: orgErr } = await supabase
      .from("organizations")
      .insert({
        name: companyName,
        owner_profile_id: ownerProfileId,
      })
      .select("id")
      .single();
    if (orgErr) throw new Error("organizations insert failed: " + orgErr.message);
    organizationId = orgCreated.id as string;
  } else {
    // Keep the org name in sync with Checkout metadata.
    await supabase
      .from("organizations")
      .update({ name: companyName })
      .eq("id", organizationId);
  }

  // Seed defaults (division, roles, owner app_user) — idempotent; also
  // backfills organizations created before this logic existed.
  {
    const { seedOrganizationDefaults } = await import("@/lib/org-defaults.server");
    await seedOrganizationDefaults(supabase, organizationId!);
  }

  // Link profile to org + grant admin. NO billing writes on profiles.
  await supabase
    .from("profiles")
    .update({ organization_id: organizationId, is_admin: true })
    .eq("id", ownerProfileId);

  // Upsert the subscription row — the single source of truth.
  const payload = await buildSubscriptionPayload(supabase, subscription, {
    owner_profile_id: ownerProfileId,
    organization_id: organizationId,
    user_id: userId,
  });
  if (payload) {
    await upsertSubscription(supabase, payload);
  }

  // Enterprise license upsert (unchanged behavior)
  const resolvedExpires = resolveLicenseExpiresAt(subscription);
  const licenseExpiresAt = resolvedExpires.value;
  const status = subscription.status;
  const paymentStatus = status === "trialing" ? "pending" : status === "active" ? "paid" : "unpaid";
  const paidAt = status === "active" ? new Date().toISOString() : null;

  const { data: existingLic } = await supabase
    .from("enterprise_licenses")
    .select("id")
    .eq("customer_account_id", customerAccountId)
    .order("issued_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  const OPTIONAL_COLS = ["paid_at", "organization_id", "payment_status", "payment_amount"];
  async function runWithColumnFallback(
    op: "insert" | "update",
    payload: Record<string, any>,
    idForUpdate?: string,
  ): Promise<{ error: any; data?: any }> {
    const attempt: Record<string, any> = { ...payload };
    for (let i = 0; i < 8; i++) {
      const q = supabase.from("enterprise_licenses");
      const res =
        op === "insert"
          ? await q.insert(attempt).select("id")
          : await q.update(attempt).eq("id", idForUpdate!).select("id");
      if (!res.error) {
        if (op === "insert" && Array.isArray(res.data) && res.data.length === 0) {
          return { error: new Error("insert returned no rows (RLS/GRANT?)") };
        }
        return { error: null, data: res.data };
      }
      const msg = res.error.message || "";
      const looksLikeSchema =
        /column|schema cache|Could not find|violates check constraint/i.test(msg);
      if (!looksLikeSchema) return { error: res.error };
      const stripped = OPTIONAL_COLS.find(
        (c) => c in attempt && msg.toLowerCase().includes(c),
      );
      if (!stripped) return { error: res.error };
      delete attempt[stripped];
    }
    return { error: new Error("exhausted column-fallback retries") };
  }

  if (existingLic?.id) {
    const { error: updErr } = await runWithColumnFallback(
      "update",
      {
        status: "active",
        expires_at: licenseExpiresAt,
        payment_status: paymentStatus,
        paid_at: paidAt,
      },
      existingLic.id,
    );
    if (updErr) throw new Error("license update failed: " + updErr.message);
  } else {
    const serial = `ENT-${subscriptionId.slice(-12).toUpperCase()}`;
    const { error } = await runWithColumnFallback("insert", {
      customer_account_id: customerAccountId,
      organization_id: organizationId,
      serial_number: serial,
      status: "active",
      issued_at: new Date().toISOString(),
      expires_at: licenseExpiresAt,
      created_by: ownerProfileId,
      payment_amount: 0,
      payment_status: paymentStatus,
      paid_at: paidAt,
    });
    if (error) throw new Error("license insert failed: " + error.message);
  }
}

async function handleSubscriptionUpdated(
  supabase: any,
  _stripe: Stripe,
  subscription: Stripe.Subscription,
) {
  const payload = await buildSubscriptionPayload(supabase, subscription);
  if (!payload) return;
  await upsertSubscription(supabase, payload);
}

async function handleInvoicePaymentFailed(
  supabase: any,
  invoice: Stripe.Invoice,
) {
  const invAny = invoice as any;
  const subRef = invAny.subscription;
  const subId = typeof subRef === "string" ? subRef : subRef?.id;
  if (!subId) return;
  const { error } = await supabase
    .from("subscriptions")
    .update({ status: "past_due" })
    .eq("stripe_subscription_id", subId);
  if (error) console.error("[stripe-webhook] past_due update failed", error.message);
}

async function handleInvoicePaid(
  supabase: any,
  stripe: Stripe,
  invoice: Stripe.Invoice,
) {
  const invAny = invoice as any;
  const subRef = invAny.subscription;
  const subId = typeof subRef === "string" ? subRef : subRef?.id;
  if (!subId) return;

  // Look up the subscription row so we can attach the invoice to the org.
  const { data: subRow } = await supabase
    .from("subscriptions")
    .select("id, organization_id, owner_profile_id")
    .eq("stripe_subscription_id", subId)
    .maybeSingle();
  const organizationId = subRow?.organization_id ?? null;
  const ownerProfileId = subRow?.owner_profile_id ?? null;

  const amountCents = Number(invoice.amount_paid ?? invoice.total ?? 0) || 0;
  const paidAt =
    typeof invAny.status_transitions?.paid_at === "number"
      ? new Date(invAny.status_transitions.paid_at * 1000).toISOString()
      : invAny.created
      ? new Date(invAny.created * 1000).toISOString()
      : new Date().toISOString();
  const period = invoice.lines?.data?.[0]?.period;
  const record: Record<string, any> = {
    organization_id: organizationId,
    owner_profile_id: ownerProfileId,
    provider: "stripe",
    provider_invoice_id: invoice.id,
    invoice_number: invoice.number ?? null,
    amount_cents: amountCents,
    currency: (invoice.currency || "usd").toUpperCase(),
    status: "paid",
    paid_at: paidAt,
    period_start: period?.start ? new Date(period.start * 1000).toISOString() : null,
    period_end: period?.end ? new Date(period.end * 1000).toISOString() : null,
    hosted_url: invoice.hosted_invoice_url ?? null,
  };
  let { error } = await supabase
    .from("billing_invoices")
    .upsert(record, { onConflict: "provider,provider_invoice_id" });
  if (error && /owner_profile_id/i.test(error.message || "")) {
    delete record.owner_profile_id;
    ({ error } = await supabase
      .from("billing_invoices")
      .upsert(record, { onConflict: "provider,provider_invoice_id" }));
  }
  if (error) console.error("[stripe-webhook] invoice upsert failed", error.message);

  // Persist the card used for this subscription so Billing lists it.
  try {
    if (ownerProfileId && invoice.customer) {
      const sub = await stripe.subscriptions.retrieve(subId, {
        expand: ["default_payment_method"],
      });
      const pm: any =
        sub.default_payment_method && typeof sub.default_payment_method !== "string"
          ? sub.default_payment_method
          : null;
      if (pm?.card) {
        const customerId =
          typeof invoice.customer === "string" ? invoice.customer : invoice.customer.id;
        const { data: existingPm } = await supabase
          .from("payment_methods")
          .select("id")
          .eq("owner_profile_id", ownerProfileId)
          .eq("provider", "stripe")
          .eq("provider_payment_method_id", pm.id)
          .maybeSingle();
        const pmPayload: Record<string, any> = {
          owner_profile_id: ownerProfileId,
          organization_id: organizationId,
          provider: "stripe",
          provider_customer_id: customerId,
          provider_payment_method_id: pm.id,
          provider_subscription_id: subId,
          brand: pm.card.brand ?? null,
          last4: pm.card.last4 ?? null,
          exp_month: pm.card.exp_month ?? null,
          exp_year: pm.card.exp_year ?? null,
          cardholder: pm.billing_details?.name ?? null,
          status: "active",
          is_default: true,
          metadata: { funding: pm.card.funding ?? null, country: pm.card.country ?? null },
        };
        await supabase
          .from("payment_methods")
          .update({ status: "inactive", is_default: false })
          .eq("owner_profile_id", ownerProfileId)
          .neq("provider_payment_method_id", pm.id);
        if (existingPm?.id) {
          await supabase.from("payment_methods").update(pmPayload).eq("id", existingPm.id);
        } else {
          await supabase.from("payment_methods").insert(pmPayload);
        }
      }
    }
  } catch (err) {
    console.warn("[stripe-webhook] payment_methods sync failed", err);
  }


  // Sync the subscription row from Stripe (advances current_period_end)
  // and extend the enterprise license.
  try {
    const sub = await stripe.subscriptions.retrieve(subId);
    const payload = await buildSubscriptionPayload(supabase, sub);
    if (payload) await upsertSubscription(supabase, payload);
    if (organizationId) {
      const newExpires =
        payload?.current_period_end || resolveLicenseExpiresAt(sub).value;
      await supabase
        .from("enterprise_licenses")
        .update({
          status: "active",
          expires_at: newExpires,
          payment_status: "paid",
          paid_at: paidAt,
        })
        .eq("organization_id", organizationId);
    }
  } catch (err) {
    console.warn("[stripe-webhook] license extension failed", err);
  }

  // Silence unused vars — ACTIVE_STATUSES is exported below for other consumers if needed.
  void ACTIVE_STATUSES;
}

export const Route = createFileRoute("/api/public/stripe-webhook")({
  server: {
    handlers: {
      OPTIONS: async () => new Response(null, { status: 204, headers: CORS }),
      POST: async ({ request }) => {
        const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
        if (!webhookSecret) return textResp("STRIPE_WEBHOOK_SECRET missing", 500);

        const signature = request.headers.get("stripe-signature");
        if (!signature) return textResp("Missing stripe-signature", 400);

        const rawBody = await request.text();
        const stripe = getStripe();

        let event: Stripe.Event;
        try {
          event = await stripe.webhooks.constructEventAsync(
            rawBody,
            signature,
            webhookSecret,
          );
        } catch (err) {
          const message = err instanceof Error ? err.message : "invalid signature";
          console.error("[stripe-webhook] signature verify failed:", message);
          return textResp(`Webhook signature error: ${message}`, 400);
        }

        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
        const supabase = supabaseAdmin as any;

        try {
          switch (event.type) {
            case "checkout.session.completed":
              await handleCheckoutCompleted(
                supabase,
                stripe,
                event.data.object as Stripe.Checkout.Session,
              );
              break;
            case "customer.subscription.created":
            case "customer.subscription.updated":
            case "customer.subscription.deleted":
            case "customer.subscription.trial_will_end":
              await handleSubscriptionUpdated(
                supabase,
                stripe,
                event.data.object as Stripe.Subscription,
              );
              break;
            case "invoice.paid":
            case "invoice.payment_succeeded":
              await handleInvoicePaid(
                supabase,
                stripe,
                event.data.object as Stripe.Invoice,
              );
              break;
            case "invoice.payment_failed":
              await handleInvoicePaymentFailed(
                supabase,
                event.data.object as Stripe.Invoice,
              );
              break;

            default:
              // ignore
              break;
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "handler failed";
          console.error("[stripe-webhook] handler error:", message);
          return textResp(`Webhook handler error: ${message}`, 500);
        }

        return textResp("ok", 200);
      },
    },
  },
});
