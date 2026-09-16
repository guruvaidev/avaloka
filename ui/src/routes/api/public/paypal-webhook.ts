import { createFileRoute } from "@tanstack/react-router";
import { getPayPalAccessToken, paypalFetch, getPayPalBase } from "@/lib/paypal-helpers";

const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Paypal-Transmission-Id, Paypal-Transmission-Time, Paypal-Transmission-Sig, Paypal-Cert-Url, Paypal-Auth-Algo",
};

function textResp(body: string, status = 200) {
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/plain", ...CORS },
  });
}

function subActiveFromStatus(status: string | null | undefined): boolean {
  const s = String(status || "").toUpperCase();
  return s === "ACTIVE" || s === "APPROVAL_PENDING" || s === "APPROVED";
}

function mapStatus(paypal: string | null | undefined): string {
  const s = String(paypal || "").toUpperCase();
  if (s === "ACTIVE") return "active";
  if (s === "APPROVAL_PENDING" || s === "APPROVED") return "trialing";
  if (s === "SUSPENDED") return "past_due";
  if (s === "CANCELLED") return "canceled";
  if (s === "EXPIRED") return "expired";
  return s.toLowerCase() || "unknown";
}

// custom_id in new builds is just the userId/profile id (UUID). Older builds
// stored a truncated JSON blob — support both for backward compatibility.
function parseCustomId(raw: string | null | undefined): {
  user_id?: string;
  company_name?: string;
  billing_address?: string;
  vat_number?: string;
} {
  if (!raw) return {};
  const s = String(raw).trim();
  // Plain UUID / short id path (new format).
  if (/^[0-9a-f-]{8,}$/i.test(s) && !s.startsWith("{")) {
    return { user_id: s };
  }
  try {
    const j = JSON.parse(s);
    return {
      user_id: j.u ?? j.user_id,
      company_name: j.c ?? j.company_name,
      billing_address: j.a ?? j.billing_address,
      vat_number: j.v ?? j.vat_number,
    };
  } catch {
    // Legacy truncated JSON — try to recover the user id from the leading
    // `{"u":"<uuid>"` fragment before the truncation cut.
    const m = s.match(/"u"\s*:\s*"([0-9a-f-]{8,36})/i);
    if (m) return { user_id: m[1] };
    return {};
  }
}

async function verifySignature(
  headers: Headers,
  rawBody: string,
  webhookId: string,
  token: string,
): Promise<boolean> {
  const payload = {
    auth_algo: headers.get("paypal-auth-algo"),
    cert_url: headers.get("paypal-cert-url"),
    transmission_id: headers.get("paypal-transmission-id"),
    transmission_sig: headers.get("paypal-transmission-sig"),
    transmission_time: headers.get("paypal-transmission-time"),
    webhook_id: webhookId,
    webhook_event: JSON.parse(rawBody),
  };
  try {
    const res = await paypalFetch("/v1/notifications/verify-webhook-signature", {
      method: "POST",
      token,
      body: JSON.stringify(payload),
    });
    return res?.verification_status === "SUCCESS";
  } catch (err) {
    console.error("[paypal-webhook] verify failed", err);
    return false;
  }
}

async function fetchSubscription(id: string, token: string) {
  return paypalFetch(`/v1/billing/subscriptions/${id}`, { token });
}

// Column-fallback helper mirrors stripe-webhook — retries writes stripping
// optional columns the target DB doesn't have, so we never 500 on schema drift.
const OPTIONAL_LICENSE_COLS = ["paid_at", "organization_id", "payment_status", "payment_amount"];
const OPTIONAL_ORGANIZATION_COLS = [
  "plan",
  "payment_provider",
  "paypal_subscription_id",
  "paypal_payer_id",
  "subscription_status",
  "subscription_active",
  "trial_ends_at",
  "current_period_end",
];
const OPTIONAL_SUBSCRIPTION_COLS = [
  "serial_number",
  "payment_provider",
  "paypal_subscription_id",
  "paypal_payer_id",
  "trial_ends_at",
  "current_period_end",
  "canceled_at",
];

function findSchemaDriftColumn(message: string, payload: Record<string, any>, optionalColumns: string[]) {
  const lower = message.toLowerCase();
  return optionalColumns.find(
    (column) => column in payload && lower.includes(column.toLowerCase()),
  );
}

async function updateWithColumnFallback(
  supabase: any,
  table: string,
  payload: Record<string, any>,
  matchColumn: string,
  matchValue: string,
  optionalColumns: string[],
  label: string,
): Promise<{ error: any }> {
  const attempt: Record<string, any> = { ...payload };
  for (let i = 0; i < optionalColumns.length + 1; i++) {
    const res = await supabase.from(table).update(attempt).eq(matchColumn, matchValue);
    if (!res.error) return { error: null };
    const msg = res.error.message || "";
    const schemaish = /column|schema cache|Could not find/i.test(msg);
    if (!schemaish) return { error: res.error };
    const stripped = findSchemaDriftColumn(msg, attempt, optionalColumns);
    if (!stripped) return { error: res.error };
    delete attempt[stripped];
    console.warn(`[paypal-webhook] retrying ${label} without "${stripped}" (${msg})`);
  }
  return { error: new Error(`${label} exhausted column-fallback retries`) };
}

async function upsertSubscriptionWithFallback(
  supabase: any,
  payload: Record<string, any>,
): Promise<{ error: any }> {
  const { attachSubscriptionSerial } = await import("@/lib/subscription-serial.server");
  const attempt: Record<string, any> = await attachSubscriptionSerial(
    supabase,
    { ...payload },
    { column: "paypal_subscription_id", value: payload.paypal_subscription_id },
  );
  for (let i = 0; i < OPTIONAL_SUBSCRIPTION_COLS.length + 2; i++) {
    const res = await supabase.from("subscriptions").upsert(attempt, {
      onConflict: "paypal_subscription_id",
    });
    if (!res.error) return { error: null };

    const msg = res.error.message || "";
    if (/no unique|no exclusion|constraint/i.test(msg) && attempt.paypal_subscription_id) {
      const existing = await supabase
        .from("subscriptions")
        .select("id")
        .eq("paypal_subscription_id", attempt.paypal_subscription_id)
        .maybeSingle();
      if (existing.data?.id) {
        return updateWithColumnFallback(
          supabase,
          "subscriptions",
          attempt,
          "id",
          existing.data.id,
          OPTIONAL_SUBSCRIPTION_COLS,
          "subscriptions update",
        );
      }
      const inserted = await supabase.from("subscriptions").insert(attempt);
      if (!inserted.error) return { error: null };
      if (!/column|schema cache|Could not find/i.test(inserted.error.message || "")) {
        return { error: inserted.error };
      }
    }

    const schemaish = /column|schema cache|Could not find/i.test(msg);
    if (!schemaish) return { error: res.error };
    const stripped = findSchemaDriftColumn(msg, attempt, OPTIONAL_SUBSCRIPTION_COLS);
    if (!stripped) return { error: res.error };
    delete attempt[stripped];
    console.warn(`[paypal-webhook] retrying subscriptions upsert without "${stripped}" (${msg})`);
  }
  return { error: new Error("subscriptions upsert exhausted column-fallback retries") };
}

async function licenseWriteWithFallback(
  supabase: any,
  op: "insert" | "update",
  payload: Record<string, any>,
  idForUpdate?: string,
): Promise<{ error: any }> {
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
      return { error: null };
    }
    const msg = res.error.message || "";
    const schemaish = /column|schema cache|Could not find|violates check constraint/i.test(msg);
    if (!schemaish) return { error: res.error };
    const stripped = findSchemaDriftColumn(msg, attempt, OPTIONAL_LICENSE_COLS);
    if (!stripped) return { error: res.error };
    delete attempt[stripped];
    console.warn(`[paypal-webhook] retrying license ${op} without "${stripped}" (${msg})`);
  }
  return { error: new Error("license write exhausted column-fallback retries") };
}

/**
 * Persist the PayPal billing method so Settings → Billing can list it next to
 * Stripe cards. Idempotent on (owner_profile_id, provider, subscription id).
 */
export async function syncPayPalPaymentMethod(
  supabase: any,
  args: {
    ownerProfileId: string;
    organizationId: string | null;
    subscriptionId: string;
    payerId: string | null;
    payerEmail: string | null;
    payerName: string | null;
  },
): Promise<void> {
  const { ownerProfileId, organizationId, subscriptionId, payerId } = args;
  const label = args.payerEmail ?? args.payerName ?? "PayPal";
  const last4 = (args.payerEmail ?? payerId ?? "").slice(-4) || null;

  const { data: existing } = await supabase
    .from("payment_methods")
    .select("id")
    .eq("owner_profile_id", ownerProfileId)
    .eq("provider", "paypal")
    .eq("provider_subscription_id", subscriptionId)
    .maybeSingle();

  const payload: Record<string, any> = {
    owner_profile_id: ownerProfileId,
    organization_id: organizationId,
    provider: "paypal",
    provider_customer_id: payerId,
    provider_payment_method_id: payerId,
    provider_subscription_id: subscriptionId,
    brand: "paypal",
    last4,
    cardholder: label,
    status: "active",
    is_default: true,
    metadata: {
      payer_email: args.payerEmail,
      payer_name: args.payerName,
      payer_id: payerId,
    },
  };

  // Only one active method per profile (partial unique index).
  await supabase
    .from("payment_methods")
    .update({ status: "inactive", is_default: false })
    .eq("owner_profile_id", ownerProfileId)
    .neq("provider_subscription_id", subscriptionId);

  if (existing?.id) {
    const { error } = await supabase
      .from("payment_methods")
      .update(payload)
      .eq("id", existing.id);
    if (error) throw new Error(error.message);
    return;
  }
  const { error } = await supabase.from("payment_methods").insert(payload);
  if (error) throw new Error(error.message);
}

/**
 * Record a completed PayPal payment as a billing_invoices row so the Billing
 * history table has real data. Idempotent on provider_invoice_id.
 */
export async function recordPayPalInvoice(
  supabase: any,
  args: {
    organizationId: string | null;
    ownerProfileId: string | null;
    saleId: string;
    amountCents: number;
    currency: string;
    paidAt: string | null;
  },
): Promise<void> {
  const { data: existing } = await supabase
    .from("billing_invoices")
    .select("id")
    .eq("provider", "paypal")
    .eq("provider_invoice_id", args.saleId)
    .maybeSingle();
  if (existing?.id) return;

  const payload: Record<string, any> = {
    organization_id: args.organizationId,
    owner_profile_id: args.ownerProfileId,
    provider: "paypal",
    provider_invoice_id: args.saleId,
    invoice_number: args.saleId,
    amount_cents: args.amountCents,
    currency: args.currency,
    status: "paid",
    paid_at: args.paidAt,
  };

  let { error } = await supabase.from("billing_invoices").insert(payload);
  if (error && /owner_profile_id/i.test(error.message || "")) {
    delete payload.owner_profile_id;
    ({ error } = await supabase.from("billing_invoices").insert(payload));
  }
  if (error) console.warn("[paypal-webhook] billing_invoices insert failed", error.message);
}

// Finalizer: profile → customer_accounts → organization → enterprise_license → profile flags.
// Idempotent. Profile flags are updated in a `finally`-style block so a
// license failure doesn't lock the user out of the app after payment.

export async function upsertFromSubscription(
  supabase: any,
  sub: any,
) {
  const meta = parseCustomId(sub.custom_id);
  const userId = meta.user_id;
  if (!userId) {
    console.error("[paypal-webhook] missing user_id in custom_id", { custom_id: sub.custom_id });
    throw new Error("missing user_id metadata");
  }

  const status = sub.status as string;
  const mappedStatus = mapStatus(status);
  const active = subActiveFromStatus(status);
  const payerId: string | null = sub.subscriber?.payer_id ?? null;
  const nextBillingTime: string | null =
    sub.billing_info?.next_billing_time ?? null;
  const cycles: any[] = Array.isArray(sub.billing_info?.cycle_executions)
    ? sub.billing_info.cycle_executions
    : [];
  const inTrialTenure = cycles.some(
    (c: any) => c.tenure_type === "TRIAL" && c.cycles_completed < c.total_cycles,
  );
  // A subscription whose start_time is still in the future (our app-managed
  // 15-day free trial) has taken no payment yet.
  const noPaymentYet =
    Number(sub.billing_info?.failed_payments_count ?? 0) === 0 &&
    !sub.billing_info?.last_payment &&
    !!nextBillingTime &&
    new Date(nextBillingTime).getTime() > Date.now();
  const trialEnd: string | null =
    status === "ACTIVE" && (inTrialTenure || noPaymentYet) ? nextBillingTime : null;

  // PayPal reports ACTIVE as soon as the mandate is approved, even when our
  // 15-day free trial means no payment has been taken yet. Persist that as
  // "trialing" so Billing shows the trial state instead of a paid plan.
  const effectiveStatus = trialEnd ? "trialing" : mappedStatus;
  const effectiveActive = trialEnd ? true : active;



  // Resolve profile.
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
    const { data: created, error: profErr } = await supabase
      .from("profiles")
      .insert({ id: userId })
      .select("id, organization_id")
      .single();
    if (profErr) throw new Error("profile create failed: " + profErr.message);
    profile = created as any;
  }
  const ownerProfileId = profile!.id as string;

  // Resolve which plan was purchased from the PayPal plan id. Defaults to
  // enterprise for legacy subscriptions with no matching plans row.
  let planType = "enterprise";
  let planRowId: string | null = null;
  try {
    if (sub.plan_id) {
      const { data: planRow } = await supabase
        .from("plans")
        .select("id, plan_type")
        .eq("paypal_plan_id", sub.plan_id)
        .maybeSingle();
      if (planRow?.plan_type) planType = String(planRow.plan_type).toLowerCase();
      if (planRow?.id) planRowId = planRow.id as string;
    }
  } catch (err) {
    console.warn("[paypal-webhook] plan lookup failed, defaulting to enterprise", err);
  }



  // Prefer billing info from customer_accounts (pre-provisioned before the
  // PayPal redirect); fall back to legacy custom_id fields if present.
  let companyName = meta.company_name || "Enterprise Organization";
  let billingAddress: string | null = meta.billing_address || null;
  let vatNumber: string | null = meta.vat_number || null;

  let customerAccountId: string | undefined;
  {
    const { data: ca } = await supabase
      .from("customer_accounts")
      .select("id, company_name, billing_address, vat_number")
      .eq("owner_user_id", ownerProfileId)
      .maybeSingle();
    if (ca?.id) {
      customerAccountId = ca.id;
      companyName = ca.company_name || companyName;
      billingAddress = ca.billing_address ?? billingAddress;
      vatNumber = ca.vat_number ?? vatNumber;
    } else {
      const { data: created, error } = await supabase
        .from("customer_accounts")
        .insert({
          owner_user_id: ownerProfileId,
          company_name: companyName,
          billing_address: billingAddress,
          vat_number: vatNumber,
          plan_type: planType,
        })
        .select("id")
        .single();
      if (error) throw new Error("customer_accounts insert failed: " + error.message);
      customerAccountId = created.id as string;
    }
  }

  // Organization.
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
      .insert({ name: companyName, owner_profile_id: ownerProfileId })
      .select("id")
      .single();
    if (orgErr) throw new Error("organizations insert failed: " + orgErr.message);
    organizationId = orgCreated.id as string;
  }

  // Seed defaults (division, roles, owner app_user) — idempotent.
  {
    const { seedOrganizationDefaults } = await import("@/lib/org-defaults.server");
    await seedOrganizationDefaults(supabase, organizationId!);
  }

  const { error: orgUpdErr } = await updateWithColumnFallback(
    supabase,
    "organizations",
    {
      name: companyName,
      plan: planType,
      payment_provider: "paypal",
      paypal_subscription_id: sub.id,
      paypal_payer_id: payerId,
      subscription_status: effectiveStatus,
      subscription_active: effectiveActive,
      trial_ends_at: trialEnd,
      current_period_end: nextBillingTime,
    },
    "id",
    organizationId,
    OPTIONAL_ORGANIZATION_COLS,
    "organizations update",
  );
  if (orgUpdErr) throw new Error("organizations update failed: " + orgUpdErr.message);

  // subscriptions — the single source of truth the app reads for plan/state.
  if (planRowId) {
    const { error: subUpsertErr } = await upsertSubscriptionWithFallback(
      supabase,
      {
        owner_profile_id: ownerProfileId,
        organization_id: organizationId,
        plan_id: planRowId,
        payment_provider: "paypal",
        paypal_subscription_id: sub.id,
        paypal_payer_id: payerId,
        status: effectiveStatus,
        trial_ends_at: trialEnd,
        current_period_end: nextBillingTime,
        canceled_at: effectiveStatus === "canceled" ? new Date().toISOString() : null,
      },
    );
    if (subUpsertErr) {
      console.error("[paypal-webhook] subscriptions upsert failed", subUpsertErr.message);
    } else {
      const { logPlanChange } = await import("@/lib/plan-change-log.server");
      await logPlanChange(supabase, {
        ownerProfileId,
        organizationId,
        planId: planRowId,
        status: effectiveStatus,
        paymentProvider: "paypal",
        subscriptionId: sub.id,
      });
    }


    // Cancel any previous gateway subscription (older PayPal one, or the
    // Stripe subscription the user is switching away from) so the customer
    // is not charged twice.
    if (effectiveStatus === "active" || effectiveStatus === "trialing") {
      const { cancelSupersededSubscriptions } = await import(
        "@/lib/subscription-supersede.server"
      );
      await cancelSupersededSubscriptions(supabase, {
        ownerProfileId,
        organizationId,
        keepPaypalSubscriptionId: sub.id as string,
      });
    }

  } else {
    console.warn("[paypal-webhook] no plans row matched paypal plan_id", sub.plan_id);
  }

  // payment_methods — so the Billing tab can list PayPal as a saved method.
  try {
    await syncPayPalPaymentMethod(supabase, {
      ownerProfileId,
      organizationId,
      subscriptionId: sub.id as string,
      payerId,
      payerEmail: sub.subscriber?.email_address ?? null,
      payerName:
        [sub.subscriber?.name?.given_name, sub.subscriber?.name?.surname]
          .filter(Boolean)
          .join(" ") || null,
    });
  } catch (err) {
    console.warn("[paypal-webhook] payment_methods sync failed", err);
  }




  // enterprise_licenses — upsert by customer_account_id. Wrap in try/catch so
  // profile flags below always run (user must not be stuck on Choose Plan
  // after a successful payment).
  let licenseError: string | null = null;
  try {
    const { data: existingLic } = await supabase
      .from("enterprise_licenses")
      .select("id")
      .eq("customer_account_id", customerAccountId)
      .order("issued_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    const paymentStatus =
      effectiveStatus === "trialing"
        ? "pending"
        : effectiveStatus === "active"
        ? "paid"
        : "unpaid";
    const paidAt = effectiveStatus === "active" ? new Date().toISOString() : null;
    const expiresAt =
      nextBillingTime ||
      new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString();

    if (existingLic?.id) {
      const { error } = await licenseWriteWithFallback(
        supabase,
        "update",
        {
          status: "active",
          expires_at: expiresAt,
          payment_status: paymentStatus,
          paid_at: paidAt,
        },
        existingLic.id,
      );
      if (error) throw error;
    } else {
      // Serial derived from subscription id + short random to avoid retries
      // colliding on the plan-unique index.
      const rand = Math.random().toString(36).slice(2, 6).toUpperCase();
      const serial = `ENT-${String(sub.id).slice(-8).toUpperCase()}-${rand}`;
      const { error } = await licenseWriteWithFallback(supabase, "insert", {
        customer_account_id: customerAccountId,
        organization_id: organizationId,
        serial_number: serial,
        status: "active",
        issued_at: new Date().toISOString(),
        expires_at: expiresAt,
        created_by: ownerProfileId,
        payment_amount: 0,
        payment_status: paymentStatus,
        paid_at: paidAt,
      });
      if (error) throw error;
    }
  } catch (err) {
    licenseError = err instanceof Error ? err.message : String(err);
    console.error("[paypal-webhook] license write failed (continuing to flag profile)", licenseError);
  }

  // Profile flags — always update, even if the license insert failed. The
  // organization row already reflects the true subscription state; the
  // profile mirror lets the app show the Billing tab instead of Choose Plan.
  const { error: pErr } = await supabase
    .from("profiles")
    .update({
      organization_id: organizationId,
      selected_plan: planType,
      subscription_active: effectiveActive,
      subscription_status: effectiveStatus,
      trial_ends_at: trialEnd,
      next_billing_date: nextBillingTime,
      is_admin: true,
    })
    .eq("id", ownerProfileId);
  if (pErr) console.error("[paypal-webhook] profile flags update failed", pErr.message);

  if (licenseError) {
    // Surface the license failure so the caller (webhook handler) can log
    // it, but the payment is still finalized from the user's perspective.
    throw new Error("license write failed: " + licenseError);
  }
}

async function markPaymentFailed(supabase: any, subscriptionId: string) {
  await supabase
    .from("organizations")
    .update({ subscription_status: "past_due", subscription_active: false })
    .eq("paypal_subscription_id", subscriptionId);

  // subscriptions is the source of truth the app reads for plan state.
  const { error } = await supabase
    .from("subscriptions")
    .update({ status: "past_due" })
    .eq("paypal_subscription_id", subscriptionId);
  if (error) console.error("[paypal-webhook] past_due update failed", error.message);
}


export const Route = createFileRoute("/api/public/paypal-webhook")({
  server: {
    handlers: {
      OPTIONS: async () => new Response(null, { status: 204, headers: CORS }),
      POST: async ({ request }) => {
        const webhookId = process.env.PAYPAL_WEBHOOK_ID;
        if (!webhookId) return textResp("PAYPAL_WEBHOOK_ID missing", 500);

        const rawBody = await request.text();
        const token = await getPayPalAccessToken();
        const ok = await verifySignature(request.headers, rawBody, webhookId, token);
        if (!ok) {
          console.error("[paypal-webhook] invalid signature", {
            base: getPayPalBase(),
          });
          return textResp("Invalid signature", 401);
        }

        let event: any;
        try {
          event = JSON.parse(rawBody);
        } catch {
          return textResp("Invalid JSON", 400);
        }

        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
        const supabase = supabaseAdmin as any;

        try {
          const type = event.event_type as string;
          const resource = event.resource ?? {};

          switch (type) {
            case "BILLING.SUBSCRIPTION.CREATED":
            case "BILLING.SUBSCRIPTION.ACTIVATED":
            case "BILLING.SUBSCRIPTION.UPDATED": {
              const subId =
                resource.id ||
                resource.billing_agreement_id ||
                resource.subscription_id;
              if (!subId) break;
              const sub = await fetchSubscription(subId, token);
              await upsertFromSubscription(supabase, sub);
              break;
            }
            case "BILLING.SUBSCRIPTION.CANCELLED":
            case "BILLING.SUBSCRIPTION.EXPIRED":
            case "BILLING.SUBSCRIPTION.SUSPENDED": {
              const subId = resource.id;
              if (!subId) break;

              // Respect end-of-period cancellations we initiated ourselves: if
              // the local row is already flagged as cancel_at_period_end, keep the
              // subscription active until the current period ends. Only mark the
              // final status as canceled once the period has passed (or the event
              // is an explicit cancellation from PayPal outside of our scheduled
              // flow).
              const { data: existing } = await supabase
                .from("subscriptions")
                .select("id, status, current_period_end, trial_ends_at, cancel_at_period_end")
                .eq("paypal_subscription_id", subId)
                .maybeSingle();

              const periodEnd = (existing?.current_period_end ?? existing?.trial_ends_at) as
                | string
                | null
                | undefined;
              const periodEnded = periodEnd ? new Date(periodEnd).getTime() <= Date.now() : true;
              const scheduledCancel = Boolean(existing?.cancel_at_period_end);

              const isSuspended = type === "BILLING.SUBSCRIPTION.SUSPENDED";
              const isCancelled = type === "BILLING.SUBSCRIPTION.CANCELLED";

              let nextStatus: string;
              let nextActive: boolean;
              let canceledAt: string | null = null;

              if (isCancelled || (scheduledCancel && periodEnded)) {
                nextStatus = "canceled";
                nextActive = false;
                canceledAt = new Date().toISOString();
              } else if (scheduledCancel && !periodEnded) {
                // User clicked "Cancel" on our UI; billing is suspended but they
                // keep access until the period ends. Preserve the existing active/
                // trialing status so the UI shows "scheduled for cancellation".
                nextStatus = (existing?.status as string | null) || "active";
                nextActive = /active|trialing/.test(nextStatus);
              } else if (isSuspended) {
                nextStatus = "past_due";
                nextActive = false;
              } else {
                nextStatus = "expired";
                nextActive = false;
              }

              await supabase
                .from("organizations")
                .update({
                  subscription_status: nextStatus,
                  subscription_active: nextActive,
                  ...(canceledAt ? { canceled_at: canceledAt } : {}),
                })
                .eq("paypal_subscription_id", subId);

              await supabase
                .from("subscriptions")
                .update({
                  status: nextStatus,
                  ...(canceledAt ? { canceled_at: canceledAt } : {}),
                })
                .eq("paypal_subscription_id", subId);

              break;
            }

            case "PAYMENT.SALE.COMPLETED": {
              const subId = resource.billing_agreement_id;
              if (subId) {
                const sub = await fetchSubscription(subId, token);
                await upsertFromSubscription(supabase, sub);

                // Billing history row for this payment.
                const { data: subRow } = await supabase
                  .from("subscriptions")
                  .select("organization_id, owner_profile_id")
                  .eq("paypal_subscription_id", subId)
                  .maybeSingle();
                const value = Number(resource?.amount?.total ?? 0);
                await recordPayPalInvoice(supabase, {
                  organizationId: subRow?.organization_id ?? null,
                  ownerProfileId: subRow?.owner_profile_id ?? null,
                  saleId: String(resource?.id ?? `${subId}-${Date.now()}`),
                  amountCents: Number.isFinite(value) ? Math.round(value * 100) : 0,
                  currency: String(resource?.amount?.currency ?? "USD"),
                  paidAt: resource?.create_time ?? new Date().toISOString(),
                });
              }
              break;
            }

            case "PAYMENT.SALE.DENIED":
            case "BILLING.SUBSCRIPTION.PAYMENT.FAILED": {
              const subId =
                resource.billing_agreement_id || resource.id;
              if (subId) await markPaymentFailed(supabase, subId);
              break;
            }
            case "PAYMENT.SALE.REFUNDED":
              // No-op for now; could flip status to "refunded".
              break;
            default:
              console.log("[paypal-webhook] unhandled event", type);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : "handler error";
          console.error("[paypal-webhook] handler failed:", message);
          // Return 200 so PayPal doesn't retry-storm partial writes. The
          // success-redirect finalizer + reconciliation loop pick up the slack.
          return textResp(`ok (handler error: ${message})`);
        }

        return textResp("ok");
      },
    },
  },
});
