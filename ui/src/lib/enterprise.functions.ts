import { createServerFn } from "@tanstack/react-start";
import { randomBytes } from "crypto";
import Stripe from "stripe";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { assertBillingDetails } from "./billing-details";

type PaymentMethodBrand =
  | "visa"
  | "mastercard"
  | "amex"
  | "discover"
  | "unionpay"
  | "diners"
  | "jcb"
  | "unknown";

type PaymentMethodInfo = {
  brand: PaymentMethodBrand;
  last4: string;
  exp_month: number;
  exp_year: number;
  cardholder: string | null;
};

function mapBrand(brand: string | null | undefined): PaymentMethodBrand {
  const b = String(brand ?? "").toLowerCase();
  if (b === "visa") return "visa";
  if (b === "mastercard") return "mastercard";
  if (b === "amex" || b === "american_express") return "amex";
  if (b === "discover") return "discover";
  if (b === "unionpay") return "unionpay";
  if (b === "diners") return "diners";
  if (b === "jcb") return "jcb";
  return "unknown";
}

function pmFromStripe(pm: Stripe.PaymentMethod | null | undefined): PaymentMethodInfo | null {
  if (!pm || pm.type !== "card" || !pm.card) return null;
  return {
    brand: mapBrand(pm.card.brand),
    last4: pm.card.last4 ?? "",
    exp_month: pm.card.exp_month ?? 0,
    exp_year: pm.card.exp_year ?? 0,
    cardholder: pm.billing_details?.name ?? null,
  };
}

async function resolvePaymentMethod(
  customerId: string | null | undefined,
  subscriptionId: string | null | undefined,
): Promise<PaymentMethodInfo | null> {
  if (!customerId) return null;
  const key = process.env.STRIPE_TEST_API_KEY || process.env.STRIPE_SECRET_KEY;
  if (!key) return null;
  try {
    const stripe = new Stripe(key, { httpClient: Stripe.createFetchHttpClient() } as any);

    if (subscriptionId) {
      try {
        const sub = await stripe.subscriptions.retrieve(subscriptionId, {
          expand: ["default_payment_method"],
        });
        const pm = sub.default_payment_method;
        if (pm && typeof pm !== "string") {
          const info = pmFromStripe(pm as Stripe.PaymentMethod);
          if (info) return info;
        }
      } catch (err) {
        console.warn("[billing] subscription payment method lookup failed", err);
      }
    }

    try {
      const customer = await stripe.customers.retrieve(customerId, {
        expand: ["invoice_settings.default_payment_method"],
      });
      if (customer && !(customer as any).deleted) {
        const c = customer as Stripe.Customer;
        const pm = c.invoice_settings?.default_payment_method;
        if (pm && typeof pm !== "string") {
          const info = pmFromStripe(pm as Stripe.PaymentMethod);
          if (info) return info;
        }
      }
    } catch (err) {
      console.warn("[billing] customer payment method lookup failed", err);
    }

    try {
      const list = await stripe.paymentMethods.list({
        customer: customerId,
        type: "card",
        limit: 1,
      });
      const info = pmFromStripe(list.data[0]);
      if (info) return info;
    } catch (err) {
      console.warn("[billing] payment methods list failed", err);
    }
  } catch (err) {
    console.warn("[billing] resolvePaymentMethod failed", err);
  }
  return null;
}

const TRIAL_DAYS = 15;
// "past_due" excluded on purpose: a failed payment revokes access immediately.
const ACTIVE_SUB_STATUSES = new Set(["trialing", "active"]);

function generateSerial(): string {
  const hex = randomBytes(6).toString("hex").toUpperCase();
  return `ENT-${hex.slice(0, 4)}-${hex.slice(4, 8)}-${hex.slice(8, 12)}`;
}

function daysRemaining(iso: string | null | undefined): number {
  if (!iso) return 0;
  const ms = new Date(iso).getTime() - Date.now();
  if (ms <= 0) return 0;
  return Math.ceil(ms / (1000 * 60 * 60 * 24));
}

type AnyRec = Record<string, any>;

/**
 * Live billing history for a PayPal subscription. Used as a fallback when the
 * billing_invoices table has no rows yet (webhook missed or older account).
 */
async function fetchPayPalInvoices(subscriptionId: string): Promise<AnyRec[]> {
  const { paypalFetch } = await import("./paypal-helpers");
  const end = new Date();
  const start = new Date(end.getTime() - 365 * 24 * 60 * 60 * 1000);
  const qs = new URLSearchParams({
    start_time: start.toISOString(),
    end_time: end.toISOString(),
  });
  const res = await paypalFetch(
    `/v1/billing/subscriptions/${subscriptionId}/transactions?${qs.toString()}`,
    {},
  );
  const txns: any[] = Array.isArray(res?.transactions) ? res.transactions : [];
  return txns.map((t) => {
    const gross = t?.amount_with_breakdown?.gross_amount ?? {};
    const value = Number(gross.value ?? 0);
    return {
      id: t?.id ?? null,
      provider: "paypal",
      provider_invoice_id: t?.id ?? null,
      invoice_number: t?.id ?? null,
      amount_cents: Number.isFinite(value) ? Math.round(value * 100) : 0,
      currency: gross.currency_code ?? "USD",
      status: String(t?.status ?? "").toUpperCase() === "COMPLETED" ? "paid" : String(t?.status ?? "pending").toLowerCase(),
      paid_at: t?.time ?? null,
      hosted_url: null,
    } as AnyRec;
  });
}


// Billing columns have been removed from profiles — subscriptions is the
// source of truth. profiles keeps identity + org link only.
const PROFILE_COLUMNS =
  "id, user_id, full_name, phone, country, is_admin, organization_id";

async function findProfile(supabase: any, userId: string): Promise<AnyRec | null> {
  const { data, error } = await supabase
    .from("profiles")
    .select(PROFILE_COLUMNS)
    .or(`id.eq.${userId},user_id.eq.${userId}`)
    .maybeSingle();
  if (error) throw new Error(`profiles read failed: ${error.message}`);
  return (data ?? null) as AnyRec | null;
}

async function ensureProfile(supabase: any, userId: string): Promise<AnyRec> {
  const existing = await findProfile(supabase, userId);
  if (existing) return existing;

  const { data, error } = await supabase
    .from("profiles")
    .insert({ user_id: userId })
    .select(PROFILE_COLUMNS)
    .single();
  if (error) throw new Error(`profiles insert failed: ${error.message}`);
  return data as AnyRec;
}

/**
 * Load the most recent subscription for either the caller's organization
 * (preferred) or their owner profile. Joins the plans row so the caller can
 * derive plan name/type, price, interval, and stripe_price_id.
 */
export async function loadActiveSubscription(
  supabase: any,
  args: { ownerProfileId?: string | null; organizationId?: string | null },
): Promise<{ subscription: AnyRec | null; plan: AnyRec | null }> {
  const { ownerProfileId, organizationId } = args;
  if (!ownerProfileId && !organizationId) return { subscription: null, plan: null };

  const select = () =>
    supabase
      .from("subscriptions")
      .select(
        "*, plan:plans(id, name, plan_type, price, billing_interval, stripe_price_id)",
      )
      .order("created_at", { ascending: false })
      .limit(1);

  let row: AnyRec | null = null;

  if (organizationId) {
    const { data, error } = await select().eq("organization_id", organizationId).maybeSingle();
    if (error) console.warn("[enterprise] subscriptions read failed", error.message);
    row = (data ?? null) as AnyRec | null;
  }

  // Fallback: the profile may not be linked to the organization that owns the
  // newest subscription (org link write can be blocked by a DB trigger). Use
  // the most recent subscription owned by this profile, whatever its org.
  if (!row && ownerProfileId) {
    const { data, error } = await select().eq("owner_profile_id", ownerProfileId).maybeSingle();
    if (error) console.warn("[enterprise] subscriptions owner read failed", error.message);
    row = (data ?? null) as AnyRec | null;
  }

  const plan = (row?.plan ?? null) as AnyRec | null;
  if (row) delete row.plan;
  return { subscription: row, plan };
}


async function loadSummary(supabase: any, userId: string, profileOverride?: AnyRec | null) {
  const profile = profileOverride === undefined ? await findProfile(supabase, userId) : profileOverride;
  const ownerProfileId = profile?.id ?? userId;
  const profileOrgId = profile?.organization_id ?? null;

  // Subscription + plan — single source of truth for all billing fields.
  const { subscription, plan } = await loadActiveSubscription(supabase, {
    ownerProfileId,
    organizationId: profileOrgId,
  });

  // The profile org link can lag behind (or fail to write), so trust the
  // organization attached to the newest subscription when it is missing.
  const orgId = (profileOrgId ?? subscription?.organization_id ?? null) as string | null;

  // Repair the missing profile → organization link so every other screen
  // (users, reports, permissions) resolves the same organization.
  if (!profileOrgId && orgId) {
    const { error: linkErr } = await supabase
      .from("profiles")
      .update({ organization_id: orgId })
      .eq("id", ownerProfileId);
    if (linkErr) console.warn("[enterprise] profile org backfill failed", linkErr.message);
  }


  const { data: customerAccount } = await supabase
    .from("customer_accounts")
    .select("*")
    .eq("owner_user_id", ownerProfileId)
    .maybeSingle();

  let license: AnyRec | null = null;
  if (customerAccount?.id) {
    const { data: lic } = await supabase
      .from("enterprise_licenses")
      .select("*")
      .eq("customer_account_id", customerAccount.id)
      .order("issued_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    license = lic ?? null;
  }

  // Organization now holds ONLY identity/branding fields.
  let organization: AnyRec | null = null;
  if (orgId) {
    const { data: org } = await supabase
      .from("organizations")
      .select("*")
      .eq("id", orgId)
      .maybeSingle();
    organization = (org ?? null) as AnyRec | null;
  }

  // Invoice history (real paid invoices from provider webhooks)
  let invoices: AnyRec[] = [];
  {
    const q = supabase
      .from("billing_invoices")
      .select("*")
      .order("paid_at", { ascending: false, nullsFirst: false })
      .order("created_at", { ascending: false })
      .limit(50);
    const { data: rows } = orgId
      ? await q.eq("organization_id", orgId)
      : await q.eq("owner_profile_id", ownerProfileId);
    invoices = (rows ?? []) as AnyRec[];
  }

  // Plan change history (upgrades / downgrades between plans)
  let planChanges: AnyRec[] = [];
  {
    const q = supabase
      .from("plan_change_log")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(50);
    const { data: rows, error } = orgId
      ? await q.eq("organization_id", orgId)
      : await q.eq("owner_profile_id", ownerProfileId);
    if (error) console.warn("[enterprise] plan_change_log lookup failed", error.message);
    planChanges = (rows ?? []) as AnyRec[];
  }





  const rawStatus = (subscription?.status ?? null) as string | null;
  const trialEndsAt = (subscription?.trial_ends_at ?? null) as string | null;
  // A subscription still inside its free trial window is "trialing", even if
  // the gateway (PayPal) already flipped it to active on mandate approval.
  const inTrialWindow =
    !!trialEndsAt && new Date(trialEndsAt).getTime() > Date.now();
  const status =
    rawStatus === "active" && inTrialWindow ? "trialing" : rawStatus;

  const stripeCustomerId = (subscription?.stripe_customer_id ?? null) as string | null;
  const stripeSubscriptionId = (subscription?.stripe_subscription_id ?? null) as
    | string
    | null;
  const paymentProvider = (subscription?.payment_provider ?? null) as string | null;
  const priceNumber =
    plan?.price !== null && plan?.price !== undefined ? Number(plan.price) : null;
  const unitAmountCents =
    priceNumber !== null && Number.isFinite(priceNumber)
      ? Math.round(priceNumber * 100)
      : null;

  const mergedProfile: AnyRec | null = profile
    ? {
        ...profile,
        // Derived from subscription + plan
        selected_plan: plan?.plan_type ?? null,
        plan_name: plan?.name ?? null,
        subscription_active: status ? ACTIVE_SUB_STATUSES.has(status) : false,
        subscription_status: status,
        trial_ends_at: trialEndsAt,
        next_billing_date: (subscription?.current_period_end ?? null) as string | null,
        current_period_start: (subscription?.current_period_start ?? null) as
          | string
          | null,
        current_period_end: (subscription?.current_period_end ?? null) as string | null,
        stripe_customer_id: stripeCustomerId,
        stripe_subscription_id: stripeSubscriptionId,
        payment_provider: paymentProvider,
        canceled_at: (subscription?.canceled_at ?? null) as string | null,
        cancel_at_period_end: Boolean(subscription?.cancel_at_period_end),
        // Pricing derived from plans
        unit_amount_cents: unitAmountCents,
        monthly_price_cents:
          plan?.billing_interval === "yearly" && unitAmountCents !== null
            ? Math.round(unitAmountCents / 12)
            : unitAmountCents,
        billing_interval: plan?.billing_interval ?? null,
        currency: "USD",
        stripe_price_id: (plan?.stripe_price_id ?? null) as string | null,
        // Card details resolved live from Stripe below
        card_brand: null,
        card_last4: null,
        card_exp_month: null,
        card_exp_year: null,
        billing_email: customerAccount?.billing_email ?? null,
        billing_name:
          customerAccount?.company_name ?? profile.full_name ?? null,
        // PayPal identifiers (read from the subscriptions row so the Billing
        // tab can render the PayPal method even before payment_methods syncs).
        paypal_subscription_id: (subscription?.paypal_subscription_id ?? null) as
          | string
          | null,
        paypal_payer_id: (subscription?.paypal_payer_id ?? null) as string | null,
      }
    : null;

  const paymentProviderIsPayPal = paymentProvider === "paypal";
  const paypalSubscriptionId = (subscription?.paypal_subscription_id ?? null) as
    | string
    | null;

  const paymentMethod = paymentProviderIsPayPal
    ? null
    : await resolvePaymentMethod(stripeCustomerId, stripeSubscriptionId);

  if (mergedProfile && paymentMethod) {
    mergedProfile.card_brand = paymentMethod.brand;
    mergedProfile.card_last4 = paymentMethod.last4;
    mergedProfile.card_exp_month = paymentMethod.exp_month;
    mergedProfile.card_exp_year = paymentMethod.exp_year;
  }

  // Fallback billing history straight from PayPal when we have no stored
  // invoices yet (webhook missed / older subscription).
  if (invoices.length === 0 && paymentProviderIsPayPal && paypalSubscriptionId) {
    try {
      invoices = await fetchPayPalInvoices(paypalSubscriptionId);
    } catch (err) {
      console.warn("[billing] paypal transactions fetch failed", err);
    }
  }


  return {
    profile: mergedProfile,
    organization,
    subscription: subscription ? { ...subscription, status } : subscription,
    plan,
    customerAccount: (customerAccount ?? null) as AnyRec | null,
    license,
    serialNumber: (subscription?.serial_number ?? null) as string | null,
    paymentMethod,
    invoices,
    planChanges,

    trialDaysRemaining: daysRemaining(trialEndsAt),
  };
}


export const getBillingSummary = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const summary = await loadSummary(supabase, context.userId);
    const status = summary.subscription?.status as string | undefined;
    const organizationId = (summary.organization?.id ??
      (summary.subscription as AnyRec | null)?.organization_id ??
      null) as string | null;

    if (
      organizationId &&
      summary.plan?.plan_type === "enterprise" &&
      status &&
      ACTIVE_SUB_STATUSES.has(status)
    ) {
      const { seedOrganizationDefaults } = await import("./org-defaults.server");
      await seedOrganizationDefaults(supabase, organizationId);
    }

    return summary;
  });

export type EnterpriseBillingInput = {
  company_name: string;
  billing_address: string;
  vat_number: string;
  /** Explicit mandate for recurring auto-payments (required). */
  auto_pay_authorized?: boolean;
};

async function findEnterpriseTrialPlan(supabase: any): Promise<AnyRec | null> {
  // Prefer monthly enterprise plan (matches the 15-day trial cadence). Fall
  // back to any active enterprise plan if the monthly one isn't seeded.
  const { data: monthly } = await supabase
    .from("plans")
    .select("id, plan_type, billing_interval, stripe_price_id")
    .eq("plan_type", "enterprise")
    .eq("billing_interval", "monthly")
    .eq("is_active", true)
    .limit(1)
    .maybeSingle();
  if (monthly?.id) return monthly as AnyRec;

  const { data: any } = await supabase
    .from("plans")
    .select("id, plan_type, billing_interval, stripe_price_id")
    .eq("plan_type", "enterprise")
    .eq("is_active", true)
    .limit(1)
    .maybeSingle();
  return (any ?? null) as AnyRec | null;
}

export const upgradeToEnterprise = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: EnterpriseBillingInput) => {
    if (!data || typeof data !== "object") throw new Error("Missing billing info");
    const { company_name, billing_address, vat_number } = assertBillingDetails(data as any);
    return { company_name, billing_address, vat_number };
  })

  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const userId = context.userId;
    const profile = await ensureProfile(supabase, userId);
    const ownerProfileId = profile.id as string;

    // Idempotent: already on an active enterprise subscription?
    const existing = await loadSummary(supabase, userId, profile);
    const existingOrganizationId = (profile.organization_id ??
      (existing.subscription as AnyRec | null)?.organization_id ??
      null) as string | null;
    if (existingOrganizationId) {
      const { seedOrganizationDefaults } = await import("./org-defaults.server");
      await seedOrganizationDefaults(supabase, existingOrganizationId);
    }
    if (
      existing.plan?.plan_type === "enterprise" &&
      existing.subscription?.status &&
      ACTIVE_SUB_STATUSES.has(existing.subscription.status as string) &&
      existing.license
    ) {
      return existing;
    }

    // 1. Ensure customer_accounts row (unchanged)
    let customerAccountId: string | undefined = existing.customerAccount?.id;
    if (!customerAccountId) {
      const { data: created, error: caErr } = await supabase
        .from("customer_accounts")
        .insert({
          owner_user_id: ownerProfileId,
          company_name: data.company_name,
          billing_address: data.billing_address,
          vat_number: data.vat_number,
          plan_type: "trial",
        })
        .select("id")
        .single();
      if (caErr) throw new Error(`customer_accounts insert failed: ${caErr.message}`);
      customerAccountId = created.id as string;
    } else {
      const { error: uErr } = await supabase
        .from("customer_accounts")
        .update({
          company_name: data.company_name,
          billing_address: data.billing_address,
          vat_number: data.vat_number,
        })
        .eq("id", customerAccountId);
      if (uErr) throw new Error(`customer_accounts update failed: ${uErr.message}`);
    }

    // 2. Ensure organization row (identity only)
    let organizationId: string | undefined = profile.organization_id ?? undefined;
    if (organizationId) {
      const { data: orgRow } = await supabase
        .from("organizations")
        .select("id, name")
        .eq("id", organizationId)
        .maybeSingle();
      if (!orgRow) {
        organizationId = undefined;
      } else if (orgRow.name !== data.company_name) {
        await supabase
          .from("organizations")
          .update({ name: data.company_name })
          .eq("id", organizationId);
      }
    }
    if (!organizationId) {
      const { data: orgCreated, error: orgErr } = await supabase
        .from("organizations")
        .insert({
          name: data.company_name,
          owner_profile_id: ownerProfileId,
        })
        .select("id")
        .single();
      if (orgErr) throw new Error(`organizations insert failed: ${orgErr.message}`);
      organizationId = orgCreated.id as string;
    }

    // Seed defaults (division, roles, owner app_user) — idempotent, also
    // backfills organizations created before this logic existed.
    {
      const { seedOrganizationDefaults } = await import("./org-defaults.server");
      await seedOrganizationDefaults(supabase, organizationId!);
    }


    // 3. Insert enterprise_licenses (retry serial collision) — unchanged
    const now = new Date();
    const expiresAt = new Date(now.getTime() + TRIAL_DAYS * 24 * 60 * 60 * 1000);
    let licenseInserted = false;
    let lastErr: string | null = null;
    for (let i = 0; i < 4 && !licenseInserted; i++) {
      const licensePayload: Record<string, any> = {
        customer_account_id: customerAccountId,
        organization_id: organizationId,
        serial_number: generateSerial(),
        status: "active",
        issued_at: now.toISOString(),
        expires_at: expiresAt.toISOString(),
        created_by: ownerProfileId,
        payment_amount: 0,
        payment_status: "unpaid",
      };
      let { error } = await supabase.from("enterprise_licenses").insert(licensePayload);
      if (error && /organization_id/i.test(error.message) && /column|schema cache|Could not find/i.test(error.message)) {
        delete licensePayload.organization_id;
        ({ error } = await supabase.from("enterprise_licenses").insert(licensePayload));
      }
      if (!error) {
        licenseInserted = true;
        break;
      }
      lastErr = error.message;
      if (!/duplicate|unique/i.test(error.message)) break;
    }
    if (!licenseInserted) {
      throw new Error(`enterprise_licenses insert failed: ${lastErr ?? "unknown"}`);
    }

    // 4. Insert the trial subscription (source of truth for plan/status)
    const trialPlan = await findEnterpriseTrialPlan(supabase);
    if (!trialPlan?.id) {
      throw new Error("No active enterprise plan found in plans table");
    }
    // The free trial is one-time: skip creation if this org has EVER had a
    // subscription (any status), not just a currently active one.
    const { data: existingSub } = await supabase
      .from("subscriptions")
      .select("id, status, serial_number")
      .eq("organization_id", organizationId)
      .limit(1)
      .maybeSingle();
    if (!existingSub?.id) {
      // Generate the subscription serial once, retrying on the rare collision.
      let subInserted = false;
      let subErrMsg: string | null = null;
      for (let i = 0; i < 4 && !subInserted; i++) {
        const subPayload: Record<string, any> = {
          owner_profile_id: ownerProfileId,
          organization_id: organizationId,
          plan_id: trialPlan.id,
          payment_provider: "stripe",
          status: "trialing",
          serial_number: generateSerial(),
          trial_starts_at: now.toISOString(),
          trial_ends_at: expiresAt.toISOString(),
        };
        let { error } = await supabase.from("subscriptions").insert(subPayload);
        if (error && /serial_number/i.test(error.message)) {
          if (/duplicate|unique/i.test(error.message)) {
            subErrMsg = error.message;
            continue; // collision — new serial on the next attempt
          }
          // Column missing from the schema — insert without it.
          delete subPayload.serial_number;
          ({ error } = await supabase.from("subscriptions").insert(subPayload));
        }
        if (!error) {
          subInserted = true;
          break;
        }
        subErrMsg = error.message;
        break;
      }
      if (!subInserted) {
        throw new Error(`subscriptions insert failed: ${subErrMsg ?? "unknown"}`);
      }
      const { logPlanChange } = await import("./plan-change-log.server");
      await logPlanChange(supabase, {
        ownerProfileId,
        organizationId,
        planId: trialPlan.id,
        status: "trialing",
        paymentProvider: "stripe",
      });

    } else if (!existingSub.serial_number) {
      // Backfill a serial for a subscription created before this column existed.
      const { error: backfillErr } = await supabase
        .from("subscriptions")
        .update({ serial_number: generateSerial() })
        .eq("id", existingSub.id);
      if (backfillErr) {
        console.warn("[enterprise] serial backfill failed", backfillErr.message);
      }
    }

    // 5. Link profile to org + grant admin. NO billing writes.
    const { error: pErr } = await supabase
      .from("profiles")
      .update({
        is_admin: true,
        organization_id: organizationId,
      })
      .eq("id", ownerProfileId);
    if (pErr) throw new Error(`profiles update failed: ${pErr.message}`);

    return loadSummary(supabase, userId);
  });
