import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { paypalFetch } from "./paypal-helpers";
import { assertBillingDetails } from "./billing-details";

import { getAppOrigin } from "./app-origin";

export type PayPalSubscriptionInput = {
  company_name: string;
  billing_address: string;
  vat_number: string;
  /** Explicit mandate for recurring auto-payments (required). */
  auto_pay_authorized?: boolean;
  /** PayPal billing plan id (plans.paypal_plan_id). Falls back to PAYPAL_PLAN_ID. */
  paypal_plan_id?: string;
};


// Pre-provision customer_accounts + organization BEFORE redirecting to PayPal
// so the webhook (or the success-redirect finalizer) has everything it needs
// keyed by owner_user_id — we no longer stuff company/address/vat into
// PayPal's `custom_id` (which is capped at 127 chars and was silently
// truncated to invalid JSON in older builds).
async function preProvisionEnterpriseRows(
  supabase: any,
  userId: string,
  data: PayPalSubscriptionInput,
): Promise<{ ownerProfileId: string; organizationId: string; customerAccountId: string }> {
  // Resolve or create profile.
  let profile: any = null;
  {
    const { data: p } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();
    profile = p ?? null;
  }
  if (!profile?.id) {
    const { data: created, error } = await supabase
      .from("profiles")
      .insert({ id: userId })
      .select("id, organization_id")
      .single();
    if (error) throw new Error(`profile ensure failed: ${error.message}`);
    profile = created;
  }
  const ownerProfileId = profile.id as string;

  // Upsert customer_accounts by owner_user_id (unique per profile).
  let customerAccountId: string;
  const { data: existingCa } = await supabase
    .from("customer_accounts")
    .select("id")
    .eq("owner_user_id", ownerProfileId)
    .maybeSingle();
  if (existingCa?.id) {
    customerAccountId = existingCa.id as string;
    await supabase
      .from("customer_accounts")
      .update({
        company_name: data.company_name,
        billing_address: data.billing_address,
        vat_number: data.vat_number,
      })
      .eq("id", customerAccountId);
  } else {
    const { data: created, error } = await supabase
      .from("customer_accounts")
      .insert({
        owner_user_id: ownerProfileId,
        company_name: data.company_name,
        billing_address: data.billing_address,
        vat_number: data.vat_number,
        plan_type: "enterprise",
      })
      .select("id")
      .single();
    if (error) throw new Error(`customer_accounts insert failed: ${error.message}`);
    customerAccountId = created.id as string;
  }

  // Ensure organization row.
  let organizationId = profile.organization_id as string | null;
  if (organizationId) {
    const { data: orgRow } = await supabase
      .from("organizations")
      .select("id")
      .eq("id", organizationId)
      .maybeSingle();
    if (!orgRow) organizationId = null;
  }
  if (!organizationId) {
    const { data: created, error } = await supabase
      .from("organizations")
      .insert({ name: data.company_name, owner_profile_id: ownerProfileId })
      .select("id")
      .single();
    if (error) throw new Error(`organizations insert failed: ${error.message}`);
    organizationId = created.id as string;
  } else {
    await supabase
      .from("organizations")
      .update({ name: data.company_name })
      .eq("id", organizationId);
  }

  // Seed defaults (division, roles, owner app_user) — idempotent.
  {
    const { seedOrganizationDefaults } = await import("./org-defaults.server");
    await seedOrganizationDefaults(supabase, organizationId!);
  }

  // Link profile → organization so the webhook can find it later.
  if (profile.organization_id !== organizationId) {
    const { error: linkErr } = await supabase
      .from("profiles")
      .update({ organization_id: organizationId })
      .eq("id", ownerProfileId);
    if (linkErr) console.warn("[paypal] profile org link failed", linkErr.message);
  }


  return { ownerProfileId, organizationId: organizationId!, customerAccountId };
}

export const createPayPalSubscription = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: PayPalSubscriptionInput) => {
    if (!data || typeof data !== "object") throw new Error("Missing billing info");
    const { company_name, billing_address, vat_number } = assertBillingDetails(data);
    const paypal_plan_id = String(data.paypal_plan_id ?? "").trim() || undefined;
    return { company_name, billing_address, vat_number, paypal_plan_id };
  })

  .handler(async ({ context, data }) => {
    // Prefer the plan-specific PayPal plan id (plans.paypal_plan_id); fall
    // back to the global env plan for legacy callers.
    const planId = data.paypal_plan_id || process.env.PAYPAL_PLAN_ID;
    if (!planId) throw new Error("No PayPal plan configured for this plan");

    const userId = context.userId;
    const claims = context.claims as { email?: string; full_name?: string; name?: string } | undefined;
    const email = claims?.email || undefined;

    // Pre-provision the enterprise rows so the buyer's billing details are
    // safely stored server-side, and the webhook can find them via
    // customer_accounts.owner_user_id (keyed off custom_id = userId).
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;

    let ownerProfileId: string | null = null;
    let organizationId: string | null = null;
    try {
      const rows = await preProvisionEnterpriseRows(supabase, userId, data);
      ownerProfileId = rows.ownerProfileId;
      organizationId = rows.organizationId ?? null;
    } catch (err) {
      // Non-fatal: log and continue. The webhook can still create these
      // rows later; we just lose the "billing info in DB before checkout"
      // guarantee for this run.
      console.warn("[paypal] pre-provision failed (continuing)", {
        message: err instanceof Error ? err.message : String(err),
      });
    }

    // Best-effort subscriber name for the PayPal approval page.
    let givenName = "Enterprise";
    let surname = "Buyer";
    const displayName = (claims?.full_name || claims?.name || "").trim();
    if (displayName) {
      const parts = displayName.split(/\s+/);
      givenName = parts[0] || givenName;
      surname = parts.slice(1).join(" ") || surname;
    }

    const origin = getAppOrigin();
    const returnUrl = `${origin}/settings?tab=billing&billing=success&provider=paypal`;
    const cancelUrl = `${origin}/settings?tab=billing&billing=cancelled&provider=paypal`;
    if (!/^https:\/\//i.test(returnUrl) || !/^https:\/\//i.test(cancelUrl)) {
      throw new Error(
        `PayPal requires HTTPS return/cancel URLs (got ${returnUrl})`,
      );
    }

    // custom_id: just the userId (36 chars UUID — always safe under PayPal's
    // 127-char cap; never truncated to invalid JSON like the old code was).
    const customId = ownerProfileId || userId;

    // 15-day free trial, matching the Stripe checkout flow: the first PayPal
    // payment is only taken when the trial ends. PayPal charges on
    // `start_time`, so we push it 15 days out. The trial is one-time only —
    // repeat subscribers (plan change / payment-method change) start billing
    // immediately.
    const { isTrialEligible } = await import("./trial-eligibility.server");
    const trialEligible = ownerProfileId
      ? await isTrialEligible(supabase, { ownerProfileId, organizationId })
      : false;
    const trialEndsAt = new Date(Date.now() + 15 * 24 * 60 * 60 * 1000);

    const subscriptionBody: Record<string, any> = {
      plan_id: planId,
      custom_id: customId,
      ...(trialEligible ? { start_time: trialEndsAt.toISOString() } : {}),

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
    };

    console.log("[paypal] creating subscription", {
      plan_id: planId,
      custom_id: customId,
      returnUrl,
      cancelUrl,
      subscriber_email: email ? "set" : "none",
      subscriber_name: `${givenName} ${surname}`,
    });

    let sub: any;
    try {
      sub = await paypalFetch("/v1/billing/subscriptions", {
        method: "POST",
        debug: true,
        body: JSON.stringify(subscriptionBody),
      });
    } catch (err: any) {
      console.error("[paypal] subscription create failed", {
        message: err?.message,
        paypal: err?.paypal,
      });
      throw err;
    }

    const links = Array.isArray(sub?.links) ? sub.links : [];
    const approveUrl: string | undefined = links.find(
      (l: any) => l.rel === "approve",
    )?.href;
    console.log("[paypal] subscription created", {
      id: sub?.id,
      status: sub?.status,
      approveUrl,
      links: links.map((l: any) => ({ rel: l.rel, href: l.href })),
    });
    if (!approveUrl) {
      throw new Error("PayPal did not return an approve URL");
    }

    // Stash the pending subscription id on the organization so the
    // success-redirect finalizer can find it even if the webhook is late.
    try {
      if (ownerProfileId) {
        const { data: prof } = await supabase
          .from("profiles")
          .select("organization_id")
          .eq("id", ownerProfileId)
          .maybeSingle();
        const orgId = prof?.organization_id ?? null;
        if (orgId && sub?.id) {
          await supabase
            .from("organizations")
            .update({
              payment_provider: "paypal",
              paypal_subscription_id: sub.id,
              subscription_status: "approval_pending",
            })
            .eq("id", orgId);
        }
      }
    } catch (err) {
      console.warn("[paypal] failed to stash pending subscription id", err);
    }

    return { approveUrl, subscriptionId: sub.id as string };
  });
