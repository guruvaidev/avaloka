import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { resolvePaymentProfile } from "./payment-profile.server";
import { assertAutoPayAuthorized } from "./billing-details";

import { ensureStripeCustomer, getStripe } from "./stripe-cards.server";

export const createStripeSetupIntent = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const stripe = getStripe();

    const { ownerProfileId, profile } = await resolvePaymentProfile(supabase, context.userId, context.claims);
    const email =
      (profile?.company_email as string | null) ??
      (context.claims?.email as string | null) ??
      null;
    const name =
      (profile?.full_name as string | null) ??
      ([profile?.first_name, profile?.last_name].filter(Boolean).join(" ") ||
        null);

    const customerId = await ensureStripeCustomer(supabase, stripe, {
      ownerProfileId,
      email,
      name,
    });

    const setupIntent = await stripe.setupIntents.create({
      customer: customerId,
      payment_method_types: ["card"],
      usage: "off_session",
    });

    return {
      client_secret: setupIntent.client_secret,
      customer_id: customerId,
    };
  });

export const saveStripePaymentMethod = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: { payment_method_id: string; customer_id: string; cardholder?: string | null; auto_pay_authorized?: boolean }) => {
    const pm = String(data?.payment_method_id ?? "").trim();
    const cust = String(data?.customer_id ?? "").trim();
    if (!pm || !cust) throw new Error("Missing payment_method_id or customer_id");
    assertAutoPayAuthorized(data?.auto_pay_authorized);
    const cardholder = String(data?.cardholder ?? "").trim();
    if (!cardholder) throw new Error("Cardholder name is required.");
    if (cardholder.length < 2 || !/\p{L}/u.test(cardholder)) {
      throw new Error("Enter the cardholder name exactly as printed on the card.");
    }
    return {
      payment_method_id: pm,
      customer_id: cust,
      cardholder,
    };
  })

  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const stripe = getStripe();

    const { ownerProfileId, organizationId } = await resolvePaymentProfile(
      supabase,
      context.userId,
      context.claims,
    );

    // Make sure the PM is attached to our customer (SetupIntent already does
    // this on confirmation, but calling attach again is idempotent).
    try {
      await stripe.paymentMethods.attach(data.payment_method_id, {
        customer: data.customer_id,
      });
    } catch (err: any) {
      // "already attached" errors are safe to swallow
      if (!/already/i.test(err?.message ?? "")) throw err;
    }

    const pm = await stripe.paymentMethods.retrieve(data.payment_method_id);
    const card = pm.card;

    const { error } = await supabase.from("payment_methods").insert({
      owner_profile_id: ownerProfileId,
      organization_id: organizationId,
      provider: "stripe",
      provider_customer_id: data.customer_id,
      provider_payment_method_id: pm.id,
      brand: card?.brand ?? null,
      last4: card?.last4 ?? null,
      exp_month: card?.exp_month ?? null,
      exp_year: card?.exp_year ?? null,
      cardholder: data.cardholder ?? pm.billing_details?.name ?? null,
      status: "inactive", // user activates manually from the Billing tab
      is_default: false,
      metadata: {
        funding: card?.funding ?? null,
        country: card?.country ?? null,
      },
    });
    if (error) throw new Error(error.message);

    return { ok: true as const };
  });
