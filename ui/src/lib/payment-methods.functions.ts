import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { resolvePaymentProfile } from "./payment-profile.server";

export type PaymentMethodRow = {
  id: string;
  owner_profile_id: string;
  organization_id: string | null;
  provider: "stripe" | "paypal";
  provider_customer_id: string | null;
  provider_payment_method_id: string | null;
  provider_subscription_id: string | null;
  brand: string | null;
  last4: string | null;
  exp_month: number | null;
  exp_year: number | null;
  cardholder: string | null;
  status: "active" | "inactive";
  is_default: boolean;
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
};

export const listMyPaymentMethods = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }): Promise<PaymentMethodRow[]> => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const { ownerProfileId } = await resolvePaymentProfile(supabase, context.userId, context.claims);
    const { data, error } = await supabase
      .from("payment_methods")
      .select("*")
      .eq("owner_profile_id", ownerProfileId)
      .order("status", { ascending: false }) // 'inactive' < 'active' lexically → desc puts active first
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    return (data ?? []) as PaymentMethodRow[];
  });

export const activatePaymentMethod = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: { id: string }) => {
    const id = String(data?.id ?? "").trim();
    if (!id) throw new Error("Missing payment method id");
    return { id };
  })
  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const { ownerProfileId, organizationId } = await resolvePaymentProfile(supabase, context.userId, context.claims);

    const { data: target, error: tErr } = await supabase
      .from("payment_methods")
      .select("*")
      .eq("id", data.id)
      .eq("owner_profile_id", ownerProfileId)
      .maybeSingle();
    if (tErr) throw new Error(tErr.message);
    if (!target) throw new Error("Payment method not found");

    // Deactivate all others first (partial unique index enforces one active).
    const { error: deactErr } = await supabase
      .from("payment_methods")
      .update({ status: "inactive" })
      .eq("owner_profile_id", ownerProfileId)
      .neq("id", data.id);
    if (deactErr) throw new Error(deactErr.message);

    const { error: actErr } = await supabase
      .from("payment_methods")
      .update({ status: "active" })
      .eq("id", data.id);
    if (actErr) throw new Error(actErr.message);

    // Sync active method onto subscriptions row (best-effort).
    try {
      const base = supabase
        .from("subscriptions")
        .select("id, organization_id, owner_profile_id")
        .order("created_at", { ascending: false })
        .limit(1);
      const subQ = organizationId
        ? base.eq("organization_id", organizationId)
        : base.eq("owner_profile_id", ownerProfileId).is("organization_id", null);
      const { data: sub } = await subQ.maybeSingle();
      if (sub?.id) {
        const patch: Record<string, any> = {};
        if (target.provider_customer_id) patch.stripe_customer_id = target.provider_customer_id;
        if (target.provider_subscription_id) patch.stripe_subscription_id = target.provider_subscription_id;
        if (Object.keys(patch).length) {
          await supabase.from("subscriptions").update(patch).eq("id", sub.id);
        }
      }
    } catch (err) {
      console.warn("[payment-methods] sync to subscriptions failed", err);
    }

    return { ok: true as const };
  });

export const removePaymentMethod = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: { id: string }) => {
    const id = String(data?.id ?? "").trim();
    if (!id) throw new Error("Missing payment method id");
    return { id };
  })
  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const { ownerProfileId } = await resolvePaymentProfile(supabase, context.userId, context.claims);
    const { error } = await supabase
      .from("payment_methods")
      .delete()
      .eq("id", data.id)
      .eq("owner_profile_id", ownerProfileId);
    if (error) throw new Error(error.message);
    return { ok: true as const };
  });
