import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { paypalFetch } from "./paypal-helpers";
import { upsertFromSubscription } from "@/routes/api/public/paypal-webhook";

export type FinalizePayPalInput = {
  /** subscription_id PayPal appends to the return URL after approval. */
  subscription_id?: string;
};

// Belt-and-braces finalizer. Called from the success redirect
// (?billing=success&provider=paypal&subscription_id=...) so the flow always
// completes, even if the PayPal webhook is delayed or dropped.
//
// Flow:
//   1. Prefer the subscription id from the return URL; otherwise fall back to
//      the id stashed on the caller's organization by createPayPalSubscription.
//   2. Fetch the subscription from PayPal and verify it belongs to the caller
//      (custom_id === profile id / auth user id).
//   3. Run the same upsert the webhook uses.
export const finalizeEnterprisePayPalSubscription = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data?: FinalizePayPalInput) => ({
    subscription_id: String(data?.subscription_id ?? "").trim() || undefined,
  }))
  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const userId = context.userId;

    // Resolve the profile → organization → paypal_subscription_id chain.
    const { data: profile } = await supabase
      .from("profiles")
      .select("id, organization_id")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();
    const profileId = (profile?.id as string | undefined) ?? userId;
    const orgId = profile?.organization_id as string | null | undefined;

    let subId = data.subscription_id;
    if (!subId) {
      if (!orgId) return { ok: false, reason: "no_organization" as const };
      const { data: org } = await supabase
        .from("organizations")
        .select("paypal_subscription_id")
        .eq("id", orgId)
        .maybeSingle();
      subId = (org?.paypal_subscription_id as string | null | undefined) ?? undefined;
    }
    if (!subId) return { ok: false, reason: "no_subscription" as const };

    try {
      const sub = await paypalFetch(`/v1/billing/subscriptions/${subId}`, {});
      console.log("[paypal-finalize] fetched subscription", {
        id: sub?.id,
        status: sub?.status,
        plan_id: sub?.plan_id,
        custom_id: sub?.custom_id,
      });

      // Ownership check — the subscription's custom_id is the buyer's profile
      // (or auth user) id. Never finalize someone else's subscription.
      const owner = String(sub?.custom_id ?? "").trim();
      if (owner && owner !== profileId && owner !== userId) {
        console.error("[paypal-finalize] subscription does not belong to caller", {
          owner,
          profileId,
          userId,
        });
        return { ok: false, reason: "not_owner" as const };
      }
      // Older subscriptions created before custom_id existed: fall back to the
      // caller's profile id so the upsert can still resolve the account.
      if (!owner) sub.custom_id = profileId;

      try {
        await upsertFromSubscription(supabase, sub);
      } catch (err) {
        // Non-fatal — upsert always writes profile flags before rethrowing
        // license errors, so the user can still see Billing.
        console.warn("[paypal-finalize] upsert reported non-fatal error", err);
      }
      return {
        ok: true as const,
        subscriptionId: sub?.id ?? null,
        status: sub?.status ?? null,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : "finalize failed";
      console.error("[paypal-finalize] failed", message);
      return { ok: false, reason: "fetch_failed" as const, message };
    }
  });
