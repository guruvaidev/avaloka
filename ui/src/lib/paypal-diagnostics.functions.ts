import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { getPayPalAccessToken, paypalFetch, getPayPalBase } from "./paypal-helpers";
import { getAppOrigin } from "./app-origin";

export const diagnosePayPalConfig = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async () => {
    const apiBase = getPayPalBase();
    const clientId = process.env.PAYPAL_CLIENT_ID || "";
    const planId = process.env.PAYPAL_PLAN_ID || "";
    const webhookId = process.env.PAYPAL_WEBHOOK_ID || "";
    const origin = getAppOrigin();
    const returnUrl = `${origin}/settings?billing=success&provider=paypal`;
    const cancelUrl = `${origin}/settings?billing=cancelled&provider=paypal`;

    const result: any = {
      apiBase,
      env: apiBase.includes("sandbox") ? "sandbox" : "live",
      clientIdSuffix: clientId ? clientId.slice(-6) : null,
      clientIdSet: Boolean(clientId),
      clientSecretSet: Boolean(process.env.PAYPAL_CLIENT_SECRET),
      planIdSet: Boolean(planId),
      webhookIdSet: Boolean(webhookId),
      origin,
      returnUrl,
      cancelUrl,
      urlsAreHttps:
        /^https:\/\//i.test(returnUrl) && /^https:\/\//i.test(cancelUrl),
      token: { ok: false as boolean, error: null as string | null },
      plan: null as any,
      product: null as any,
      webhook: null as any,
    };

    let token: string;
    try {
      token = await getPayPalAccessToken();
      result.token.ok = true;
    } catch (err: any) {
      result.token.error = err?.message || String(err);
      return result;
    }

    if (planId) {
      try {
        const plan = await paypalFetch(`/v1/billing/plans/${planId}`, { token });
        result.plan = {
          id: plan.id,
          status: plan.status,
          product_id: plan.product_id,
          name: plan.name,
          billing_cycles: (plan.billing_cycles || []).map((c: any) => ({
            tenure_type: c.tenure_type,
            sequence: c.sequence,
            total_cycles: c.total_cycles,
            frequency: c.frequency,
            price: c.pricing_scheme?.fixed_price,
          })),
        };
        if (plan.product_id) {
          try {
            const product = await paypalFetch(
              `/v1/catalogs/products/${plan.product_id}`,
              { token },
            );
            result.product = {
              id: product.id,
              name: product.name,
              type: product.type,
            };
          } catch (err: any) {
            result.product = {
              error: err?.message,
              paypal: err?.paypal ?? null,
            };
          }
        }
      } catch (err: any) {
        result.plan = { error: err?.message, paypal: err?.paypal ?? null };
      }
    }

    if (webhookId) {
      try {
        const wh = await paypalFetch(
          `/v1/notifications/webhooks/${webhookId}`,
          { token },
        );
        result.webhook = {
          id: wh.id,
          url: wh.url,
          event_count: (wh.event_types || []).length,
        };
      } catch (err: any) {
        result.webhook = { error: err?.message, paypal: err?.paypal ?? null };
      }
    }

    return result;
  });
