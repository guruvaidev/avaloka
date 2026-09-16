import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import {
  getPayPalAccessToken,
  paypalFetch,
} from "./paypal-helpers";
import { getAppOrigin } from "./app-origin";

const PLAN_EVENTS = [
  "BILLING.SUBSCRIPTION.CREATED",
  "BILLING.SUBSCRIPTION.ACTIVATED",
  "BILLING.SUBSCRIPTION.UPDATED",
  "BILLING.SUBSCRIPTION.CANCELLED",
  "BILLING.SUBSCRIPTION.EXPIRED",
  "BILLING.SUBSCRIPTION.SUSPENDED",
  "BILLING.SUBSCRIPTION.PAYMENT.FAILED",
  "PAYMENT.SALE.COMPLETED",
  "PAYMENT.SALE.DENIED",
  "PAYMENT.SALE.REFUNDED",
];

export const bootstrapPayPalSandbox = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context: _context }) => {
    // Any authenticated user can run one-time PayPal provisioning.



    const price = process.env.PAYPAL_PLAN_PRICE || "199";
    const currency = process.env.PAYPAL_PLAN_CURRENCY || "USD";

    const token = await getPayPalAccessToken();

    // 1) Product
    const product = await paypalFetch("/v1/catalogs/products", {
      method: "POST",
      token,
      body: JSON.stringify({
        name: "Avaloka Enterprise",
        description: "Avaloka Enterprise subscription",
        type: "SERVICE",
        category: "SOFTWARE",
      }),
    });
    const productId = product.id as string;

    // 2) Plan (15-day trial + monthly recurring)
    const plan = await paypalFetch("/v1/billing/plans", {
      method: "POST",
      token,
      body: JSON.stringify({
        product_id: productId,
        name: "Avaloka Enterprise Monthly",
        description: "Enterprise plan with 15-day free trial",
        status: "ACTIVE",
        billing_cycles: [
          {
            frequency: { interval_unit: "DAY", interval_count: 15 },
            tenure_type: "TRIAL",
            sequence: 1,
            total_cycles: 1,
            pricing_scheme: {
              fixed_price: { value: "0", currency_code: currency },
            },
          },
          {
            frequency: { interval_unit: "MONTH", interval_count: 1 },
            tenure_type: "REGULAR",
            sequence: 2,
            total_cycles: 0,
            pricing_scheme: {
              fixed_price: { value: price, currency_code: currency },
            },
          },
        ],
        payment_preferences: {
          auto_bill_outstanding: true,
          setup_fee_failure_action: "CANCEL",
          payment_failure_threshold: 2,
        },
      }),
    });
    const planId = plan.id as string;

    // 3) Webhook. If one already exists on this URL, reuse it.
    const webhookUrl = `${getAppOrigin()}/api/public/paypal-webhook`;
    let webhookId: string | null = null;

    try {
      const created = await paypalFetch("/v1/notifications/webhooks", {
        method: "POST",
        token,
        body: JSON.stringify({
          url: webhookUrl,
          event_types: PLAN_EVENTS.map((name) => ({ name })),
        }),
      });
      webhookId = created.id as string;
    } catch (err) {
      // Fall back to list-and-match on duplicate-url errors.
      const list = await paypalFetch("/v1/notifications/webhooks", { token });
      const match = (list.webhooks || []).find((w: any) => w.url === webhookUrl);
      if (match?.id) {
        webhookId = match.id;
      } else {
        throw err;
      }
    }

    // 4) Delete stale webhooks pointing at other origins (e.g. old preview URL).
    const staleDeleted: string[] = [];
    const staleFailed: Array<{ id: string; url: string; error: string }> = [];
    try {
      const list = await paypalFetch("/v1/notifications/webhooks", { token });
      for (const wh of list.webhooks || []) {
        if (wh.id && wh.id !== webhookId && wh.url !== webhookUrl) {
          try {
            await paypalFetch(`/v1/notifications/webhooks/${wh.id}`, {
              method: "DELETE",
              token,
            });
            staleDeleted.push(`${wh.id} (${wh.url})`);
          } catch (e: any) {
            staleFailed.push({ id: wh.id, url: wh.url, error: e?.message || String(e) });
          }
        }
      }
    } catch (e) {
      console.warn("[paypal-bootstrap] could not enumerate webhooks for cleanup", e);
    }

    return {
      productId,
      planId,
      webhookId,
      webhookUrl,
      staleDeleted,
      staleFailed,
      currentEnvPlanId: process.env.PAYPAL_PLAN_ID || null,
      currentEnvWebhookId: process.env.PAYPAL_WEBHOOK_ID || null,
      apiBase: process.env.PAYPAL_API_BASE || "https://api-m.sandbox.paypal.com",
    };
  });
