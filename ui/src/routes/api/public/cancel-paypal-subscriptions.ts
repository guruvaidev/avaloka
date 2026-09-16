import { createFileRoute } from "@tanstack/react-router";
import { paypalFetch } from "@/lib/paypal-helpers";

/**
 * Cron-safe endpoint that finalizes PayPal cancellations that were scheduled
 * via Settings → Billing → Cancel Subscription.
 *
 * PayPal has no native "cancel at period end" operation, so we suspend the
 * subscription at the moment the user clicks cancel, and then actually cancel
 * it in PayPal once the current billing period (or trial) ends. This endpoint
 * finds all PayPal subscriptions flagged for end-of-period cancellation whose
 * period has passed, cancels them at PayPal, and marks them as canceled locally.
 *
 * Call with a shared secret via `Authorization: Bearer <CRON_SECRET>` to prevent
 * public abuse. Returns counts so the caller can log/alert.
 */
export const Route = createFileRoute("/api/public/cancel-paypal-subscriptions")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        const cronSecret = process.env.CRON_SECRET;
        const auth = request.headers.get("Authorization") ?? "";
        if (cronSecret && auth !== `Bearer ${cronSecret}`) {
          return new Response("Unauthorized", { status: 401 });
        }

        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
        const supabase = supabaseAdmin as any;

        const { data: rows, error } = await supabase
          .from("subscriptions")
          .select("id, paypal_subscription_id, current_period_end, trial_ends_at, organization_id")
          .eq("payment_provider", "paypal")
          .eq("cancel_at_period_end", true)
          .not("status", "in", "(canceled,expired)");

        if (error) {
          console.error("[cron-paypal-cancel] lookup failed", error.message);
          return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { "Content-Type": "application/json" },
          });
        }

        const candidates = (rows ?? []) as any[];
        const now = Date.now();
        const results = { canceled: 0 as number, skipped: 0 as number, failed: 0 as number };

        for (const row of candidates) {
          const periodEnd = (row.current_period_end ?? row.trial_ends_at) as string | null;
          const periodEndMs = periodEnd ? new Date(periodEnd).getTime() : 0;
          if (periodEndMs > now) {
            results.skipped++;
            continue;
          }

          const paypalSubId = row.paypal_subscription_id as string | null;
          if (!paypalSubId) {
            results.skipped++;
            continue;
          }

          try {
            await paypalFetch(`/v1/billing/subscriptions/${paypalSubId}/cancel`, {
              method: "POST",
              body: JSON.stringify({ reason: "End of billing period reached" }),
            });

            await supabase
              .from("subscriptions")
              .update({ status: "canceled", canceled_at: new Date().toISOString() })
              .eq("id", row.id);

            if (row.organization_id) {
              await supabase
                .from("organizations")
                .update({
                  subscription_status: "canceled",
                  subscription_active: false,
                  canceled_at: new Date().toISOString(),
                })
                .eq("id", row.organization_id);
            }

            results.canceled++;
          } catch (err: any) {
            console.warn(
              "[cron-paypal-cancel] failed to cancel subscription",
              row.id,
              paypalSubId,
              err?.message,
            );
            results.failed++;
          }
        }

        return new Response(JSON.stringify(results), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      },
    },
  },
});
