import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { assertAutoPayAuthorized } from "./billing-details";
import { startPayPalMethodChange } from "./paypal-method.server";

/**
 * Change the PayPal payment method for an existing subscriber.
 * No billing form: company / address / VAT are reused from the first purchase.
 * The recurring auto-payment mandate must be re-confirmed for the new method.
 */
export const changePayPalPaymentMethod = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: { auto_pay_authorized?: boolean } | undefined) => {
    assertAutoPayAuthorized(data?.auto_pay_authorized);
    return { auto_pay_authorized: true as const };
  })
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    return startPayPalMethodChange(supabaseAdmin as any, context.userId, context.claims);
  });
