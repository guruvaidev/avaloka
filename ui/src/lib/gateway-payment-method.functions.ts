import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { resolvePaymentProfile } from "./payment-profile.server";
import {
  loadGatewayPaymentMethod,
  type GatewayPaymentMethod,
} from "./gateway-payment-method.server";

export type { GatewayPaymentMethod };

/**
 * Payment method attached to the active subscription, read live from the
 * gateway (Stripe / PayPal). No local payment_methods rows involved.
 */
export const getActiveGatewayPaymentMethod = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }): Promise<GatewayPaymentMethod | null> => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;
    const { ownerProfileId, organizationId } = await resolvePaymentProfile(
      supabase,
      context.userId,
      context.claims,
    );
    return loadGatewayPaymentMethod(supabase, { ownerProfileId, organizationId });
  });
