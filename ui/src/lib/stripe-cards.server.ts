import Stripe from "stripe";

export function getStripe(): Stripe {
  const key = process.env.STRIPE_TEST_API_KEY || process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("STRIPE_SECRET_KEY is not configured");
  return new Stripe(key, { httpClient: Stripe.createFetchHttpClient() } as any);
}

export async function ensureStripeCustomer(
  supabase: any,
  stripe: Stripe,
  args: { ownerProfileId: string; email: string | null; name: string | null },
): Promise<string> {
  const { data: existingMethod } = await supabase
    .from("payment_methods")
    .select("provider_customer_id")
    .eq("owner_profile_id", args.ownerProfileId)
    .eq("provider", "stripe")
    .not("provider_customer_id", "is", null)
    .limit(1)
    .maybeSingle();
  if (existingMethod?.provider_customer_id) return existingMethod.provider_customer_id;

  const { data: sub } = await supabase
    .from("subscriptions")
    .select("stripe_customer_id")
    .eq("owner_profile_id", args.ownerProfileId)
    .not("stripe_customer_id", "is", null)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (sub?.stripe_customer_id) return sub.stripe_customer_id;

  const customer = await stripe.customers.create({
    email: args.email ?? undefined,
    name: args.name ?? undefined,
    metadata: { owner_profile_id: args.ownerProfileId },
  });
  return customer.id;
}