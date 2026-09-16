// Where this deployment's Supabase project and public payment identifiers come
// from — all of them, at runtime, from the environment.
//
// These were previously literals pointing at Avaloka's own hosted project. The
// anon key is publishable, so that was not a credential leak; it was worse in a
// different way. Anyone self-hosting who did not set the variables got a UI
// authenticating against our project — their accounts in our database, their
// traffic on our bill, and no error to tell them. A default that silently
// borrows someone else's infrastructure is not a convenience.
//
// So: no fallbacks. Unset configuration surfaces as a setup message naming the
// variable, which is the difference between "this needs configuring" and a
// blank screen.
//
// See README.md → "Setting up your own credentials".

import { runtimeEnv } from "@/lib/runtime-env";

export const SUPABASE_URL = runtimeEnv.SUPABASE_URL || "";
export const SUPABASE_INCLUSTER_URL = runtimeEnv.SUPABASE_INCLUSTER_URL || SUPABASE_URL;
export const SUPABASE_ANON_KEY = runtimeEnv.SUPABASE_ANON_KEY || "";
export const SUPABASE_STORAGE_KEY = `sb-auth-token`;
export const SUPABASE_ADMIN_KEY_ENV = "PRIMARY_SUPABASE_SERVICE_ROLE_KEY";

/** Variables that must be set before authentication can work. */
export function missingSupabaseConfig(): string[] {
  const missing: string[] = [];
  if (!SUPABASE_URL) missing.push("SUPABASE_URL");
  if (!SUPABASE_ANON_KEY) missing.push("SUPABASE_ANON_KEY");
  return missing;
}

export const isSupabaseConfigured = () => missingSupabaseConfig().length === 0;

/**
 * Throw with something a person can act on.
 *
 * Called where a client is constructed rather than at module load: an
 * import-time throw white-screens the whole application, including the page
 * that would explain what to set.
 */
export function assertSupabaseConfigured(): void {
  const missing = missingSupabaseConfig();
  if (missing.length === 0) return;
  throw new Error(
    `Avaloka is not configured yet: ${missing.join(" and ")} ${
      missing.length === 1 ? "is" : "are"
    } unset.\n\n` +
      `Avaloka uses Supabase for sign-in, and you supply your own project — we do ` +
      `not ship one, because a default would point your users at somebody else's ` +
      `database.\n\n` +
      `Create a free project at https://supabase.com, then set SUPABASE_URL and ` +
      `SUPABASE_ANON_KEY. Both values are safe to expose in a browser.\n\n` +
      `Step-by-step: README.md → "Setting up your own credentials".`,
  );
}

// ─── Payment gateway PUBLIC identifiers ──────────────────────────────────────
// Only publishable values belong here — this file is in the browser bundle. The
// matching SECRETS (STRIPE_SECRET_KEY, PAYPAL_CLIENT_SECRET) stay in server-only
// env and are read inside server handlers. Never move a secret into this file.
//
// Empty by default. Billing is a commercial-edition concern; an open-source
// deployment has nothing to charge for, and a hardcoded key here would have
// pointed a self-hosted instance's payment flows at our Stripe account.

export const STRIPE_PUBLISHABLE_KEY = runtimeEnv.STRIPE_PUBLISHABLE_KEY || "";
export const PAYPAL_CLIENT_ID = runtimeEnv.PAYPAL_CLIENT_ID || "";
export const isBillingConfigured = () => Boolean(STRIPE_PUBLISHABLE_KEY);
