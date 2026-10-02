// Single source of truth for the Supabase project this deployment uses.
// An external URL and anon key must always be supplied as a pair. When neither
// is configured, use the self-hosted local development pair.

import { runtimeEnv } from "@/lib/runtime-env";

const LOCAL_SUPABASE_URL = "http://localhost:30091";
const LOCAL_SUPABASE_ANON_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlLWRlbW8iLCJpYXQiOjE2NDE3NjkyMDAsImV4cCI6MTc5OTUzNTYwMH0.F_rDxRTPE8OU83L_CNgEGXfmirMXmMMugT29Cvc8ygQ";

if (Boolean(runtimeEnv.SUPABASE_URL) !== Boolean(runtimeEnv.SUPABASE_ANON_KEY)) {
  throw new Error("SUPABASE_URL and SUPABASE_ANON_KEY must be configured together.");
}

export const SUPABASE_URL = runtimeEnv.SUPABASE_URL || LOCAL_SUPABASE_URL;
export const SUPABASE_INCLUSTER_URL = runtimeEnv.SUPABASE_INCLUSTER_URL || SUPABASE_URL;
export const SUPABASE_ANON_KEY =
  runtimeEnv.SUPABASE_ANON_KEY || LOCAL_SUPABASE_ANON_KEY;
export const SUPABASE_STORAGE_KEY = `sb-auth-token`;
export const SUPABASE_ADMIN_KEY_ENV = "PRIMARY_SUPABASE_SERVICE_ROLE_KEY";

// ─── Payment gateway PUBLIC identifiers ──────────────────────────────────────
// Only publishable / public values live here — this file is imported by the
// browser bundle. The matching SECRETS (STRIPE_SECRET_KEY, PAYPAL_CLIENT_SECRET)
// must stay in server-only env (process.env.*) and are read inside server
// function handlers only. Never move the secrets into this file.

// Empty by default. Billing is a commercial-edition concern; an open-source
// deployment has nothing to charge for, and a committed publishable key pointed
// every self-hosted instance's payment flows at one Stripe account.
export const STRIPE_PUBLISHABLE_KEY = runtimeEnv.STRIPE_PUBLISHABLE_KEY || "";
export const isBillingConfigured = () => Boolean(STRIPE_PUBLISHABLE_KEY);

// PayPal client ID is public (already embedded in the JS SDK URL when loaded).
// The value is provisioned via process.env.PAYPAL_CLIENT_ID on the server
// (Lovable secret). A public literal can be added here when needed for the
// browser SDK; leaving empty means server-side code is the sole reader.
export const PAYPAL_CLIENT_ID = "";
