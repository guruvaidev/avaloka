// Single source of truth for the Supabase project this deployment uses.
// An external URL and anon key must always be supplied as a pair. When neither
// is configured, use the self-hosted local development pair.
//
// AUTH_MODE=local (the quickstart, no Supabase at all): the API signs in one
// local account and serves the few auth endpoints supabase-js calls
// (app/api/local_auth_routes.py). The browser reaches them same-origin through
// /api/public/local-auth, which the UI server forwards to the API; the UI
// server itself calls the API directly. So the Supabase client, and every
// page that uses it, works unchanged.

import { runtimeEnv } from "@/lib/runtime-env";
import { BACKEND_API_BASE } from "@/lib/api/backend-config";

export const isLocalAuth = () => runtimeEnv.AUTH_MODE.trim().toLowerCase() === "local";

/** Where supabase-js should send auth calls in local mode, on this side of the wire. */
export function localAuthUrl(): string {
  return typeof window !== "undefined"
    ? `${window.location.origin}/api/public/local-auth`
    : `${BACKEND_API_BASE}/local-auth`;
}

// Not a secret and not checked: supabase-js insists on a key string.
const LOCAL_MODE_ANON_KEY = "avaloka-local-auth";

const LOCAL_SUPABASE_URL = "http://localhost:30091";
const LOCAL_SUPABASE_ANON_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlLWRlbW8iLCJpYXQiOjE2NDE3NjkyMDAsImV4cCI6MTc5OTUzNTYwMH0.F_rDxRTPE8OU83L_CNgEGXfmirMXmMMugT29Cvc8ygQ";

if (Boolean(runtimeEnv.SUPABASE_URL) !== Boolean(runtimeEnv.SUPABASE_ANON_KEY)) {
  throw new Error("SUPABASE_URL and SUPABASE_ANON_KEY must be configured together.");
}
// The API refuses local mode while SUPABASE_URL is set; say so here too rather
// than quietly signing in against one and verifying against the other.
if (isLocalAuth() && runtimeEnv.SUPABASE_URL) {
  throw new Error("AUTH_MODE=local and SUPABASE_URL are mutually exclusive.");
}

export const SUPABASE_URL = isLocalAuth()
  ? localAuthUrl()
  : runtimeEnv.SUPABASE_URL || LOCAL_SUPABASE_URL;
export const SUPABASE_INCLUSTER_URL = isLocalAuth()
  ? `${BACKEND_API_BASE}/local-auth`
  : runtimeEnv.SUPABASE_INCLUSTER_URL || SUPABASE_URL;
export const SUPABASE_ANON_KEY = isLocalAuth()
  ? LOCAL_MODE_ANON_KEY
  : runtimeEnv.SUPABASE_ANON_KEY || LOCAL_SUPABASE_ANON_KEY;
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
