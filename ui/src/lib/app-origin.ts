// Canonical app origin. All server-side URL construction (PayPal, Stripe,
// webhooks, return/cancel URLs) MUST use this — never fall back to the
// request Host/Origin header, which leaks preview URLs into provisioning.
//
// NOT for auth: magic-link / invite redirects are environment-aware and come
// from the caller's window.location.origin via `src/lib/auth-redirect.ts`.

//export const APP_ORIGIN = window.location.origin;
export const APP_ORIGIN =
  typeof window !== "undefined"
    ? window.location.origin
    : (process.env.SITE_URL ?? "https://test.avaloka.ai");

export function getAppOrigin(): string {
  const envUrl = process.env.APP_URL || process.env.VITE_APP_URL;
  if (envUrl && /^https:\/\//i.test(envUrl)) {
    return envUrl.replace(/\/$/, "");
  }
  return APP_ORIGIN;
}
