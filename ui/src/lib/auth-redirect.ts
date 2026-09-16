// Single source of truth for auth redirect origins.
//
// Magic-link / invite emails must return the user to the environment they
// started from (Lovable preview, dev, test, localhost…). The destination is
// always derived from the running frontend's `window.location.origin`, and is
// re-validated here server-side against an explicit allow-list so a tampered
// value can never turn this into an open redirect.
//
// NOTE: `APP_ORIGIN` in `src/lib/app-origin.ts` is the canonical origin for
// server-side payment/webhook URLs only — never use it as an auth redirect.

export function isAllowedRedirectOrigin(value: string): boolean {
  try {
    const url = new URL(value);
    if (!/^https?:$/.test(url.protocol)) return false;
    const host = url.hostname.toLowerCase();
    if (host === "localhost" || host === "127.0.0.1") return true;
    if (url.protocol !== "https:") return false;
    return (
      host === "avaloka.ai" ||
      host.endsWith(".avaloka.ai") ||
      host.endsWith(".lovable.app") ||
      host.endsWith(".lovableproject.com") ||
      host.endsWith(".lovable.dev")
    );
  } catch {
    return false;
  }
}

/**
 * Validates and reduces a redirect URL to its origin, with a trailing slash.
 * The trailing slash matters: Supabase matches allow-list patterns such as
 * `https://host/**` against the FULL url, and a bare origin (no path) does
 * not satisfy them — the redirect is then silently replaced by the Site URL.
 * Throws when the origin is not on the allow-list.
 */
export function normalizeRedirectOrigin(value: string): string {
  const raw = String(value ?? "").trim();
  if (!isAllowedRedirectOrigin(raw)) throw new Error("Invalid redirect URL");
  return `${new URL(raw).origin}/`;
}

/** Non-throwing variant: returns undefined when the value is not allowed. */
export function safeRedirectOrigin(value?: string | null): string | undefined {
  if (!value) return undefined;
  try {
    return normalizeRedirectOrigin(value);
  } catch {
    return undefined;
  }
}
