// Validation for organizations.logo_url.
//
// The column is read back by the send-report-email edge function and fetched
// server-side to embed in the emailed PDF, so the value is a server-side fetch
// target, not just something an <img> renders. updateOrganizationProfile used
// to accept `String(logoRaw).trim().slice(0, 2048)` — a length limit and
// nothing else, no URL parse at all. The UI only ever setting this from a
// Supabase storage publicUrl is a client-side convention, not a control: the
// server function is callable directly.
//
// The edge function validates again at the point of fetch
// (ui/supabase/functions/_shared/ssrf.ts). Both checks exist on purpose; this
// one is not a substitute for that one, and that one is not a reason to skip
// this one. Rows written before this validation existed are only covered by
// the fetch-time check, which is the other reason it has to stay.
//
// Kept free of imports so it can be unit-tested without the server-function
// framework.

export const LOGO_URL_MAX_LENGTH = 2048;

/**
 * undefined  -> field absent, leave the column alone
 * null       -> clear the logo
 * string     -> a normalized https: URL
 * throws     -> anything else, with a message the settings form can show
 */
export function normalizeLogoUrl(raw: unknown): string | null | undefined {
  if (raw === undefined) return undefined;
  if (raw === null) return null;

  const text = String(raw).trim();
  if (!text) return null;
  if (text.length > LOGO_URL_MAX_LENGTH) {
    throw new Error(`Logo URL must be ${LOGO_URL_MAX_LENGTH} characters or fewer.`);
  }

  let url: URL;
  try {
    url = new URL(text);
  } catch {
    throw new Error("Logo URL must be a full URL, for example https://example.com/logo.png");
  }

  if (url.protocol !== "https:") {
    throw new Error("Logo URL must use https://.");
  }
  if (url.username || url.password) {
    throw new Error("Logo URL must not contain a username or password.");
  }
  if (!url.hostname) {
    throw new Error("Logo URL must include a host.");
  }

  return url.toString();
}
