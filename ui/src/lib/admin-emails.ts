// Who is an administrator comes from configuration, never from source code: the
// server's ADMIN_EMAILS environment variable, a comma-separated list of addresses
// (set by the Helm chart from admin.emails). Unset or empty means nobody is an
// admin. There is no fallback address and no "first user becomes admin".
//
// Server-only in practice: the browser has no process.env, so a call that slips
// into client code sees an empty list and answers "not an admin".

/** Normalised (trimmed, lower-cased) addresses from a comma-separated list. */
export function parseAdminEmails(raw: string | null | undefined): Set<string> {
  const out = new Set<string>();
  for (const part of String(raw ?? "").split(",")) {
    const email = part.trim().toLowerCase();
    if (email) out.add(email);
  }
  return out;
}

function adminEmailsFromEnv(): string | undefined {
  return typeof process !== "undefined" ? process.env?.ADMIN_EMAILS : undefined;
}

/** True only when `email` is listed in ADMIN_EMAILS (or in `raw`, when given). */
export function isAdminEmail(
  email: unknown,
  raw: string | null | undefined = adminEmailsFromEnv(),
): boolean {
  const normalized = String(email ?? "").trim().toLowerCase();
  if (!normalized) return false;
  return parseAdminEmails(raw).has(normalized);
}
