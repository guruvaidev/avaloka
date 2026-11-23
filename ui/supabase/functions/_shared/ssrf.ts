// SSRF guard for the one place an edge function fetches a customer-supplied URL:
// the organization logo embedded in emailed report PDFs.
//
// organizations.logo_url is written by updateOrganizationProfile (an org owner or
// a global admin), and the edge function used to hand it straight to `fetch(url)`
// with no scheme, host or address check, redirects followed, no size cap, and a
// content type guessed from the URL suffix. The edge functions run in-cluster
// (deploy/helm/avaloka/templates/supabase.yaml) and the chart has no egress
// NetworkPolicy at all, so the reachable surface was the whole cluster network.
//
// Five clauses, all of which have to hold or the guard is decorative:
//   1. https: only.
//   2. No IP literals; resolve the host and reject every private, loopback,
//      link-local, unique-local, CGNAT, multicast or otherwise non-global answer.
//   3. redirect: "manual" — every hop is re-validated from clause 1.
//   4. Response size capped and content type restricted to PNG/JPEG, confirmed
//      against the magic bytes rather than the URL suffix.
//   5. Any failure returns { ok: false } and nothing else. The caller omits the
//      logo; no response body is ever handed back for embedding.
//
// Known limits, stated rather than papered over:
//   - resolve-then-fetch is inherently TOCTOU: `fetch` resolves the name again,
//     so a rebinding resolver could answer differently the second time. Closing
//     that needs connect-to-the-checked-IP, which Deno's fetch does not offer.
//     Clause 4 keeps the consequence to "image-shaped bytes only".
//   - If DNS resolution is unavailable, clause 2 cannot be evaluated, so the
//     fetch fails closed and the logo is omitted. That is deliberate, and it
//     is a fallback rather than the expected case: measured in a user worker
//     on supabase/edge-runtime:v1.67.4 (Deno 1.45.2), Deno.resolveDns is both
//     present and permitted -- it answers rather than throwing PermissionDenied,
//     which it would do if net permission were withheld.
//   - redirect: "manual" returns the real 3xx there, with its status and
//     Location readable, which is what the per-hop re-check needs; also
//     measured on that image, against a 302 from a throwaway server. It is not
//     filtered to an opaque redirect. Were some future runtime to filter it,
//     the response would fall through to the non-2xx branch and the logo would
//     be omitted -- still nothing followed unchecked, only a worse outcome for
//     a redirecting logo URL. Both facts are pinned to that image; re-check if
//     deploy/docker/Dockerfile.functions bumps it.

export type ResolveDns = (hostname: string) => Promise<string[]>;

export type UrlCheck =
  | { ok: true; url: URL; addresses: string[] }
  | { ok: false; reason: string };

export type ImageFetch =
  | { ok: true; bytes: Uint8Array; kind: "png" | "jpg" }
  | { ok: false; reason: string };

export const MAX_IMAGE_BYTES = 2 * 1024 * 1024;
export const MAX_REDIRECTS = 3;
export const FETCH_TIMEOUT_MS = 6000;

/** Content types we are willing to embed. Anything else is omitted. */
const ALLOWED_CONTENT_TYPES = new Set(["image/png", "image/jpeg", "image/jpg"]);

// ---------------------------------------------------------------- IP handling

/** Parse a strict dotted-quad into a 32-bit integer, or null. */
export function parseIpv4(host: string): number | null {
  const parts = host.split(".");
  if (parts.length !== 4) return null;
  let value = 0;
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) return null;
    const octet = Number(part);
    if (octet > 255) return null;
    value = value * 256 + octet;
  }
  return value;
}

/** Parse an IPv6 literal (no brackets) into eight 16-bit groups, or null. */
export function parseIpv6(host: string): number[] | null {
  if (host.includes("%")) return null; // zone id — never a global address
  let text = host;

  // A dotted-quad tail (::ffff:192.168.0.1, 64:ff9b::192.168.0.1) becomes two
  // hextets in place, so the rest of the parse has one shape to handle.
  const lastColon = text.lastIndexOf(":");
  if (lastColon >= 0 && text.slice(lastColon + 1).includes(".")) {
    const v4 = parseIpv4(text.slice(lastColon + 1));
    if (v4 === null) return null;
    const hi = Math.floor(v4 / 65536).toString(16);
    const lo = (v4 % 65536).toString(16);
    text = `${text.slice(0, lastColon + 1)}${hi}:${lo}`;
  }

  const doubleColon = text.indexOf("::");
  let head: number[];
  let rest: number[];
  if (doubleColon >= 0) {
    if (text.indexOf("::", doubleColon + 1) >= 0) return null;
    const left = text.slice(0, doubleColon);
    const right = text.slice(doubleColon + 2);
    const leftGroups = left ? left.split(":") : [];
    const rightGroups = right ? right.split(":") : [];
    head = [];
    rest = [];
    for (const g of leftGroups) {
      const n = hextet(g);
      if (n === null) return null;
      head.push(n);
    }
    for (const g of rightGroups) {
      const n = hextet(g);
      if (n === null) return null;
      rest.push(n);
    }
    const filled = head.length + rest.length;
    if (filled > 8) return null;
    return [...head, ...new Array(8 - filled).fill(0), ...rest];
  }

  const groups: number[] = [];
  for (const g of text ? text.split(":").filter((s) => s !== "") : []) {
    const n = hextet(g);
    if (n === null) return null;
    groups.push(n);
  }
  return groups.length === 8 ? groups : null;
}

function hextet(group: string): number | null {
  if (!/^[0-9a-fA-F]{1,4}$/.test(group)) return null;
  return parseInt(group, 16);
}

type V4Range = { prefix: number; bits: number; why: string };

// Every IPv4 block that is not globally routable unicast.
const BLOCKED_V4: V4Range[] = [
  { prefix: 0x00000000, bits: 8, why: "this-network 0.0.0.0/8" },
  { prefix: 0x0a000000, bits: 8, why: "private 10.0.0.0/8" },
  { prefix: 0x64400000, bits: 10, why: "carrier-grade NAT 100.64.0.0/10" },
  { prefix: 0x7f000000, bits: 8, why: "loopback 127.0.0.0/8" },
  { prefix: 0xa9fe0000, bits: 16, why: "link-local 169.254.0.0/16" },
  { prefix: 0xac100000, bits: 12, why: "private 172.16.0.0/12" },
  { prefix: 0xc0000000, bits: 24, why: "IETF protocol assignments 192.0.0.0/24" },
  { prefix: 0xc0000200, bits: 24, why: "documentation 192.0.2.0/24" },
  { prefix: 0xc0586300, bits: 24, why: "6to4 relay anycast (RFC 3068)" },
  { prefix: 0xc0a80000, bits: 16, why: "private 192.168.0.0/16" },
  { prefix: 0xc6120000, bits: 15, why: "benchmarking 198.18.0.0/15" },
  { prefix: 0xc6336400, bits: 24, why: "documentation 198.51.100.0/24" },
  { prefix: 0xcb007100, bits: 24, why: "documentation 203.0.113.0/24" },
  { prefix: 0xe0000000, bits: 4, why: "multicast (RFC 5771)" },
  { prefix: 0xf0000000, bits: 4, why: "reserved 240.0.0.0/4" },
];

type V6Range = { prefix: number[]; bits: number; why: string };

const BLOCKED_V6: V6Range[] = [
  { prefix: [0, 0, 0, 0, 0, 0, 0, 0], bits: 128, why: "unspecified ::" },
  { prefix: [0, 0, 0, 0, 0, 0, 0, 1], bits: 128, why: "loopback ::1" },
  { prefix: [0x0100, 0, 0, 0, 0, 0, 0, 0], bits: 64, why: "discard-only 100::/64" },
  { prefix: [0x2001, 0x0db8, 0, 0, 0, 0, 0, 0], bits: 32, why: "documentation 2001:db8::/32" },
  { prefix: [0x2001, 0, 0, 0, 0, 0, 0, 0], bits: 32, why: "Teredo 2001::/32" },
  { prefix: [0xfc00, 0, 0, 0, 0, 0, 0, 0], bits: 7, why: "unique-local fc00::/7 (includes fd00::/8)" },
  { prefix: [0xfe80, 0, 0, 0, 0, 0, 0, 0], bits: 10, why: "link-local fe80::/10" },
  { prefix: [0xfec0, 0, 0, 0, 0, 0, 0, 0], bits: 10, why: "site-local fec0::/10" },
  { prefix: [0xff00, 0, 0, 0, 0, 0, 0, 0], bits: 8, why: "multicast ff00::/8" },
];

function v4InRange(value: number, range: V4Range): boolean {
  const mask = range.bits === 0 ? 0 : (0xffffffff << (32 - range.bits)) >>> 0;
  return ((value >>> 0) & mask) === ((range.prefix >>> 0) & mask);
}

function v6InRange(groups: number[], range: V6Range): boolean {
  let remaining = range.bits;
  for (let i = 0; i < 8 && remaining > 0; i++) {
    const take = Math.min(16, remaining);
    const mask = take === 0 ? 0 : (0xffff << (16 - take)) & 0xffff;
    if ((groups[i] & mask) !== (range.prefix[i] & mask)) return false;
    remaining -= take;
  }
  return true;
}

/**
 * Why `address` must not be contacted, or null if it is a global unicast address.
 * Accepts IPv4 dotted-quad and IPv6 (with or without brackets).
 */
export function blockedReason(address: string): string | null {
  const host = address.startsWith("[") && address.endsWith("]")
    ? address.slice(1, -1)
    : address;

  const v4 = parseIpv4(host);
  if (v4 !== null) {
    for (const range of BLOCKED_V4) {
      if (v4InRange(v4, range)) return range.why;
    }
    return null;
  }

  const v6 = parseIpv6(host);
  if (v6 !== null) {
    // IPv4-mapped (::ffff:0:0/96), NAT64 (64:ff9b::/96) and 6to4 (2002::/16)
    // all wrap a v4 address. Judge the wrapped address, not the wrapper.
    const embedded = embeddedV4(v6);
    if (embedded !== null) {
      for (const range of BLOCKED_V4) {
        if (v4InRange(embedded, range)) return `${range.why} (via IPv6 ${host})`;
      }
    }
    for (const range of BLOCKED_V6) {
      if (v6InRange(v6, range)) return range.why;
    }
    return null;
  }

  return "not a recognizable IP address";
}

function embeddedV4(g: number[]): number | null {
  const isMapped = g[0] === 0 && g[1] === 0 && g[2] === 0 && g[3] === 0 && g[4] === 0 && g[5] === 0xffff;
  const isNat64 = g[0] === 0x0064 && g[1] === 0xff9b;
  if (isMapped || isNat64) return g[6] * 65536 + g[7];
  if (g[0] === 0x2002) return g[1] * 65536 + g[2]; // 6to4
  return null;
}

// --------------------------------------------------------------- URL checking

/** True when `hostname` (as the URL parser normalized it) is an IP literal. */
export function isIpLiteral(hostname: string): boolean {
  if (hostname.startsWith("[")) return true;
  if (parseIpv4(hostname) !== null) return true;
  // The URL parser normalizes 0x7f000001 / 2130706433 / 0177.1 into dotted quad,
  // so anything still all-digits here is an address shape we do not understand.
  return /^[0-9]+$/.test(hostname);
}

/**
 * Clauses 1 and 2 for one URL. Does not fetch.
 * `resolve` is injected so this is testable with string inputs and no network.
 */
export async function checkUrl(raw: string, resolve: ResolveDns): Promise<UrlCheck> {
  if (typeof raw !== "string" || !raw.trim()) return { ok: false, reason: "empty url" };
  if (raw.length > 2048) return { ok: false, reason: "url longer than 2048 characters" };

  let url: URL;
  try {
    url = new URL(raw.trim());
  } catch {
    return { ok: false, reason: "not a parsable absolute url" };
  }

  // Clause 1: https only. This also closes file:, http:, data:, blob: and
  // gopher: without needing to know what the runtime's fetch does with them.
  if (url.protocol !== "https:") {
    return { ok: false, reason: `scheme ${url.protocol} is not https:` };
  }
  if (url.username || url.password) {
    return { ok: false, reason: "url embeds credentials" };
  }

  const hostname = url.hostname.toLowerCase();
  if (!hostname) return { ok: false, reason: "url has no host" };

  // Clause 2a: no IP literals. A logo is served from a name, and a literal is
  // how every published SSRF payload addresses an internal service.
  if (isIpLiteral(hostname)) {
    return { ok: false, reason: `host is an IP literal (${hostname})` };
  }
  if (hostname === "localhost" || hostname.endsWith(".localhost")) {
    return { ok: false, reason: "host is localhost" };
  }

  // Clause 2b: resolve-then-check. Every answer must be global unicast; one
  // bad answer rejects the URL.
  let addresses: string[];
  try {
    addresses = await resolve(hostname);
  } catch (err) {
    return { ok: false, reason: `dns resolution failed: ${String(err)}` };
  }
  if (!addresses.length) return { ok: false, reason: "host did not resolve" };

  for (const address of addresses) {
    const why = blockedReason(address);
    if (why !== null) {
      return { ok: false, reason: `${hostname} resolves to ${address}: ${why}` };
    }
  }

  return { ok: true, url, addresses };
}

// ------------------------------------------------------------------- fetching

export type FetchDeps = {
  fetchFn?: typeof fetch;
  resolve?: ResolveDns;
  maxBytes?: number;
  maxRedirects?: number;
  timeoutMs?: number;
};

/** Deno's resolver, when there is one. Absent resolver => clause 2 fails closed. */
export const denoResolve: ResolveDns = async (hostname) => {
  // deno-lint-ignore no-explicit-any
  const deno = (globalThis as any).Deno;
  if (!deno?.resolveDns) throw new Error("no dns resolver available");
  const settled = await Promise.allSettled([
    deno.resolveDns(hostname, "A"),
    deno.resolveDns(hostname, "AAAA"),
  ]);
  const out: string[] = [];
  for (const r of settled) {
    if (r.status === "fulfilled" && Array.isArray(r.value)) out.push(...r.value);
  }
  if (!out.length) {
    const why = settled.find((r) => r.status === "rejected") as PromiseRejectedResult | undefined;
    if (why) throw why.reason;
  }
  return out;
};

function sniffImage(bytes: Uint8Array): "png" | "jpg" | null {
  if (
    bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e &&
    bytes[3] === 0x47 && bytes[4] === 0x0d && bytes[5] === 0x0a && bytes[6] === 0x1a && bytes[7] === 0x0a
  ) return "png";
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "jpg";
  return null;
}

async function readCapped(res: Response, maxBytes: number): Promise<Uint8Array | null> {
  const declared = Number(res.headers.get("content-length") ?? "");
  if (Number.isFinite(declared) && declared > maxBytes) return null;

  const body = res.body;
  if (!body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    return buf.length > maxBytes ? null : buf;
  }

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.length;
      // Clause 4: stop reading rather than trusting content-length.
      if (total > maxBytes) {
        await reader.cancel().catch(() => {});
        return null;
      }
      chunks.push(value);
    }
  } finally {
    try { reader.releaseLock(); } catch { /* already released */ }
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

/**
 * Fetch an image from a customer-supplied URL, or refuse.
 *
 * Clause 5: there is no path through this function that returns bytes which did
 * not pass every check. A refusal carries a reason for the log and nothing else.
 */
export async function fetchImageSafely(raw: string, deps: FetchDeps = {}): Promise<ImageFetch> {
  const fetchFn = deps.fetchFn ?? fetch;
  const resolve = deps.resolve ?? denoResolve;
  const maxBytes = deps.maxBytes ?? MAX_IMAGE_BYTES;
  const maxRedirects = deps.maxRedirects ?? MAX_REDIRECTS;
  const timeoutMs = deps.timeoutMs ?? FETCH_TIMEOUT_MS;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    let target = raw;
    for (let hop = 0; hop <= maxRedirects; hop++) {
      // Clause 3: the check runs again on every hop, from clause 1 down.
      const checked = await checkUrl(target, resolve);
      if (!checked.ok) {
        return { ok: false, reason: hop === 0 ? checked.reason : `redirect hop ${hop}: ${checked.reason}` };
      }

      let res: Response;
      try {
        res = await fetchFn(checked.url.toString(), {
          redirect: "manual",
          signal: controller.signal,
          headers: { accept: "image/png,image/jpeg" },
        });
      } catch (err) {
        return { ok: false, reason: `fetch failed: ${String(err)}` };
      }

      if (res.status >= 300 && res.status < 400) {
        const location = res.headers.get("location");
        await res.body?.cancel().catch(() => {});
        if (!location) return { ok: false, reason: `redirect ${res.status} without location` };
        let next: string;
        try {
          next = new URL(location, checked.url).toString();
        } catch {
          return { ok: false, reason: "redirect location is not a usable url" };
        }
        target = next;
        continue;
      }

      if (!res.ok) {
        await res.body?.cancel().catch(() => {});
        return { ok: false, reason: `http ${res.status}` };
      }

      // Clause 4: declared type must be an image we accept...
      const contentType = (res.headers.get("content-type") ?? "").split(";")[0].trim().toLowerCase();
      if (!ALLOWED_CONTENT_TYPES.has(contentType)) {
        await res.body?.cancel().catch(() => {});
        return { ok: false, reason: `content-type ${contentType || "(absent)"} is not an accepted image type` };
      }

      const bytes = await readCapped(res, maxBytes);
      if (bytes === null) return { ok: false, reason: `body exceeds ${maxBytes} bytes` };
      if (!bytes.length) return { ok: false, reason: "empty body" };

      // ...and the bytes must actually be that image. The old code guessed the
      // kind from the URL suffix, so a text body behind a ".png" path was handed
      // to embedPng.
      const kind = sniffImage(bytes);
      if (kind === null) return { ok: false, reason: "body is not PNG or JPEG" };
      if (kind === "png" && contentType !== "image/png") {
        return { ok: false, reason: `content-type ${contentType} disagrees with PNG magic bytes` };
      }
      if (kind === "jpg" && contentType === "image/png") {
        return { ok: false, reason: "content-type image/png disagrees with JPEG magic bytes" };
      }

      return { ok: true, bytes, kind };
    }
    return { ok: false, reason: `more than ${maxRedirects} redirects` };
  } finally {
    clearTimeout(timer);
  }
}
