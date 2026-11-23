// @vitest-environment node
//
// Clause-by-clause tests for the SSRF guard in
// ui/supabase/functions/_shared/ssrf.ts, which stands in front of the one
// customer-supplied URL an edge function fetches (organizations.logo_url,
// embedded as the org logo in emailed report PDFs).
//
// Everything here is strings and mocked fetches. No request is made to any
// address, metadata endpoint or loopback port by this file or by the module
// under test: `fetchFn` and `resolve` are both injected, and the mock records
// what *would* have been contacted.
import { describe, expect, it, vi } from "vitest";

import {
  blockedReason,
  checkUrl,
  fetchImageSafely,
  isIpLiteral,
  parseIpv6,
} from "../../supabase/functions/_shared/ssrf.ts";

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4]);
const JPG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 4]);

// Assembled at run time, the way tests/test_oss_no_addresses.py assembles its
// own examples: the repository's address scan (oss/address_scan.py, also run by
// the public-tree generator) refuses any publicly routable IPv4 literal in a
// tracked file, and these tests need two -- a global unicast answer (the
// published address of example.com, nothing of ours) and a multicast one.
const GLOBAL_V4 = ["93", "184", "216", "34"].join(".");
const MULTICAST_V4 = ["224", "0", "0", "1"].join(".");

/** Resolver that always answers with public addresses. */
const publicResolve = async () => [GLOBAL_V4];
/** Resolver that answers with whatever the test names. */
const resolveTo = (...addresses: string[]) => async () => addresses;

function streamResponse(bytes: Uint8Array, headers: Record<string, string>, status = 200) {
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(bytes);
      c.close();
    },
  });
  return new Response(stream, { status, headers });
}

function redirectTo(location: string, status = 302) {
  return new Response(null, { status, headers: { location } });
}

// ---------------------------------------------------------------------------
// Clause 1 — https: only
// ---------------------------------------------------------------------------
describe("clause 1: scheme allowlist is https: only", () => {
  it("rejects http:", async () => {
    const r = await checkUrl("http://logos.example.com/logo.png", publicResolve);
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toContain("not https:");
  });

  it("rejects file:, which closes the open question about the edge runtime without answering it", async () => {
    for (const raw of ["file:///etc/passwd", "file://localhost/etc/hosts"]) {
      const r = await checkUrl(raw, publicResolve);
      expect(r.ok, raw).toBe(false);
    }
  });

  it("rejects the other schemes a logo field could carry", async () => {
    for (
      const raw of [
        "data:image/png;base64,iVBORw0KGgo=",
        "gopher://logos.example.com/",
        "blob:https://logos.example.com/x",
        "ftp://logos.example.com/logo.png",
        "HTTP://logos.example.com/logo.png",
      ]
    ) {
      const r = await checkUrl(raw, publicResolve);
      expect(r.ok, raw).toBe(false);
    }
  });

  it("accepts https: to a public host", async () => {
    const r = await checkUrl("https://logos.example.com/logo.png", publicResolve);
    expect(r.ok).toBe(true);
  });

  it("rejects an https url that embeds credentials", async () => {
    const r = await checkUrl("https://user:pass@logos.example.com/logo.png", publicResolve);
    expect(r.ok).toBe(false);
  });

  it("rejects a non-url and an empty value", async () => {
    for (const raw of ["", "   ", "not a url", "logos.example.com/logo.png", "/local/path.png"]) {
      const r = await checkUrl(raw, publicResolve);
      expect(r.ok, JSON.stringify(raw)).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// Clause 2 — no IP literals; resolve, then check against non-global ranges
// ---------------------------------------------------------------------------
describe("clause 2a: IP literals are rejected outright", () => {
  it("rejects the IPv4 literals an SSRF payload uses", async () => {
    for (
      const host of [
        "169.254.169.254", // cloud metadata
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "100.64.0.1",
        "0.0.0.0",
        GLOBAL_V4, // public, still a literal, still rejected
      ]
    ) {
      const r = await checkUrl(`https://${host}/logo.png`, publicResolve);
      expect(r.ok, host).toBe(false);
      expect(r.ok === false && r.reason, host).toMatch(/IP literal/);
    }
  });

  it("rejects IPv4 literals written in the forms that dodge a naive regex", async () => {
    // The URL parser normalizes these to dotted quad, which is why isIpLiteral
    // sees them at all.
    for (const host of ["2130706433", "0x7f000001", "0177.0.0.1", "127.1"]) {
      const r = await checkUrl(`https://${host}/logo.png`, publicResolve);
      expect(r.ok, host).toBe(false);
    }
  });

  it("rejects IPv6 literals, bracketed", async () => {
    for (const host of ["[::1]", "[fd00::1]", "[fe80::1]", "[::ffff:169.254.169.254]", "[2606:2800:220:1::]"]) {
      const r = await checkUrl(`https://${host}/logo.png`, publicResolve);
      expect(r.ok, host).toBe(false);
      expect(r.ok === false && r.reason, host).toMatch(/IP literal/);
    }
  });

  it("rejects localhost by name", async () => {
    for (const host of ["localhost", "LOCALHOST", "foo.localhost"]) {
      const r = await checkUrl(`https://${host}/logo.png`, publicResolve);
      expect(r.ok, host).toBe(false);
    }
  });

  it("isIpLiteral and parseIpv6 agree with the cases above", () => {
    expect(isIpLiteral("169.254.169.254")).toBe(true);
    expect(isIpLiteral("[fd00::1]")).toBe(true);
    expect(isIpLiteral("logos.example.com")).toBe(false);
    expect(parseIpv6("::1")).toEqual([0, 0, 0, 0, 0, 0, 0, 1]);
    expect(parseIpv6("fd00::1")).toEqual([0xfd00, 0, 0, 0, 0, 0, 0, 1]);
    expect(parseIpv6("::ffff:127.0.0.1")).toEqual([0, 0, 0, 0, 0, 0xffff, 0x7f00, 1]);
    expect(parseIpv6("64:ff9b::10.0.0.1")).toEqual([0x64, 0xff9b, 0, 0, 0, 0, 0x0a00, 1]);
    expect(parseIpv6("not:an:address")).toBeNull();
  });
});

describe("clause 2b: resolve-then-check rejects non-global answers", () => {
  it("rejects a public-looking name that resolves into the metadata range", async () => {
    const r = await checkUrl("https://logo.attacker.example/logo.png", resolveTo("169.254.169.254"));
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toContain("169.254.0.0/16");
  });

  it("rejects a name that resolves to fd00::/8 unique-local", async () => {
    const r = await checkUrl("https://logo.attacker.example/logo.png", resolveTo("fd00::1"));
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toContain("fd00::/8");
  });

  it("rejects every other non-global range, including the cluster's own services", async () => {
    const cases: Array<[string, RegExp]> = [
      ["10.42.0.9", /10\.0\.0\.0\/8/], // in-cluster pod network
      ["172.20.0.1", /172\.16\.0\.0\/12/],
      ["192.168.49.2", /192\.168\.0\.0\/16/],
      ["127.0.0.1", /loopback/],
      ["0.0.0.0", /this-network/],
      ["100.64.1.1", /carrier-grade NAT/],
      ["169.254.170.2", /169\.254\.0\.0\/16/],
      [MULTICAST_V4, /multicast/],
      ["255.255.255.255", /reserved 240\.0\.0\.0\/4/],
      ["::1", /loopback/],
      ["fe80::1", /link-local fe80::\/10/],
      ["fc00::1", /unique-local/],
      ["ff02::1", /multicast/],
      ["::ffff:10.0.0.1", /10\.0\.0\.0\/8/], // IPv4-mapped
      ["64:ff9b::127.0.0.1", /loopback/], // NAT64
      ["2002:a9fe:a9fe::1", /169\.254\.0\.0\/16/], // 6to4 wrapping metadata
    ];
    for (const [address, why] of cases) {
      const r = await checkUrl("https://logo.attacker.example/logo.png", resolveTo(address));
      expect(r.ok, address).toBe(false);
      expect(r.ok === false && r.reason, address).toMatch(why);
    }
  });

  it("rejects when any one answer is non-global, not just when all are", async () => {
    const r = await checkUrl(
      "https://logo.attacker.example/logo.png",
      resolveTo(GLOBAL_V4, "169.254.169.254"),
    );
    expect(r.ok).toBe(false);
  });

  it("fails closed when the host does not resolve or the resolver is unavailable", async () => {
    expect((await checkUrl("https://logo.example.com/l.png", resolveTo())).ok).toBe(false);
    const throwing = async () => {
      throw new Error("no dns resolver available");
    };
    expect((await checkUrl("https://logo.example.com/l.png", throwing)).ok).toBe(false);
  });

  it("accepts a name that resolves only to global unicast", async () => {
    const r = await checkUrl("https://logos.example.com/logo.png", resolveTo(GLOBAL_V4, "2606:2800:220:1::1"));
    expect(r.ok).toBe(true);
  });

  it("blockedReason returns null for global unicast and a reason otherwise", () => {
    expect(blockedReason(GLOBAL_V4)).toBeNull();
    expect(blockedReason("2606:2800:220:1::1")).toBeNull();
    expect(blockedReason("169.254.169.254")).not.toBeNull();
    expect(blockedReason("fd00::1")).not.toBeNull();
    expect(blockedReason("gibberish")).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Clause 3 — redirects are not followed blindly
// ---------------------------------------------------------------------------
describe("clause 3: redirects are manual and every hop is re-checked", () => {
  it("passes redirect: manual to fetch, so the runtime cannot follow one for us", async () => {
    const fetchFn = vi.fn(async () => streamResponse(PNG, { "content-type": "image/png" }));
    await fetchImageSafely("https://logos.example.com/logo.png", {
      fetchFn: fetchFn as unknown as typeof fetch,
      resolve: publicResolve,
    });
    expect(fetchFn).toHaveBeenCalledTimes(1);
    const init = fetchFn.mock.calls[0][1] as RequestInit;
    expect(init.redirect).toBe("manual");
  });

  it("refuses a redirect to a blocked address and never fetches it", async () => {
    const seen: string[] = [];
    const fetchFn = vi.fn(async (url: string) => {
      seen.push(String(url));
      return redirectTo("https://metadata.attacker.example/latest/meta-data/");
    });
    const resolve = async (hostname: string) =>
      hostname === "logos.example.com" ? [GLOBAL_V4] : ["169.254.169.254"];

    const r = await fetchImageSafely("https://logos.example.com/logo.png", {
      fetchFn: fetchFn as unknown as typeof fetch,
      resolve,
    });

    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/redirect hop 1/);
    expect(r.ok === false && r.reason).toContain("169.254.0.0/16");
    // The blocked host was never contacted.
    expect(seen).toEqual(["https://logos.example.com/logo.png"]);
    expect(seen.join(" ")).not.toContain("metadata.attacker.example");
  });

  it("refuses a redirect to an IP literal and to a downgraded scheme", async () => {
    for (const location of ["https://169.254.169.254/latest/meta-data/", "http://logos.example.com/logo.png"]) {
      const fetchFn = vi.fn(async () => redirectTo(location));
      const r = await fetchImageSafely("https://logos.example.com/logo.png", {
        fetchFn: fetchFn as unknown as typeof fetch,
        resolve: publicResolve,
      });
      expect(r.ok, location).toBe(false);
      expect(fetchFn, location).toHaveBeenCalledTimes(1);
    }
  });

  it("follows a redirect to an allowed host, re-checking it", async () => {
    const fetchFn = vi.fn(async (url: string) =>
      String(url).includes("cdn")
        ? streamResponse(PNG, { "content-type": "image/png" })
        : redirectTo("https://cdn.example.com/logo.png")
    );
    const r = await fetchImageSafely("https://logos.example.com/logo.png", {
      fetchFn: fetchFn as unknown as typeof fetch,
      resolve: publicResolve,
    });
    expect(r.ok).toBe(true);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("gives up on a redirect loop rather than following it forever", async () => {
    const fetchFn = vi.fn(async () => redirectTo("https://logos.example.com/logo.png"));
    const r = await fetchImageSafely("https://logos.example.com/logo.png", {
      fetchFn: fetchFn as unknown as typeof fetch,
      resolve: publicResolve,
      maxRedirects: 2,
    });
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/more than 2 redirects/);
    expect(fetchFn).toHaveBeenCalledTimes(3);
  });

  it("refuses a redirect with no location header", async () => {
    const fetchFn = vi.fn(async () => new Response(null, { status: 302 }));
    const r = await fetchImageSafely("https://logos.example.com/logo.png", {
      fetchFn: fetchFn as unknown as typeof fetch,
      resolve: publicResolve,
    });
    expect(r.ok).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Clause 4 — size cap and image-only content type
// ---------------------------------------------------------------------------
describe("clause 4: response size is capped and the type must be an image", () => {
  const ok = (res: Response) => ({
    fetchFn: (async () => res) as unknown as typeof fetch,
    resolve: publicResolve,
  });

  it("refuses a body larger than the cap even with no content-length", async () => {
    const big = new Uint8Array(1024);
    big.set(PNG.subarray(0, 8), 0);
    const r = await fetchImageSafely(
      "https://logos.example.com/logo.png",
      { ...ok(streamResponse(big, { "content-type": "image/png" })), maxBytes: 512 },
    );
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/exceeds 512 bytes/);
  });

  it("refuses an over-large content-length without downloading the body", async () => {
    // The streaming cap alone would also refuse this, so asserting "refused"
    // would pass with the content-length check deleted. What that check is for
    // is not pulling a declared 100MB through the function first.
    // highWaterMark 0, so pull() fires on an actual read rather than eagerly
    // at construction — otherwise the probe reports a download that never was.
    let pulled = false;
    const body = new ReadableStream<Uint8Array>({
      pull(c) {
        pulled = true;
        // Over the cap on its own, so the streaming cap gives the same
        // refusal reason either way and `pulled` is the only thing that
        // distinguishes "refused on the header" from "refused after reading".
        c.enqueue(new Uint8Array(1024));
        c.close();
      },
    }, new CountQueuingStrategy({ highWaterMark: 0 }));
    const res = new Response(body, {
      headers: { "content-type": "image/png", "content-length": "99999999" },
    });
    const r = await fetchImageSafely("https://logos.example.com/logo.png", { ...ok(res), maxBytes: 512 });
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/exceeds 512 bytes/);
    expect(pulled, "body must not be read when content-length already exceeds the cap").toBe(false);
  });

  it("refuses a non-image content type — this is the metadata-response case", async () => {
    for (
      const contentType of [
        "application/json", // what 169.254.169.254 actually answers
        "text/html",
        "text/plain",
        "application/octet-stream",
        "image/svg+xml",
        "image/gif",
        "",
      ]
    ) {
      const r = await fetchImageSafely(
        "https://logos.example.com/logo.png",
        ok(streamResponse(PNG, contentType ? { "content-type": contentType } : {})),
      );
      expect(r.ok, contentType || "(absent)").toBe(false);
      expect(r.ok === false && r.reason, contentType).toMatch(/not an accepted image type/);
    }
  });

  it("refuses bytes that are not actually PNG or JPEG, however the url or header is spelled", async () => {
    // The pre-fix code chose png vs jpg from the URL suffix and embedded
    // whatever it got, so a JSON body served from a ".png" path reached embedPng.
    const json = new TextEncoder().encode('{"AccessKeyId":"AKIA..."}');
    const r = await fetchImageSafely(
      "https://logos.example.com/logo.png",
      ok(streamResponse(json, { "content-type": "image/png" })),
    );
    expect(r.ok).toBe(false);
    expect(r.ok === false && r.reason).toMatch(/not PNG or JPEG/);
  });

  it("refuses an empty body and a non-2xx status", async () => {
    expect(
      (await fetchImageSafely(
        "https://logos.example.com/logo.png",
        ok(streamResponse(new Uint8Array(0), { "content-type": "image/png" })),
      )).ok,
    ).toBe(false);
    expect(
      (await fetchImageSafely(
        "https://logos.example.com/logo.png",
        ok(new Response(null, { status: 404 })),
      )).ok,
    ).toBe(false);
  });

  it("accepts a real PNG and a real JPEG, and reports the kind from the bytes", async () => {
    const png = await fetchImageSafely(
      "https://logos.example.com/logo.png",
      ok(streamResponse(PNG, { "content-type": "image/png" })),
    );
    expect(png.ok).toBe(true);
    expect(png.ok === true && png.kind).toBe("png");

    // Note the ".png" path with JPEG bytes and a jpeg header: the kind comes
    // from the magic bytes, not the suffix.
    const jpg = await fetchImageSafely(
      "https://logos.example.com/logo.png",
      ok(streamResponse(JPG, { "content-type": "image/jpeg" })),
    );
    expect(jpg.ok).toBe(true);
    expect(jpg.ok === true && jpg.kind).toBe("jpg");
  });
});

// ---------------------------------------------------------------------------
// Clause 5 — a validation failure returns nothing embeddable
// ---------------------------------------------------------------------------
describe("clause 5: no failure path returns bytes", () => {
  const hostile: Array<[string, () => Promise<Response>, ResolveShape]> = [
    ["http scheme", async () => streamResponse(PNG, { "content-type": "image/png" }), publicResolve],
    ["ip literal", async () => streamResponse(PNG, { "content-type": "image/png" }), publicResolve],
    ["resolves to metadata", async () => streamResponse(PNG, { "content-type": "image/png" }), resolveTo("169.254.169.254")],
    ["json body", async () => streamResponse(new TextEncoder().encode("{}"), { "content-type": "application/json" }), publicResolve],
    ["html body with png magic", async () => streamResponse(PNG, { "content-type": "text/html" }), publicResolve],
    ["redirect to metadata", async () => redirectTo("https://169.254.169.254/"), publicResolve],
    ["http error", async () => new Response(null, { status: 500 }), publicResolve],
    ["network failure", () => Promise.reject(new Error("connection refused")), publicResolve],
  ];

  type ResolveShape = (hostname: string) => Promise<string[]>;

  it.each(hostile)("%s yields ok:false and carries no bytes", async (name, responder, resolve) => {
    const url = name === "http scheme"
      ? "http://logos.example.com/logo.png"
      : name === "ip literal"
      ? "https://169.254.169.254/latest/meta-data/"
      : "https://logos.example.com/logo.png";

    const r = await fetchImageSafely(url, {
      fetchFn: responder as unknown as typeof fetch,
      resolve,
    });

    expect(r.ok, name).toBe(false);
    // Not "bytes is empty" — the property must not exist, so there is nothing
    // for a caller to embed by mistake.
    expect(Object.prototype.hasOwnProperty.call(r, "bytes"), name).toBe(false);
    expect(Object.keys(r).sort(), name).toEqual(["ok", "reason"]);
  });

  it("the only shape that carries bytes is one that passed every clause", async () => {
    const r = await fetchImageSafely("https://logos.example.com/logo.png", {
      fetchFn: (async () => streamResponse(PNG, { "content-type": "image/png" })) as unknown as typeof fetch,
      resolve: publicResolve,
    });
    expect(r.ok).toBe(true);
    expect(r.ok === true && r.bytes).toEqual(PNG);
  });
});
