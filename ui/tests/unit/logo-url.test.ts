// @vitest-environment node
//
// The source side of the logo SSRF. updateOrganizationProfile's validator used
// to do `String(logoRaw).trim().slice(0, 2048) || null` and no URL parse, so
// anything at all could be written to organizations.logo_url — which the
// send-report-email edge function then fetched server-side.
import { describe, expect, it } from "vitest";

import { LOGO_URL_MAX_LENGTH, normalizeLogoUrl } from "@/lib/logo-url";

describe("normalizeLogoUrl", () => {
  it("accepts an https url and returns it normalized", () => {
    expect(normalizeLogoUrl("https://cdn.example.com/logo.png")).toBe("https://cdn.example.com/logo.png");
    expect(normalizeLogoUrl("  https://cdn.example.com/logo.png  ")).toBe("https://cdn.example.com/logo.png");
  });

  it("accepts the Supabase storage publicUrl shape the settings page produces", () => {
    const url = "https://abcdefgh.supabase.co/storage/v1/object/public/org-logos/123/logo.png";
    expect(normalizeLogoUrl(url)).toBe(url);
  });

  it("rejects a non-https scheme", () => {
    for (
      const raw of [
        "http://cdn.example.com/logo.png",
        "file:///etc/passwd",
        "data:image/png;base64,iVBORw0KGgo=",
        "javascript:alert(1)",
        "ftp://cdn.example.com/logo.png",
        "gopher://169.254.169.254/",
      ]
    ) {
      expect(() => normalizeLogoUrl(raw), raw).toThrow();
    }
  });

  it("rejects something that is not a URL at all", () => {
    for (const raw of ["cdn.example.com/logo.png", "/logo.png", "not a url", "<script>", "../../etc/passwd"]) {
      expect(() => normalizeLogoUrl(raw), raw).toThrow(/full URL/);
    }
  });

  it("rejects a url that embeds credentials", () => {
    expect(() => normalizeLogoUrl("https://user:pass@cdn.example.com/logo.png")).toThrow(/username or password/);
  });

  it("rejects a value longer than the column limit instead of silently truncating it", () => {
    // The old code sliced to 2048 and kept going, which could turn a long value
    // into a different, valid-looking URL.
    const long = `https://cdn.example.com/${"a".repeat(LOGO_URL_MAX_LENGTH)}.png`;
    expect(() => normalizeLogoUrl(long)).toThrow(/characters or fewer/);
  });

  it("treats an absent field as absent and an empty or null value as clearing the logo", () => {
    expect(normalizeLogoUrl(undefined)).toBeUndefined();
    expect(normalizeLogoUrl(null)).toBeNull();
    expect(normalizeLogoUrl("")).toBeNull();
    expect(normalizeLogoUrl("   ")).toBeNull();
  });

  it("does not accept a non-string by coercing it into something URL-shaped", () => {
    expect(() => normalizeLogoUrl(12345)).toThrow();
    expect(() => normalizeLogoUrl({ toString: () => "http://169.254.169.254/" })).toThrow();
  });
});
