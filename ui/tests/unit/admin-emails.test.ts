import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

// Loaded through import.meta.glob so that a tree without the module (the
// hard-coded-admin code this replaced) fails on an assertion, not on import.
const modules = import.meta.glob<typeof import("../../src/lib/admin-emails")>(
  "../../src/lib/admin-emails.ts",
);
let isAdminEmail: (typeof import("../../src/lib/admin-emails"))["isAdminEmail"];
let parseAdminEmails: (typeof import("../../src/lib/admin-emails"))["parseAdminEmails"];

beforeAll(async () => {
  const loaders = Object.values(modules);
  expect(loaders, "ui/src/lib/admin-emails.ts is missing").toHaveLength(1);
  ({ isAdminEmail, parseAdminEmails } = await loaders[0]());
});

// Administrators come from ADMIN_EMAILS (server env), never from source. Empty
// means nobody: no fallback address, no "first user is admin".
describe("admin emails from configuration", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("treats an unset or empty list as no admin at all", () => {
    for (const raw of [undefined, null, "", "   ", " , ,"]) {
      expect(parseAdminEmails(raw).size).toBe(0);
      expect(isAdminEmail("ops@example.com", raw)).toBe(false);
    }
  });

  it("grants admin to a listed address only", () => {
    const raw = "ops@example.com,lead@example.com";
    expect(isAdminEmail("ops@example.com", raw)).toBe(true);
    expect(isAdminEmail("lead@example.com", raw)).toBe(true);
    expect(isAdminEmail("someone@example.com", raw)).toBe(false);
    expect(isAdminEmail("ops@example.com.evil.test", raw)).toBe(false);
  });

  it("ignores case and surrounding whitespace on both sides", () => {
    const raw = "  Ops@Example.COM , lead@example.com ,";
    expect([...parseAdminEmails(raw)]).toEqual(["ops@example.com", "lead@example.com"]);
    expect(isAdminEmail("OPS@example.com", raw)).toBe(true);
    expect(isAdminEmail("  lead@EXAMPLE.com  ", raw)).toBe(true);
  });

  it("never treats a missing caller email as admin", () => {
    const raw = "ops@example.com";
    for (const email of [undefined, null, "", "   "]) {
      expect(isAdminEmail(email, raw)).toBe(false);
    }
  });

  it("reads ADMIN_EMAILS from the environment by default", () => {
    vi.stubEnv("ADMIN_EMAILS", "Desk@Example.com");
    expect(isAdminEmail("desk@example.com")).toBe(true);
    vi.stubEnv("ADMIN_EMAILS", "");
    expect(isAdminEmail("desk@example.com")).toBe(false);
  });
});
