// @vitest-environment happy-dom
// UI-B12: the privacy copy must say only what the product does. Cloud and
// service credentials are AES-GCM encrypted at rest; so are database passwords,
// which also travel to the MCP server sealed (app/mcp_server/credentials.py,
// under DB_ENCRYPTION_KEY); and with a hosted
// model the coder sends the schema plus sample rows to the model provider. The
// old copy claimed data was "never shared" and credentials "encrypted
// server-side" -- these tests keep that claim from coming back.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { SecurityFooter } from "../../src/components/dashboard/SecurityFooter";

(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | null = null;
let host: HTMLElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
});

const source = (rel: string) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf8");
const flat = (s: string) => s.replace(/\s+/g, " ");

// Phrases that over-claim. None may appear in the shipped copy.
const FALSE_CLAIMS = [
  /never shared/i,
  /your data is encrypted/i,
  /credentials are encrypted server-side/i,
  /never leaves your environment/i,
];

describe("SecurityFooter privacy copy", () => {
  function footerText() {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    act(() => root!.render(<SecurityFooter />));
    return flat(host.textContent ?? "").trim();
  }

  it("says what is encrypted and where hosted-model data goes", () => {
    expect(footerText()).toBe(
      "Cloud and service credentials are encrypted at rest. With a hosted model, your schema and " +
        "sample rows are sent to the model provider you configure; a local model keeps them on your " +
        "machine.",
    );
  });

  it("claims nothing beyond its scope: no 'never shared', nothing about passwords", () => {
    const text = footerText();
    for (const claim of FALSE_CLAIMS) expect(text).not.toMatch(claim);
    expect(text).not.toMatch(/password/i);
  });
});

describe("database connect dialog privacy copy", () => {
  const modal = flat(source("../../src/components/database/DatabaseConnectModal.tsx"));

  it("states that the database password is encrypted at rest and sent to MCP sealed", () => {
    expect(modal).toContain(
      "Opens a live connection to your database. Your database password is encrypted at rest and sent to the MCP server sealed.",
    );
  });

  it("does not over-claim (no 'credentials are encrypted server-side', no 'your data is encrypted')", () => {
    for (const claim of FALSE_CLAIMS) expect(modal).not.toMatch(claim);
  });
});

describe("docs privacy copy", () => {
  it("the user guide says how database passwords are protected, and nothing broader", () => {
    const guide = flat(source("../../../docs/TECHNICAL_USER_GUIDE.md"));
    expect(guide).toContain(
      "Database passwords are encrypted at rest under `DB_ENCRYPTION_KEY` too, and sent to the MCP server sealed, never in plaintext",
    );
    expect(guide).not.toMatch(/not encrypted yet/i);
    expect(guide).not.toMatch(/(^|\. )Credentials are encrypted at rest/);
  });

  it("the editions page does not claim work never leaves your environment", () => {
    expect(flat(source("../../../docs/EDITIONS.md"))).not.toMatch(/never leaves your environment/i);
  });
});
