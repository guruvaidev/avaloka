// NEW-BG-01, server side (owner rulings 2026-11-01, revised): the org-structure
// server functions refuse unless the database says the caller's plan allows it
// -- Professional or higher for teams (with Professional-or-higher members),
// Enterprise for departments, roles, branches and app users -- and they record
// the action name for the audited app-admin bypass. Handlers are called directly
// with a fake Supabase client; the checks run before any write.
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-start", () => {
  const builder = () => {
    const b: any = {
      middleware: () => b,
      inputValidator: () => b,
      handler: (fn: any) => fn,
    };
    return b;
  };
  return { createServerFn: builder, createMiddleware: builder };
});
vi.mock("@/integrations/supabase/primary-auth-middleware", () => ({ requireSupabaseAuth: {} }));

import * as fns from "../../src/lib/configurations.functions";
import {
  COLLAB_FEATURES_REQUIRED,
  ORG_FEATURES_AUDIT_FAILED,
  ORG_FEATURES_REQUIRED,
  TEAM_MEMBERS_NEED_PRO,
  assertPlanFeature,
} from "../../src/lib/org-features";

type Plan = { collab: unknown; org: unknown; proMembers?: string[]; rpcError?: unknown; logError?: unknown };

/** A fake client answering the plan RPCs; every table access is recorded so a
 *  test can prove nothing was written. */
function fakeClient(plan: Plan) {
  const tables: string[] = [];
  const chain: any = new Proxy(
    {},
    {
      get: (_t, prop) => {
        if (prop === "then") return undefined;
        if (prop === "data") return [];
        if (prop === "error") return null;
        if (prop === "single" || prop === "maybeSingle")
          return async () => ({ data: { id: "t1", organization_id: "o1" }, error: null });
        return () => chain;
      },
    },
  );
  const client = {
    rpc: vi.fn(async (name: string, args?: any) => {
      switch (name) {
        case "collab_features_enabled":
          return { data: plan.collab, error: plan.rpcError ?? null };
        case "org_features_enabled":
          return { data: plan.org, error: plan.rpcError ?? null };
        case "team_member_allowed":
          return { data: plan.org === true || (plan.proMembers ?? []).includes(args?.p_app_user_id), error: null };
        case "log_org_feature_bypass":
          return { data: null, error: plan.logError ?? null };
        default:
          return { data: null, error: { message: `unexpected rpc ${name}` } };
      }
    }),
    from: vi.fn((table: string) => {
      tables.push(table);
      return chain;
    }),
  };
  return { client, tables };
}

const FREE: Plan = { collab: false, org: false };
const PRO: Plan = { collab: true, org: false, proMembers: ["pro-member"] };
const ENT: Plan = { collab: true, org: true };
const team = { name: "Analysts", branch: "HQ" };

const call = (name: string, plan: Plan, data: any = { rows: [], email: "x@example.com", id: "i1" }) => {
  const { client, tables } = fakeClient(plan);
  return { client, tables, run: () => (fns as any)[name]({ data, context: { supabase: client, userId: "u" } }) };
};

describe("assertPlanFeature", () => {
  it("passes only when the database answers exactly true, per level", async () => {
    await expect(assertPlanFeature(fakeClient(PRO).client, "professional", "upsertTeam")).resolves.toBeUndefined();
    await expect(assertPlanFeature(fakeClient(PRO).client, "enterprise", "upsertBranch")).rejects.toThrow(
      ORG_FEATURES_REQUIRED,
    );
    for (const answer of [false, null, undefined, "true", 1]) {
      await expect(
        assertPlanFeature(fakeClient({ collab: answer, org: answer }).client, "professional", "upsertTeam"),
      ).rejects.toThrow(COLLAB_FEATURES_REQUIRED);
    }
  });

  it("fails closed when the check itself errors", async () => {
    await expect(
      assertPlanFeature(fakeClient({ ...ENT, rpcError: { message: "boom" } }).client, "enterprise", "x"),
    ).rejects.toThrow(ORG_FEATURES_REQUIRED);
    const throwing = { rpc: () => Promise.reject(new Error("network")) };
    await expect(assertPlanFeature(throwing, "professional", "x")).rejects.toThrow(COLLAB_FEATURES_REQUIRED);
  });
});

describe("upsertTeam: Professional or higher, members Professional or higher", () => {
  it("refuses a Free caller with a clear error and writes nothing", async () => {
    const c = call("upsertTeam", FREE, team);
    await expect(c.run()).rejects.toThrow(COLLAB_FEATURES_REQUIRED);
    expect(c.client.rpc).toHaveBeenCalledWith("collab_features_enabled");
    expect(c.tables).toEqual([]);
  });

  it("lets a Professional caller form a team with Professional members", async () => {
    const c = call("upsertTeam", PRO, { ...team, member_ids: ["pro-member"] });
    await c.run();
    expect(c.tables).toContain("organization_teams");
  });

  it("refuses a Free member on a Professional team with the upgrade message, before any write", async () => {
    const c = call("upsertTeam", PRO, { ...team, member_ids: ["pro-member", "free-member"] });
    await expect(c.run()).rejects.toThrow(TEAM_MEMBERS_NEED_PRO);
    expect(c.tables).toEqual([]);
  });

  it("lets an Enterprise caller add any org member", async () => {
    const c = call("upsertTeam", ENT, { ...team, member_ids: ["free-member", "anyone"] });
    await c.run();
    expect(c.tables).toContain("organization_teams");
  });
});

describe("departments, roles, branches and app users are Enterprise-only", () => {
  const enterpriseOnly = [
    "upsertDepartment",
    "bulkUpsertDepartments",
    "upsertConfigRole",
    "bulkUpsertConfigRoles",
    "upsertBranch",
    "bulkUpsertBranches",
    "upsertAppUser",
    "inviteAppUser",
    "resendAppUserInvite",
  ] as const;

  it.each(enterpriseOnly)("%s refuses Free and Professional before touching a table", async (name) => {
    for (const plan of [FREE, PRO]) {
      const c = call(name, plan);
      await expect(c.run()).rejects.toThrow(ORG_FEATURES_REQUIRED);
      expect(c.tables).toEqual([]);
    }
  });
});

describe("app-admin bypass audit (server side)", () => {
  it("records the action name and level only, after the check passes", async () => {
    const { client } = fakeClient(ENT);
    await assertPlanFeature(client, "enterprise", "inviteAppUser");
    expect(client.rpc).toHaveBeenNthCalledWith(1, "org_features_enabled");
    expect(client.rpc).toHaveBeenNthCalledWith(2, "log_org_feature_bypass", {
      p_action: "inviteAppUser",
      p_required: "enterprise",
    });
  });

  it("does not record anything when the plan check refuses", async () => {
    const { client } = fakeClient(FREE);
    await expect(assertPlanFeature(client, "professional", "upsertTeam")).rejects.toThrow();
    expect(client.rpc).toHaveBeenCalledTimes(1);
  });

  it("refuses the action when the audit record cannot be written", async () => {
    const c = call("upsertTeam", { ...PRO, logError: { message: "audit down" } }, team);
    await expect(c.run()).rejects.toThrow(ORG_FEATURES_AUDIT_FAILED);
    expect(c.tables).toEqual([]);
  });

  it("upsertTeam records itself at the Professional level", async () => {
    const c = call("upsertTeam", PRO, team);
    await c.run();
    expect(c.client.rpc).toHaveBeenCalledWith("log_org_feature_bypass", {
      p_action: "upsertTeam",
      p_required: "professional",
    });
  });
});
