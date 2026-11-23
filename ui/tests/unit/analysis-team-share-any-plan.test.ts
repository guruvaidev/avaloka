// Owner ruling: sharing an analysis with a team member follows the any-plan
// rule. Any member of the sharer's team can receive its shared analyses,
// whatever their own plan; someone outside the sharer's teams still cannot.
// These pin the server functions: the recipient list (getAnalysisShareSettings)
// and a direct share (createAnalysisShare).
//
// The plan resolver is mocked to answer each person's plan, so a function that
// still checked the recipient's plan would leave out or refuse that person here.
import { describe, expect, it, vi } from "vitest";

const m = vi.hoisted(() => ({
  admin: null as any,
  tiers: { "ent-auth": "enterprise", "free-auth": "free", "pro-auth": "professional", "out-auth": "enterprise" } as Record<string, string>,
}));

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
vi.mock("@/integrations/supabase/client.server", () => ({
  get supabaseAdmin() {
    return m.admin;
  },
}));
vi.mock("@/lib/plan-tier.server", () => ({
  planTierForAuthUser: vi.fn(async (_admin: unknown, authId: string) => m.tiers[authId] ?? "free"),
}));

import * as fns from "../../src/lib/analysis-sharing.functions";

type Op = { op: string; args: unknown[] };

/** People: the sharer (owner of team-1 and of the analysis), three teammates on
 *  each plan, and an Enterprise user who is on none of the sharer's teams. */
const PEOPLE: Record<string, { auth: string; name: string }> = {
  "owner-profile": { auth: "owner-auth", name: "Owner" },
  "ent-profile": { auth: "ent-auth", name: "Enterprise teammate" },
  "free-profile": { auth: "free-auth", name: "Free teammate" },
  "pro-profile": { auth: "pro-auth", name: "Professional teammate" },
  "out-profile": { auth: "out-auth", name: "Outsider" },
};
const TEAM_MEMBERS = ["ent-profile", "free-profile", "pro-profile"];

const arg = (ops: Op[], op: string, col: string) => ops.find((o) => o.op === op && o.args[0] === col)?.args[1];
const has = (ops: Op[], op: string) => ops.some((o) => o.op === op);

function fakeAdmin() {
  const writes: Array<{ table: string; ops: Op[] }> = [];
  const resolve = (table: string, ops: Op[]): { data: unknown; error: unknown } => {
    if (table === "profiles") {
      const byUser = arg(ops, "eq", "user_id");
      if (byUser) {
        const id = Object.keys(PEOPLE).find((k) => PEOPLE[k].auth === byUser);
        return { data: id ? { id, user_id: byUser, full_name: PEOPLE[id].name, avatar_url: null } : null, error: null };
      }
      const id = arg(ops, "eq", "id") as string | undefined;
      const p = id ? PEOPLE[id] : undefined;
      return { data: p ? { id, user_id: p.auth, full_name: p.name, avatar_url: null } : null, error: null };
    }
    if (table === "analyses")
      return { data: { id: "a1", owner_id: "owner-profile", project_id: null, name: "Churn", parent_analysis_id: null }, error: null };
    if (table === "teams")
      return { data: [{ id: "team-1", name: "Analysts", owner_profile_id: "owner-profile", lead_profile_id: "owner-profile" }], error: null };
    if (table === "team_memberships") {
      const user = arg(ops, "eq", "user_id") as string | undefined;
      if (user && has(ops, "in")) return { data: TEAM_MEMBERS.includes(user) ? [{ team_id: "team-1" }] : [], error: null };
      if (user) return { data: [], error: null }; // the sharer's own memberships (they own the team)
      return { data: TEAM_MEMBERS.map((user_id) => ({ team_id: "team-1", user_id })), error: null };
    }
    if (table === "analysis_shares" && has(ops, "insert")) return { data: { id: "share-1" }, error: null };
    if (table === "analysis_shares" && has(ops, "maybeSingle")) return { data: null, error: null };
    if (table === "analysis_shares") return { data: [], error: null };
    return { data: null, error: null };
  };
  const from = (table: string) => {
    const ops: Op[] = [];
    const done = (terminal?: string) => {
      if (terminal) ops.push({ op: terminal, args: [] });
      if (has(ops, "insert") || has(ops, "update")) writes.push({ table, ops });
      return resolve(table, ops);
    };
    const chain: any = new Proxy(
      {},
      {
        get: (_t, prop: string) => {
          if (prop === "then") return (ok: any, bad: any) => Promise.resolve(done()).then(ok, bad);
          if (prop === "single" || prop === "maybeSingle") return async () => done(prop);
          return (...args: unknown[]) => {
            ops.push({ op: prop, args });
            return chain;
          };
        },
      },
    );
    return chain;
  };
  return {
    writes,
    client: {
      from: vi.fn(from),
      auth: {
        admin: {
          getUserById: vi.fn(async (id: string) => ({ data: { user: { email: `${id}@example.com` } }, error: null })),
        },
      },
    },
  };
}

/** The sharer's own client: Professional, so the plan feature check passes. */
const callerClient = () => ({
  rpc: vi.fn(async (name: string) => ({ data: name === "collab_features_enabled" ? true : null, error: null })),
  from: vi.fn(),
});
const ctx = () => ({ supabase: callerClient(), userId: "owner-auth" });

describe("the recipient list: every teammate, whatever their plan", () => {
  it("lists the Enterprise, Free and Professional teammates", async () => {
    m.admin = fakeAdmin().client;
    const out = await (fns as any).getAnalysisShareSettings({ data: { analysisId: "a1" }, context: ctx() });
    expect((out.members as Array<{ id: string }>).map((p) => p.id).sort()).toEqual(["ent-profile", "free-profile", "pro-profile"]);
  });
});

describe("a direct share: a teammate on any plan, never an outsider", () => {
  it.each(["ent-profile", "free-profile", "pro-profile"])("shares with teammate %s", async (recipientProfileId) => {
    const db = fakeAdmin();
    m.admin = db.client;
    // .resolves: a plan refusal fails here as an assertion, not as a thrown error.
    await expect(
      (fns as any).createAnalysisShare({ data: { analysisId: "a1", recipientProfileId, accessLevel: "view" }, context: ctx() }),
    ).resolves.toEqual({ ok: true, id: "share-1" });
    const insert = db.writes.find((w) => w.table === "analysis_shares");
    expect(insert?.ops.find((o) => o.op === "insert")?.args[0]).toMatchObject({
      analysis_id: "a1",
      recipient_profile_id: recipientProfileId,
      team_id: null,
      access_level: "view",
    });
  });

  it("refuses someone who is on none of the sharer's teams, and writes no share", async () => {
    const db = fakeAdmin();
    m.admin = db.client;
    await expect(
      (fns as any).createAnalysisShare({
        data: { analysisId: "a1", recipientProfileId: "out-profile", accessLevel: "view" },
        context: ctx(),
      }),
    ).rejects.toThrow("Share this analysis with a member of one of your teams.");
    expect(db.writes.filter((w) => w.table === "analysis_shares")).toEqual([]);
  });
});
