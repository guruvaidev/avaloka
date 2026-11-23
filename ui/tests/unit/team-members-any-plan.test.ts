// Owner ruling: a user on any plan (Free, Professional or Enterprise) may be
// invited to and added as a member of a Professional team. Creating a team stays
// Professional and above (canCreateTeams). The database holds the same rule
// (migration 20261116140000_team_members_any_plan.sql); these pin the server
// functions on both member paths: the picker's eligibility answer, and the email
// invite (send, claim, accept).
//
// The plan resolver is mocked to answer the invitee's plan, so a function that
// still checked it would refuse an Enterprise or Free member here.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const m = vi.hoisted(() => ({
  tier: "enterprise" as string,
  admin: null as any,
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
  planTierForAuthUser: vi.fn(async () => m.tier),
  isProfessionalOrHigher: (t: string) => t === "professional" || t === "enterprise",
}));
vi.mock("@/lib/team-invitation.server", () => ({
  findTeamInvitee: vi.fn(async () => ({ id: "invitee-auth" })),
}));

import * as fns from "../../src/lib/configurations.functions";

type Op = { op: string; args: unknown[] };
type Resolver = (table: string, ops: Op[]) => { data: unknown; error: unknown };

/** A Supabase-shaped fake: every chain records its calls, and the resolver
 *  answers by table and the calls made. */
function fakeDb(resolve: Resolver, rpc: Record<string, unknown> = {}) {
  const writes: Array<{ table: string; ops: Op[] }> = [];
  const from = (table: string) => {
    const ops: Op[] = [];
    const done = () => {
      if (ops.some((o) => o.op === "insert" || o.op === "update")) writes.push({ table, ops });
      return resolve(table, ops);
    };
    const chain: any = new Proxy(
      {},
      {
        get: (_t, prop: string) => {
          if (prop === "then") return (ok: any, bad: any) => Promise.resolve(done()).then(ok, bad);
          if (prop === "single" || prop === "maybeSingle") return async () => done();
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
      rpc: vi.fn(async (name: string) => ({ data: name in rpc ? rpc[name] : null, error: null })),
      auth: {
        admin: {
          generateLink: vi.fn(async () => ({
            data: { properties: { action_link: "https://app.example.test/link" }, user: { id: "invitee-auth" } },
            error: null,
          })),
          getUserById: vi.fn(async () => ({
            data: { user: { email: "ent@example.com", email_confirmed_at: "2026-01-01T00:00:00Z" } },
            error: null,
          })),
        },
      },
    },
  };
}

const has = (ops: Op[], op: string, ...args: unknown[]) =>
  ops.some((o) => o.op === op && args.every((a, i) => o.args[i] === a));

/** The caller's own client: Professional (collab on), signed in with a profile. */
function callerClient() {
  return fakeDb(
    (table, ops) => {
      if (table === "profiles" && has(ops, "eq", "user_id")) return { data: { id: "caller-profile" }, error: null };
      return { data: null, error: null };
    },
    { collab_features_enabled: true, current_org_id: "org-1" },
  ).client;
}

beforeEach(() => {
  vi.stubEnv("SUPABASE_INCLUSTER_URL", "http://mail.example.test");
  vi.stubEnv("INTERNAL_MAIL_TOKEN", "test-token");
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, text: async () => "" })));
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("the picker's eligibility answer: any plan, with an account", () => {
  const admin = () =>
    fakeDb((table) => {
      if (table === "app_users")
        return {
          data: [
            { id: "u-free", auth_user_id: "free-auth" },
            { id: "u-ent", auth_user_id: "ent-auth" },
            { id: "u-none", auth_user_id: null },
          ],
          error: null,
        };
      return { data: null, error: null };
    });

  it.each(["free", "professional", "enterprise"])("a %s member with an account may join", async (tier) => {
    m.tier = tier;
    m.admin = admin().client;
    const out = await (fns as any).getTeamMemberEligibility({
      data: { appUserIds: ["u-free", "u-ent", "u-none"] },
      context: { supabase: callerClient(), userId: "caller-auth" },
    });
    expect(out).toEqual({ "u-free": true, "u-ent": true, "u-none": false });
  });
});

describe("the email invite: any plan", () => {
  function inviteAdmin() {
    return fakeDb((table, ops) => {
      if (table === "teams") return { data: { id: "team-1", owner_profile_id: "caller-profile", name: "Analysts" }, error: null };
      if (table === "profiles" && has(ops, "eq", "user_id", "invitee-auth"))
        return { data: { id: "invitee-profile" }, error: null };
      if (table === "team_invitations" && has(ops, "insert"))
        return {
          data: { id: "inv-1", team_id: "team-1", invited_profile_id: "invitee-profile", status: "pending" },
          error: null,
        };
      return { data: null, error: null };
    });
  }

  it.each(["free", "professional", "enterprise"])("invites a %s user", async (tier) => {
    m.tier = tier;
    const db = inviteAdmin();
    m.admin = db.client;
    // .resolves: a plan refusal fails here as an assertion, not as a thrown error.
    await expect(
      (fns as any).inviteProfessionalTeamMember({
        data: { teamId: "team-1", email: "Invitee@Example.com" },
        context: { supabase: callerClient(), userId: "caller-auth" },
      }),
    ).resolves.toMatchObject({ id: "inv-1", status: "pending", signupRequired: false });
    const insert = db.writes.find((w) => w.table === "team_invitations");
    expect(insert?.ops.find((o) => o.op === "insert")?.args[0]).toMatchObject({
      team_id: "team-1",
      invited_profile_id: "invitee-profile",
      invited_email: "invitee@example.com",
    });
  });

  it.each(["free", "professional", "enterprise"])("a %s invitee's verified email invitation is attached", async (tier) => {
    m.tier = tier;
    const db = fakeDb((table, ops) => {
      if (table === "team_invitations" && has(ops, "update")) return { data: { id: "inv-1" }, error: null };
      if (table === "team_invitations") return { data: [{ id: "inv-1", invited_profile_id: null }], error: null };
      if (table === "profiles" && has(ops, "eq", "user_id")) return { data: { id: "invitee-profile" }, error: null };
      return { data: null, error: null };
    });
    m.admin = db.client;
    const out = await (fns as any).claimPendingProfessionalTeamInvitations({
      context: { supabase: callerClient(), userId: "invitee-auth" },
    });
    expect(out).toEqual({ accepted: 0, attached: 1 });
  });

  it.each(["free", "professional", "enterprise"])("a %s invitee may accept", async (tier) => {
    m.tier = tier;
    const db = fakeDb((table, ops) => {
      if (table === "team_invitations" && has(ops, "update")) return { data: { id: "inv-1", status: "accepted" }, error: null };
      if (table === "team_invitations")
        return { data: { id: "inv-1", team_id: "team-1", invited_profile_id: "caller-profile", status: "pending" }, error: null };
      return { data: null, error: null };
    });
    m.admin = db.client;
    await expect(
      (fns as any).respondToProfessionalTeamInvitation({
        data: { invitationId: "inv-1", response: "accepted" },
        context: { supabase: callerClient(), userId: "invitee-auth" },
      }),
    ).resolves.toEqual({ id: "inv-1", status: "accepted" });
    const update = db.writes.find((w) => w.table === "team_invitations");
    expect(update?.ops.find((o) => o.op === "update")?.args[0]).toEqual({ status: "accepted" });
  });
});
