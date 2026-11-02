// NEW-BG-01, server side: collaboration and organisation features by plan.
//
// Owner rulings (2026-11-01, revised): Professional or higher forms teams (whose
// members must be Professional or higher) and invites collaborators; Enterprise
// adds departments, roles, branches and app users. The source of truth is the
// database (migration 20261101140000_plan_rules_pro_collab.sql, which supersedes 20261101120000_plan_enforced_org_features.sql):
// public.collab_features_enabled(), public.org_features_enabled() and
// public.team_member_allowed(), the same checks the RLS policies and triggers
// apply. Server functions call them through the caller's own client, so they
// resolve the caller's plan, and they fail closed: an error or anything but
// `true` refuses.
//
// The app-admin bypass is allowed but audited: after the check passes, the
// action name is recorded through public.log_org_feature_bypass(), which writes a
// row only when the caller is actually using the bypass for that level, and is a
// no-op otherwise. Only the action name is sent, never customer data. If the
// record cannot be written, the action is refused.

export type PlanLevel = "professional" | "enterprise";

export const COLLAB_FEATURES_REQUIRED =
  "Teams and collaborator invites need a Professional plan or higher.";
export const ORG_FEATURES_REQUIRED =
  "Departments, roles, branches and users need an Enterprise plan.";
export const TEAM_MEMBERS_NEED_PRO = "Team members need a Professional plan or higher.";
export const ORG_FEATURES_AUDIT_FAILED =
  "This action could not be recorded in the audit log, so it was not carried out.";

async function rpcIsTrue(supabase: any, fn: string, args?: Record<string, unknown>): Promise<boolean> {
  try {
    const { data, error } = args ? await supabase.rpc(fn, args) : await supabase.rpc(fn);
    return !error && data === true;
  } catch {
    return false;
  }
}

export async function assertPlanFeature(supabase: any, level: PlanLevel, action: string): Promise<void> {
  const fn = level === "enterprise" ? "org_features_enabled" : "collab_features_enabled";
  if (!(await rpcIsTrue(supabase, fn))) {
    throw new Error(level === "enterprise" ? ORG_FEATURES_REQUIRED : COLLAB_FEATURES_REQUIRED);
  }
  let logged = false;
  try {
    const { error } = await supabase.rpc("log_org_feature_bypass", { p_action: action, p_required: level });
    logged = !error;
  } catch {
    logged = false;
  }
  if (!logged) throw new Error(ORG_FEATURES_AUDIT_FAILED);
}

/** Every proposed team member must be allowed (Professional or higher on a
 *  Professional team; any org member on Enterprise). */
export async function assertTeamMembersAllowed(supabase: any, appUserIds: string[]): Promise<void> {
  for (const id of appUserIds) {
    if (!(await rpcIsTrue(supabase, "team_member_allowed", { p_app_user_id: id }))) {
      throw new Error(TEAM_MEMBERS_NEED_PRO);
    }
  }
}
