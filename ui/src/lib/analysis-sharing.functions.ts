import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/primary-auth-middleware";
import { assertPlanFeature } from "@/lib/org-features";

type ShareResource = "analysis" | "project";
const shareTable = (type: ShareResource = "analysis") => type === "project" ? "project_shares" : "analysis_shares";
const shareKey = (type: ShareResource = "analysis") => type === "project" ? "project_id" : "analysis_id";

type ShareAccess = "view" | "comment" | "edit";

async function profileForAuthUser(admin: any, authUserId: string): Promise<any> {
  const byUser = await admin
    .from("profiles")
    .select("id,user_id,full_name,avatar_url")
    .eq("user_id", authUserId)
    .maybeSingle();
  if (byUser.error) throw new Error(byUser.error.message);
  if (byUser.data) return byUser.data;
  const byId = await admin.from("profiles").select("id,user_id,full_name,avatar_url").eq("id", authUserId).maybeSingle();
  if (byId.error) throw new Error(byId.error.message);
  if (!byId.data) throw new Error("Could not find your profile.");
  return byId.data;
}

async function loadShareResource(admin: any, analysisId: string, resourceType: ShareResource = "analysis"): Promise<any> {
  const { data, error } = await admin
    .from(resourceType === "project" ? "projects" : "analyses")
    .select(resourceType === "project" ? "id,owner_id,name" : "id,owner_id,project_id,name,parent_analysis_id")
    .eq("id", analysisId)
    .maybeSingle();
  if (error) throw new Error(error.message);
  if (!data) throw new Error("Analysis not found.");
  if (data.project_id) throw new Error("Team sharing is available for standalone analyses only.");
  return { ...data, resourceType };
}

async function getActorTeams(admin: any, profileId: string): Promise<any[]> {
  const [{ data: allTeams, error: teamsError }, { data: memberships, error: membershipsError }] = await Promise.all([
    admin.from("teams").select("id,name,owner_profile_id,lead_profile_id"),
    admin.from("team_memberships").select("team_id").eq("user_id", profileId),
  ]);
  if (teamsError) throw new Error(teamsError.message);
  if (membershipsError) throw new Error(membershipsError.message);
  const memberTeamIds = new Set((memberships ?? []).map((row: any) => row.team_id));
  return (allTeams ?? []).filter((team: any) =>
    memberTeamIds.has(team.id) || team.owner_profile_id === profileId || team.lead_profile_id === profileId,
  );
}

async function accessLevelForActor(admin: any, analysis: any, profileId: string, teamIds: string[]): Promise<number> {
  if (analysis.owner_id === profileId) return 3;
  const rootId = analysis.parent_analysis_id ?? analysis.id;
  const { data: shares, error } = await admin
    .from(shareTable(analysis.resourceType))
    .select("access_level,can_share,recipient_profile_id,team_id")
    .in(shareKey(analysis.resourceType), [analysis.id, rootId]);
  if (error) throw new Error(error.message);
  const matching = (shares ?? []).filter((share: any) =>
    share.recipient_profile_id === profileId || (share.team_id && teamIds.includes(share.team_id)),
  );
  const delegatable = matching.filter((share: any) => share.can_share);
  if (!delegatable.length) return 0;
  return Math.max(0, ...delegatable.map((share: any) => ({ view: 1, comment: 2, edit: 3 }[share.access_level as ShareAccess] ?? 0)));
}

async function assertCanManage(admin: any, analysis: any, profileId: string, teamIds: string[]): Promise<boolean> {
  if (analysis.owner_id === profileId) return true;
  return (await accessLevelForActor(admin, analysis, profileId, teamIds)) > 0;
}

async function profileSummary(admin: any, profileId: string): Promise<any | null> {
  const { data: profile, error } = await admin
    .from("profiles")
    .select("id,user_id,full_name,avatar_url")
    .eq("id", profileId)
    .maybeSingle();
  if (error) throw new Error(error.message);
  if (!profile) return null;
  let email: string | null = null;
  if (profile.user_id) {
    const { data: authUser } = await admin.auth.admin.getUserById(profile.user_id);
    email = authUser?.user?.email ?? null;
  }
  return {
    id: profile.id,
    name: profile.full_name?.trim() || email?.split("@")[0] || "Avaloka user",
    email,
    avatar: profile.avatar_url ?? null,
  };
}

async function notifyShareRecipients(admin: any, recipientProfileIds: string[], actorProfileId: string, analysis: any, teamName?: string) {
  const recipients = Array.from(new Set(recipientProfileIds.filter((id) => id && id !== actorProfileId)));
  if (!recipients.length) return;
  const { data: actor } = await admin.from("profiles").select("full_name").eq("id", actorProfileId).maybeSingle();
  const actorName = actor?.full_name?.trim() || "A teammate";
  const target = teamName ? `the ${teamName} team` : "you";
  const rows = recipients.map((recipient_id) => ({
    recipient_id,
    type: analysis.resourceType === "project" ? "generic" : "analysis_shared",
    title: analysis.resourceType === "project" ? "Project shared with you" : "Analysis shared with you",
    body: `${actorName} shared “${analysis.name}” with ${target}.`,
    actor_id: actorProfileId,
    resource_type: analysis.resourceType,
    resource_id: analysis.id,
    link: analysis.resourceType === "project" ? "/dashboard" : `/analysis?aid=${analysis.id}`,
    metadata: { analysis_id: analysis.id, team_name: teamName ?? null },
  }));
  const { error } = await admin.from("notifications").insert(rows);
  if (error) console.warn("Failed to notify analysis share recipients", error.message);
}

export const getAnalysisShareSettings = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { analysisId: string; resourceType?: ShareResource }) => input)
  .handler(async ({ data, context }) => {
    await assertPlanFeature(context.supabase, "professional", "getAnalysisShareSettings");
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const admin = supabaseAdmin as any;
    const actor = await profileForAuthUser(admin, context.userId);
    const analysis = await loadShareResource(admin, data.analysisId, data.resourceType);
    const actorTeams = await getActorTeams(admin, actor.id);
    const teamIds = actorTeams.map((team: any) => team.id);
    const canManage = await assertCanManage(admin, analysis, actor.id, teamIds);
    if (!canManage) throw new Error("You do not have permission to manage sharing for this analysis.");

    const memberRows = teamIds.length
      ? await admin.from("team_memberships").select("team_id,user_id").in("team_id", teamIds)
      : { data: [], error: null };
    if (memberRows.error) throw new Error(memberRows.error.message);
    const candidateIds = Array.from(new Set([
      ...(memberRows.data ?? []).map((row: any) => row.user_id),
      ...actorTeams.flatMap((team: any) => [team.owner_profile_id, team.lead_profile_id].filter(Boolean)),
    ])).filter((id) => id !== actor.id);
    // Any member of the actor's teams may receive a share, whatever their own
    // plan (owner ruling); they need an Avaloka account.
    const candidates = (await Promise.all(candidateIds.map(async (id) => {
      const profile = await profileSummary(admin, id);
      if (!profile) return null;
      const { data: row } = await admin.from("profiles").select("user_id").eq("id", id).maybeSingle();
      return row?.user_id ? profile : null;
    }))).filter(Boolean);

    const rootId = analysis.parent_analysis_id ?? analysis.id;
    const { data: grants, error: grantsError } = await admin
      .from(shareTable(data.resourceType))
      .select("*")
      .in(shareKey(data.resourceType), [analysis.id, rootId])
      .order("created_at", { ascending: true });
    if (grantsError) throw new Error(grantsError.message);
    const grantProfileIds = Array.from(new Set((grants ?? []).map((row: any) => row.recipient_profile_id).filter(Boolean)));
    const { data: grantProfiles, error: grantProfilesError } = grantProfileIds.length
      ? await admin.from("profiles").select("id,full_name,user_id").in("id", grantProfileIds)
      : { data: [], error: null };
    if (grantProfilesError) throw new Error(grantProfilesError.message);
    const grantProfileNames = new Map<string, string>();
    for (const profile of grantProfiles ?? []) {
      let email = "";
      if (profile.user_id) {
        const { data: authUser } = await admin.auth.admin.getUserById(profile.user_id);
        email = authUser?.user?.email ?? "";
      }
      grantProfileNames.set(profile.id, profile.full_name?.trim() || email || "Avaloka user");
    }
    const grantTeamIds = Array.from(new Set((grants ?? []).map((row: any) => row.team_id).filter(Boolean)));
    const { data: grantTeams, error: grantTeamsError } = grantTeamIds.length
      ? await admin.from("teams").select("id,name").in("id", grantTeamIds)
      : { data: [], error: null };
    if (grantTeamsError) throw new Error(grantTeamsError.message);
    const grantTeamNames = new Map((grantTeams ?? []).map((team: any) => [team.id, team.name]));
    const shareRows = (grants ?? []).map((row: any) => ({
      id: row.id,
      targetType: row.team_id ? "team" : "profile",
      targetId: row.team_id ?? row.recipient_profile_id,
      targetName: row.team_id ? grantTeamNames.get(row.team_id) ?? "Team" : grantProfileNames.get(row.recipient_profile_id) ?? "Avaloka user",
      accessLevel: row.access_level,
      canShare: row.can_share,
      sharedBy: row.shared_by,
    }));
    return {
      analysis: { id: analysis.id, name: analysis.name },
      viewerProfileId: actor.id,
      isOwner: analysis.owner_id === actor.id,
      teams: actorTeams.map((team: any) => ({ id: team.id, name: team.name })),
      members: candidates,
      shares: shareRows,
    };
  });

export const createAnalysisShare = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    analysisId: string;
    resourceType?: ShareResource;
    recipientProfileId?: string;
    teamId?: string;
    accessLevel: ShareAccess;
    canShare?: boolean;
  }) => input)
  .handler(async ({ data, context }) => {
    await assertPlanFeature(context.supabase, "professional", "createStandaloneAnalysisShare");
    if (!(["view", "comment", "edit"] as string[]).includes(data.accessLevel)) throw new Error("Invalid access level.");
    const hasProfile = Boolean(data.recipientProfileId);
    const hasTeam = Boolean(data.teamId);
    if (hasProfile === hasTeam) throw new Error("Choose either one user or one team.");

    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const admin = supabaseAdmin as any;
    const actor = await profileForAuthUser(admin, context.userId);
    const analysis = await loadShareResource(admin, data.analysisId, data.resourceType);
    const actorTeams = await getActorTeams(admin, actor.id);
    const teamIds = actorTeams.map((team: any) => team.id);
    const isOwner = analysis.owner_id === actor.id;
    const actorLevel = isOwner ? 3 : await accessLevelForActor(admin, analysis, actor.id, teamIds);
    if (!isOwner && actorLevel === 0) throw new Error("You do not have permission to share this analysis.");
    const accessRank = { view: 1, comment: 2, edit: 3 }[data.accessLevel];
    if (accessRank > actorLevel) throw new Error("You cannot grant more access than you have.");
    if (data.canShare && !isOwner) throw new Error("Only the analysis owner can let another user reshare it.");

    let targetProfileId: string | null = null;
    let targetTeamId: string | null = null;
    let teamName: string | undefined;
    let recipients: string[] = [];
    if (data.recipientProfileId) {
      targetProfileId = data.recipientProfileId;
      if (targetProfileId === actor.id) throw new Error("You already own or have access to this analysis.");
      const { data: targetMemberships, error: targetMembershipsError } = await admin
        .from("team_memberships")
        .select("team_id")
        .eq("user_id", targetProfileId)
        .in("team_id", teamIds.length ? teamIds : ["00000000-0000-0000-0000-000000000000"]);
      if (targetMembershipsError) throw new Error(targetMembershipsError.message);
      const targetTeamSet = new Set((targetMemberships ?? []).map((row: any) => row.team_id));
      const targetIsTeamOwnerOrLead = actorTeams.some((team: any) =>
        team.id && (team.owner_profile_id === targetProfileId || team.lead_profile_id === targetProfileId),
      );
      if (!targetTeamSet.size && !targetIsTeamOwnerOrLead) throw new Error("Share this analysis with a member of one of your teams.");
      const target = await profileSummary(admin, targetProfileId);
      if (!target) throw new Error("User not found.");
      // A teammate on any plan may receive it (owner ruling).
      recipients = [targetProfileId];
    } else if (data.teamId) {
      targetTeamId = data.teamId;
      const team = actorTeams.find((candidate: any) => candidate.id === targetTeamId);
      if (!team) throw new Error("Share this analysis with one of your teams.");
      teamName = team.name;
      const { data: memberships, error: membershipsError } = await admin
        .from("team_memberships")
        .select("user_id")
        .eq("team_id", targetTeamId);
      if (membershipsError) throw new Error(membershipsError.message);
      recipients = [
        ...(memberships ?? []).map((row: any) => row.user_id),
        team.owner_profile_id,
        team.lead_profile_id,
      ].filter(Boolean);
    }

    const rootId = analysis.parent_analysis_id ?? analysis.id;
    const query = await admin
      .from(shareTable(data.resourceType))
      .select("id,shared_by")
      .eq(shareKey(data.resourceType), rootId)
      .match(targetTeamId ? { team_id: targetTeamId } : { recipient_profile_id: targetProfileId })
      .maybeSingle();
    if (query.error) throw new Error(query.error.message);
    if (query.data && !isOwner && query.data.shared_by !== actor.id) {
      throw new Error("Only the analysis owner can change this share.");
    }
    const payload = {
      [shareKey(data.resourceType)]: rootId,
      recipient_profile_id: targetProfileId,
      team_id: targetTeamId,
      access_level: data.accessLevel,
      can_share: isOwner ? Boolean(data.canShare) : false,
      shared_by: actor.id,
    };
    const write = query.data
      ? await admin.from(shareTable(data.resourceType)).update(payload).eq("id", query.data.id).select("id").single()
      : await admin.from(shareTable(data.resourceType)).insert(payload).select("id").single();
    if (write.error) throw new Error(write.error.message);
    await notifyShareRecipients(admin, recipients, actor.id, analysis, teamName);
    return { ok: true, id: write.data.id };
  });

export const updateAnalysisShare = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { resourceType?: ShareResource; shareId: string; accessLevel: ShareAccess; canShare?: boolean }) => input)
  .handler(async ({ data, context }) => {
    await assertPlanFeature(context.supabase, "professional", "updateStandaloneAnalysisShare");
    if (!(["view", "comment", "edit"] as string[]).includes(data.accessLevel)) throw new Error("Invalid access level.");
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const admin = supabaseAdmin as any;
    const actor = await profileForAuthUser(admin, context.userId);
    const { data: share, error } = await admin.from(shareTable(data.resourceType)).select("*").eq("id", data.shareId).maybeSingle();
    if (error) throw new Error(error.message);
    if (!share) throw new Error("Share not found.");
    const analysis = await loadShareResource(admin, share[shareKey(data.resourceType)], data.resourceType);
    const actorTeams = await getActorTeams(admin, actor.id);
    const teamIds = actorTeams.map((team: any) => team.id);
    const isOwner = analysis.owner_id === actor.id;
    if (!isOwner && share.shared_by !== actor.id) throw new Error("Only the analysis owner or the person who shared this access can change it.");
    if (data.canShare && !isOwner) throw new Error("Only the analysis owner can let another user reshare it.");
    if (!isOwner) {
      const actorLevel = await accessLevelForActor(admin, analysis, actor.id, teamIds);
      const requestedLevel = { view: 1, comment: 2, edit: 3 }[data.accessLevel];
      if (requestedLevel > actorLevel) throw new Error("You cannot grant more access than you have.");
    }
    const payload: Record<string, unknown> = { access_level: data.accessLevel };
    if (isOwner && data.canShare !== undefined) payload.can_share = Boolean(data.canShare);
    const { error: updateError } = await admin.from(shareTable(data.resourceType)).update(payload).eq("id", data.shareId);
    if (updateError) throw new Error(updateError.message);
    return { ok: true };
  });

export const removeAnalysisShare = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { resourceType?: ShareResource; shareId: string }) => input)
  .handler(async ({ data, context }) => {
    await assertPlanFeature(context.supabase, "professional", "removeStandaloneAnalysisShare");
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const admin = supabaseAdmin as any;
    const actor = await profileForAuthUser(admin, context.userId);
    const { data: share, error } = await admin.from(shareTable(data.resourceType)).select("*").eq("id", data.shareId).maybeSingle();
    if (error) throw new Error(error.message);
    if (!share) throw new Error("Share not found.");
    const analysis = await loadShareResource(admin, share[shareKey(data.resourceType)], data.resourceType);
    if (analysis.owner_id !== actor.id && share.shared_by !== actor.id) {
      throw new Error("Only the analysis owner or the person who shared this access can remove it.");
    }
    const { error: removeError } = await admin.from(shareTable(data.resourceType)).delete().eq("id", data.shareId);
    if (removeError) throw new Error(removeError.message);
    return { ok: true };
  });
