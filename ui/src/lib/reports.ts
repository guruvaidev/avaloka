import { supabase } from "@/integrations/supabase/client";

export type ReportUserDetails = {
  id: string;
  auth_user_id: string | null;
  email: string | null;
  name: string;
  role: string | null;
  avatar_url: string | null;
  initials: string;
};

export type ReportRecord = {
  id: string;
  organization_id: string;
  project_id: string | null;
  analysis_id: string;
  title: string;
  dataset_label: string | null;
  key_insights: unknown;
  status: string;
  created_by: ReportUserDetails;
  created_by_id?: string;
  created_at: string;
  updated_at: string;
  analyses?: {
    name: string | null;
    team_label: string | null;
    team_count: number | null;
    viz_config?: unknown;
    samples?: unknown;
  } | null;
  projects?: {
    id: string;
    name: string;
  } | null;
  shared_members?: { id: string; name: string; initials: string }[];
  creator?: ReportUserDetails | null;
};

function rawCreatedByOf(report: ReportRecord): string {
  return report.created_by_id ?? (typeof report.created_by === "string" ? report.created_by : report.created_by.id);
}

function initialsOf(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

function nameFromEmail(email: string | null | undefined): string {
  if (!email) return "";
  const localPart = email.split("@")[0] ?? "";
  return localPart
    .split(/[._-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

function metadataName(metadata: Record<string, unknown> | undefined): string {
  const value = metadata?.full_name ?? metadata?.name;
  return typeof value === "string" ? value.trim() : "";
}

async function currentOrgId(): Promise<string | null> {
  const { data, error } = await supabase.rpc("current_org_id");
  if (error) {
    console.error("current_org_id failed", error);
    return null;
  }
  return (data as string | null) ?? null;
}

export async function createReport(input: {
  projectId: string | null;
  analysisId: string;
  tabId?: string | null;
  title: string;
  datasetLabel?: string | null;
  keyInsights?: unknown;
}): Promise<ReportRecord> {
  const { data: userData, error: userErr } = await supabase.auth.getUser();
  if (userErr) throw userErr;
  const userId = userData.user?.id;
  if (!userId) throw new Error("Not signed in");

  const orgId = await currentOrgId();
  if (!orgId) throw new Error("Missing organization");

  const { data, error } = await supabase
    .from("reports")
    .insert({
      organization_id: orgId,
      project_id: input.projectId,
      analysis_id: input.analysisId,
      title: input.title || "Untitled report",
      dataset_label: input.datasetLabel ?? null,
      key_insights: input.keyInsights ?? [],
      created_by: userId,
    } as never)
    .select("*")
    .single();

  if (error) throw error;
  return data as ReportRecord;
}

export async function fetchReportProjects(): Promise<{ id: string; name: string }[]> {
  const { data, error } = await supabase
    .from("reports")
    .select("projects:project_id (id, name)")
    .not("project_id", "is", null);
  if (error) throw error;
  const seen = new Map<string, { id: string; name: string }>();
  const rows = [
    ...((data ?? []) as Array<{ projects: any }>),
    ...(await orgMemberReportRows()).map((r) => ({ projects: (r as any).projects })),
  ];
  for (const row of rows) {
    const p = Array.isArray(row.projects) ? row.projects[0] : row.projects;
    if (p?.id && !seen.has(p.id)) seen.set(p.id, { id: p.id, name: p.name });
  }
  return Array.from(seen.values());
}

/** Organization owners also see reports created by their members (no invite needed). */
async function orgMemberReportRows(projectId?: string | null): Promise<ReportRecord[]> {
  try {
    const { hasSupabaseSession } = await import("@/lib/has-session");
    if (!(await hasSupabaseSession())) return [];
    const { listOrgMemberReports } = await import("@/lib/org-owner-access.functions");
    return (((await listOrgMemberReports({ data: { projectId: projectId ?? null } })) ??
      []) as ReportRecord[]);
  } catch {
    return [];
  }
}

export async function fetchReports(projectId?: string | null): Promise<ReportRecord[]> {
  let query = supabase
    .from("reports")
    .select("*, analyses:analysis_id (name, team_label, team_count), projects:project_id (id, name)")
    .order("created_at", { ascending: false });
  if (projectId) query = query.eq("project_id", projectId);
  const { data, error } = await query;
  if (error) throw error;

  const byId = new Map<string, ReportRecord>();
  for (const row of [...((data ?? []) as ReportRecord[]), ...(await orgMemberReportRows(projectId))]) {
    if (!byId.has((row as any).id)) byId.set((row as any).id, row);
  }
  const merged = Array.from(byId.values()).sort((a, b) =>
    String((b as any).created_at).localeCompare(String((a as any).created_at)),
  );
  return enrichReports(merged);
}

export async function enrichReports(reports: ReportRecord[]): Promise<ReportRecord[]> {
  if (!reports.length) return reports;
  const { data: authData } = await supabase.auth.getUser();
  const currentUser = authData.user ?? null;
  const currentUserId = currentUser?.id ?? null;
  const currentUserEmail = currentUser?.email ?? null;
  const currentUserName =
    metadataName(currentUser?.user_metadata as Record<string, unknown> | undefined) || nameFromEmail(currentUserEmail);

  const analysisIds = Array.from(new Set(reports.map((r) => r.analysis_id).filter(Boolean)));
  const projectIds = Array.from(new Set(reports.map((r) => r.project_id).filter((v): v is string => !!v)));
  const reportIds = reports.map((r) => r.id);

  const resourceIds = [...analysisIds, ...projectIds, ...reportIds];
  if (!resourceIds.length) return reports;

  const { data: collabs } = await supabase
    .from("resource_collaborators")
    .select("resource_type,resource_id,app_user_id,app_users:app_user_id(id,email,auth_user_id)")
    .in("resource_id", resourceIds);

  type CollabRow = {
    resource_type: string;
    resource_id: string;
    app_user_id: string;
    app_users:
      | { id: string; email: string | null; auth_user_id: string | null }
      | { id: string; email: string | null; auth_user_id: string | null }[]
      | null;
  };
  const collabRows = (collabs ?? []) as unknown as CollabRow[];

  // Resolve names for collaborators from profiles (via auth_user_id)
  const collabAuthIds = Array.from(
    new Set(
      collabRows
        .map((c) => {
          const au = Array.isArray(c.app_users) ? c.app_users[0] : c.app_users;
          return au?.auth_user_id ?? null;
        })
        .filter((v): v is string => !!v),
    ),
  );
  const collabProfMap = new Map<
    string,
    { first_name: string | null; last_name: string | null; full_name: string | null }
  >();
  if (collabAuthIds.length) {
    const { data: cprofs } = await supabase
      .from("profiles")
      .select("id, user_id, first_name, last_name, full_name")
      .or(`id.in.(${collabAuthIds.join(",")}),user_id.in.(${collabAuthIds.join(",")})`);
    for (const p of (cprofs ?? []) as Array<{
      id: string;
      user_id: string | null;
      first_name: string | null;
      last_name: string | null;
      full_name: string | null;
    }>) {
      collabProfMap.set(p.id, p);
      if (p.user_id) collabProfMap.set(p.user_id, p);
    }
  }

  const byAnalysis = new Map<string, { id: string; name: string; initials: string }[]>();
  const byProject = new Map<string, { id: string; name: string; initials: string }[]>();
  const byReport = new Map<string, { id: string; name: string; initials: string }[]>();
  for (const c of collabRows) {
    const au = Array.isArray(c.app_users) ? c.app_users[0] : c.app_users;
    const prof = au?.auth_user_id ? collabProfMap.get(au.auth_user_id) : null;
    const fullName = prof ? prof.full_name || `${prof.first_name ?? ""} ${prof.last_name ?? ""}`.trim() : "";
    const name = fullName || au?.email || "Unknown";
    const member = { id: c.app_user_id, name, initials: initialsOf(name) || "U" };
    const map =
      c.resource_type === "analysis"
        ? byAnalysis
        : c.resource_type === "project"
          ? byProject
          : c.resource_type === "report"
            ? byReport
            : null;
    if (!map) continue;
    const arr = map.get(c.resource_id) ?? [];
    if (!arr.some((m) => m.id === member.id)) arr.push(member);
    map.set(c.resource_id, arr);
  }

  for (const r of reports) {
    const a = byAnalysis.get(r.analysis_id) ?? [];
    const p = r.project_id ? (byProject.get(r.project_id) ?? []) : [];
    const rep = byReport.get(r.id) ?? [];
    const merged: typeof a = [];
    for (const m of [...rep, ...a, ...p]) if (!merged.some((x) => x.id === m.id)) merged.push(m);
    r.shared_members = merged;
  }

  // Cross-org enrichment: get_report_teams is SECURITY DEFINER and returns the
  // full team (creator + all collaborators across report/analysis/project) for
  // every report the caller can access, even collaborators outside the
  // caller's own organization which the RLS-filtered read above misses.
  const teamMemberByReport = new Map<
    string,
    Map<string, { name: string | null; email: string | null; avatar_url: string | null }>
  >();
  try {
    const { data: teamRows, error: teamErr } = await supabase.rpc("get_report_teams", {
      _report_ids: reportIds,
    } as never);
    if (!teamErr && Array.isArray(teamRows)) {
      const byReportFull = new Map<string, { id: string; name: string; initials: string }[]>();
      for (const row of teamRows as Array<{
        report_id: string;
        member_id: string;
        name: string | null;
        email: string | null;
        avatar_url: string | null;
      }>) {
        const name = (row.name && row.name.trim()) || row.email || "Unknown";
        const arr = byReportFull.get(row.report_id) ?? [];
        if (!arr.some((m) => m.id === row.member_id)) {
          arr.push({ id: row.member_id, name, initials: initialsOf(name) || "U" });
        }
        byReportFull.set(row.report_id, arr);
        const mm = teamMemberByReport.get(row.report_id) ?? new Map();
        mm.set(row.member_id, { name: row.name, email: row.email, avatar_url: row.avatar_url });
        teamMemberByReport.set(row.report_id, mm);
      }
      for (const r of reports) {
        const full = byReportFull.get(r.id);
        if (full && full.length) r.shared_members = full;
      }
    }
  } catch (err) {
    console.warn("get_report_teams enrichment failed", err);
  }

  // Attach creator info (name, role, avatar) from profiles + app_users.
  // reports.created_by has existed as either auth user id, profile id, or app_user id,
  // so resolve all three shapes before falling back to Unknown.
  const creatorIds = Array.from(new Set(reports.map((r) => rawCreatedByOf(r)).filter(Boolean)));
  if (creatorIds.length) {
    const { data: aus } = await supabase
      .from("app_users")
      .select("id, auth_user_id, profile_id, email, role")
      .or(`id.in.(${creatorIds.join(",")}),auth_user_id.in.(${creatorIds.join(",")})`);

    const profileLookupIds = new Set(creatorIds);
    for (const a of (aus ?? []) as Array<{ auth_user_id: string | null; profile_id: string | null }>) {
      if (a.auth_user_id) profileLookupIds.add(a.auth_user_id);
      if (a.profile_id) profileLookupIds.add(a.profile_id);
    }

    const profileIdList = Array.from(profileLookupIds);
    const { data: profs } = await supabase
      .from("profiles")
      .select("id, user_id, first_name, last_name, full_name, avatar_url")
      .or(`id.in.(${profileIdList.join(",")}),user_id.in.(${profileIdList.join(",")})`);

    const profMap = new Map<
      string,
      {
        id: string;
        user_id: string | null;
        first_name: string | null;
        last_name: string | null;
        full_name: string | null;
        avatar_url: string | null;
      }
    >();
    for (const p of (profs ?? []) as Array<{
      id: string;
      user_id: string | null;
      first_name: string | null;
      last_name: string | null;
      full_name: string | null;
      avatar_url: string | null;
    }>) {
      profMap.set(p.id, p);
      if (p.user_id) profMap.set(p.user_id, p);
    }
    const roleMap = new Map<string, string | null>();
    const emailMap = new Map<string, string | null>();
    const appUserAuthIdMap = new Map<string, string | null>();
    const appUserProfileIdMap = new Map<string, string | null>();
    for (const a of (aus ?? []) as Array<{
      id: string;
      auth_user_id: string | null;
      profile_id: string | null;
      email: string | null;
      role: string | null;
    }>) {
      roleMap.set(a.id, a.role);
      emailMap.set(a.id, a.email);
      appUserAuthIdMap.set(a.id, a.auth_user_id);
      appUserProfileIdMap.set(a.id, a.profile_id);
      if (a.auth_user_id) {
        roleMap.set(a.auth_user_id, a.role);
        emailMap.set(a.auth_user_id, a.email);
      }
      if (a.profile_id) {
        roleMap.set(a.profile_id, a.role);
        emailMap.set(a.profile_id, a.email);
      }
    }
    for (const r of reports) {
      const rawCreatedBy = rawCreatedByOf(r);
      const appAuthId = appUserAuthIdMap.get(rawCreatedBy);
      const appProfileId = appUserProfileIdMap.get(rawCreatedBy);
      const p =
        profMap.get(rawCreatedBy) ||
        (appAuthId ? profMap.get(appAuthId) : undefined) ||
        (appProfileId ? profMap.get(appProfileId) : undefined);
      const profileAuthId = p?.user_id ?? null;
      const isCurrentCreator =
        !!currentUserId &&
        (rawCreatedBy === currentUserId || appAuthId === currentUserId || profileAuthId === currentUserId);
      const fullName = p ? p.full_name || `${p.first_name ?? ""} ${p.last_name ?? ""}`.trim() : "";
      const fallback = teamMemberByReport.get(r.id)?.get(rawCreatedBy) ?? null;
      const email =
        emailMap.get(rawCreatedBy) ||
        (appAuthId ? emailMap.get(appAuthId) : null) ||
        (appProfileId ? emailMap.get(appProfileId) : null) ||
        (isCurrentCreator ? currentUserEmail : null) ||
        fallback?.email ||
        null;
      const name =
        fullName ||
        (isCurrentCreator ? currentUserName : "") ||
        (fallback?.name && fallback.name.trim()) ||
        email ||
        "Unknown";
      const creator: ReportUserDetails = {
        id: rawCreatedBy,
        auth_user_id: appAuthId ?? profileAuthId ?? (isCurrentCreator ? currentUserId : null),
        email,
        name,
        role: roleMap.get(rawCreatedBy) ?? (appAuthId ? roleMap.get(appAuthId) : null) ?? null,
        avatar_url: p?.avatar_url ?? fallback?.avatar_url ?? null,
        initials: initialsOf(name) || "U",
      };
      r.created_by_id = rawCreatedBy;
      r.created_by = creator;
      r.creator = creator;
    }
  } else {
    // No profile/app_user access (e.g. shared user across orgs). Build creators
    // purely from get_report_teams so we still show name/avatar instead of a uuid.
    for (const r of reports) {
      const rawCreatedBy = rawCreatedByOf(r);
      const fallback = teamMemberByReport.get(r.id)?.get(rawCreatedBy) ?? null;
      const name = (fallback?.name && fallback.name.trim()) || fallback?.email || "Unknown";
      const creator: ReportUserDetails = {
        id: rawCreatedBy,
        auth_user_id: null,
        email: fallback?.email ?? null,
        name,
        role: null,
        avatar_url: fallback?.avatar_url ?? null,
        initials: initialsOf(name) || "U",
      };
      r.created_by_id = rawCreatedBy;
      r.created_by = creator;
      r.creator = creator;
    }
  }

  return reports;
}

export async function fetchReportById(id: string): Promise<ReportRecord | null> {
  const { data } = await supabase
    .from("reports")
    .select(
      "*, analyses:analysis_id (name, team_label, team_count, viz_config, samples), projects:project_id (id, name)",
    )
    .eq("id", id)
    .maybeSingle();

  // Org owners can open member reports even without an explicit invite.
  const row =
    data ?? (await orgMemberReportRows()).find((r) => (r as any).id === id) ?? null;
  if (!row) return null;
  const [enriched] = await enrichReports([row as ReportRecord]);
  return enriched ?? null;
}

export async function deleteReport(id: string): Promise<void> {
  const { error } = await supabase.from("reports").delete().eq("id", id);
  if (error) throw error;
}

export async function renameReport(id: string, title: string): Promise<void> {
  const { error } = await supabase
    .from("reports")
    .update({ title } as never)
    .eq("id", id);
  if (error) throw error;
}
