// Share links + collaborator access for projects and analyses.
// Kept separate from chat, comments, and insight features.

import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";

export type ResourceType = "project" | "analysis" | "report";

export type CollaboratorMember = {
  id: string;
  appUserId: string;
  name: string;
  role: string;
  initials: string;
  access: string;
};

export type DepartmentOption = {
  name: string;
  memberCount: number;
};

export type DepartmentMemberPreview = {
  id: string;
  name: string;
  role: string;
  initials: string;
  avatar: string | null;
};

export type AppUserRow = {
  id: string;
  name: string;
  email?: string;
  role: string;
  department: string;
  status: string;
  avatar?: string | null;
};

const LINK_UI_TO_DB: Record<string, string> = {
  View: "view",
  Edit: "edit",
  Comment: "comment",
  Restricted: "restricted",
};

const LINK_DB_TO_UI: Record<string, string> = {
  view: "View",
  edit: "Edit",
  comment: "Comment",
  restricted: "Restricted",
};

const ACCESS_UI_TO_DB: Record<string, string> = {
  View: "view",
  Edit: "edit",
  Comment: "comment",
  Admin: "edit",
};

const ACCESS_DB_TO_UI: Record<string, string> = {
  view: "View",
  edit: "Edit",
  comment: "Comment",
};

export function initialsFrom(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

export function buildDepartmentOptions(
  deptRows: Array<{ department: string; status?: string }>,
  appUsers: AppUserRow[],
): DepartmentOption[] {
  const activeUsers = appUsers.filter((u) => u.status === "Active");
  const countByDept = new Map<string, number>();
  const seenByDept = new Map<string, Set<string>>();

  for (const user of activeUsers) {
    const name = user.department?.trim();
    if (!name) continue;
    if (!seenByDept.has(name)) seenByDept.set(name, new Set());
    const seen = seenByDept.get(name)!;
    if (seen.has(user.id)) continue;
    seen.add(user.id);
    countByDept.set(name, (countByDept.get(name) ?? 0) + 1);
  }

  const names = new Set<string>();
  for (const row of deptRows) {
    if (row.status && row.status !== "Active") continue;
    const name = row.department?.trim();
    if (name) names.add(name);
  }
  for (const name of countByDept.keys()) names.add(name);

  return Array.from(names)
    .sort((a, b) => a.localeCompare(b))
    .map((name) => ({ name, memberCount: countByDept.get(name) ?? 0 }));
}

export function filterActiveAppUsers(
  appUsers: AppUserRow[],
  query: string,
  excludeIds?: Set<string>,
): DepartmentMemberPreview[] {
  const q = query.trim().toLowerCase();
  const seen = new Set<string>();

  return appUsers
    .filter((u) => u.status === "Active")
    .filter((u) => !excludeIds?.has(u.id))
    .filter((u) => {
      if (!q) return true;
      const name = u.name?.toLowerCase() ?? "";
      const email = u.email?.toLowerCase() ?? "";
      return name.includes(q) || email.includes(q);
    })
    .filter((u) => {
      if (seen.has(u.id)) return false;
      seen.add(u.id);
      return true;
    })
    .map((u) => ({
      id: u.id,
      name: u.name,
      role: u.role,
      initials: initialsFrom(u.name),
      avatar: u.avatar ?? null,
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function filterDepartmentMembers(appUsers: AppUserRow[], department: string): DepartmentMemberPreview[] {
  const dept = department.trim();
  if (!dept) return [];

  const seen = new Set<string>();
  return appUsers
    .filter((u) => u.status === "Active" && u.department?.trim() === dept)
    .filter((u) => {
      if (seen.has(u.id)) return false;
      seen.add(u.id);
      return true;
    })
    .map((u) => ({
      id: u.id,
      name: u.name,
      role: u.role,
      initials: initialsFrom(u.name),
      avatar: u.avatar ?? null,
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function mergeCollaboratorMembers(
  rows: Array<{ id: string; app_user_id: string; access_level: string }>,
  appUsers: AppUserRow[],
): CollaboratorMember[] {
  const userById = new Map(appUsers.map((u) => [u.id, u]));
  return rows.map((row) => {
    const au = userById.get(row.app_user_id);
    const name = au?.name ?? "Unknown";
    return {
      id: row.id,
      appUserId: row.app_user_id,
      name,
      role: au?.role ?? "",
      initials: initialsFrom(name),
      access: ACCESS_DB_TO_UI[row.access_level] ?? "View",
    };
  });
}

async function currentUserId(): Promise<string | null> {
  const data = { user: await getCachedAuthUser() };
  return data.user?.id ?? null;
}

function sharedPathForResource(resourceType: ResourceType, token: string): string {
  if (resourceType === "report") return `/shared-report/${token}`;
  return `/shared/${token}`;
}

export function shareLinkDisplayUrl(token: string, resourceType: ResourceType = "analysis"): string {
  const origin = typeof window !== "undefined" ? window.location.origin : "https://app.example.com";
  const host = origin.replace(/^https?:\/\//, "");
  const short = token.slice(0, 8);
  const prefix = resourceType === "report" ? "shared-report" : "shared";
  return `${host}/${prefix}/${short}…`;
}

export function shareLinkFullUrl(token: string, resourceType: ResourceType = "analysis"): string {
  const origin = typeof window !== "undefined" ? window.location.origin : "https://app.example.com";
  return `${origin}${sharedPathForResource(resourceType, token)}`;
}

export async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall through to execCommand (non-secure contexts).
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    return copied;
  } catch {
    return false;
  }
}

export async function ensureResourceShareLink(
  resourceType: ResourceType,
  resourceId: string,
): Promise<{ token: string; linkAccess: string; isPublic: boolean }> {
  const { data: existing, error: readErr } = await supabase
    .from("resource_share_links")
    .select("share_token,link_access,is_public")
    .eq("resource_type", resourceType)
    .eq("resource_id", resourceId)
    .maybeSingle();
  if (readErr) throw readErr;
  if (existing) {
    return {
      token: existing.share_token,
      linkAccess: LINK_DB_TO_UI[existing.link_access] ?? "View",
      isPublic: Boolean((existing as any).is_public),
    };
  }

  const { data, error } = await supabase
    .from("resource_share_links")
    .insert({ resource_type: resourceType, resource_id: resourceId })
    .select("share_token,link_access,is_public")
    .single();
  if (error) throw error;
  return {
    token: data.share_token,
    linkAccess: LINK_DB_TO_UI[data.link_access] ?? "View",
    isPublic: Boolean((data as any).is_public),
  };
}

export async function updateResourceShareLinkAccess(
  resourceType: ResourceType,
  resourceId: string,
  linkAccessUi: string,
): Promise<void> {
  const link_access = LINK_UI_TO_DB[linkAccessUi] ?? "view";
  const { error } = await supabase
    .from("resource_share_links")
    .update({ link_access, updated_at: new Date().toISOString() })
    .eq("resource_type", resourceType)
    .eq("resource_id", resourceId);
  if (error) throw error;
}

export async function updateResourceShareLinkPublic(
  resourceType: ResourceType,
  resourceId: string,
  isPublic: boolean,
): Promise<void> {
  const { error } = await supabase
    .from("resource_share_links")
    .update({ is_public: isPublic, updated_at: new Date().toISOString() })
    .eq("resource_type", resourceType)
    .eq("resource_id", resourceId);
  if (error) throw error;
}


export async function loadResourceCollaboratorRows(
  resourceType: ResourceType,
  resourceId: string,
): Promise<Array<{ id: string; app_user_id: string; access_level: string }>> {
  const { data, error } = await supabase
    .from("resource_collaborators")
    .select("id,app_user_id,access_level")
    .eq("resource_type", resourceType)
    .eq("resource_id", resourceId)
    .order("created_at", { ascending: true });
  if (error) throw error;
  return data ?? [];
}

export async function updateResourceCollaboratorAccess(collaboratorId: string, accessUi: string): Promise<void> {
  const access_level = ACCESS_UI_TO_DB[accessUi] ?? "view";
  const { error } = await supabase
    .from("resource_collaborators")
    .update({ access_level, updated_at: new Date().toISOString() })
    .eq("id", collaboratorId);
  if (error) throw error;
}

export async function removeResourceCollaborator(collaboratorId: string): Promise<void> {
  const { error } = await supabase.from("resource_collaborators").delete().eq("id", collaboratorId);
  if (error) throw error;
}

export async function inviteAppUsersToResource(input: {
  resourceType: ResourceType;
  resourceId: string;
  accessUi: string;
  appUserIds: string[];
  appUsers?: AppUserRow[];
}): Promise<number> {
  const invitedBy = await currentUserId();
  const access_level = ACCESS_UI_TO_DB[input.accessUi] ?? "view";
  const uniqueUserIds = [...new Set(input.appUserIds)];
  if (!uniqueUserIds.length) return 0;

  const userById = new Map((input.appUsers ?? []).map((u) => [u.id, u]));

  const rows = uniqueUserIds.map((appUserId) => ({
    resource_type: input.resourceType,
    resource_id: input.resourceId,
    app_user_id: appUserId,
    access_level,
    department: userById.get(appUserId)?.department?.trim() || "General",
    invited_by: invitedBy,
  }));

  const { error } = await supabase
    .from("resource_collaborators")
    .upsert(rows, { onConflict: "resource_type,resource_id,app_user_id", ignoreDuplicates: true });
  if (error) throw error;

  // Fire-and-forget: notify invitees.
  (async () => {
    try {
      const { data: users } = await supabase
        .from("app_users")
        .select("id,auth_user_id,email,full_name")
        .in("id", uniqueUserIds);
      const authIds = ((users ?? []) as any[])
        .map((u) => u.auth_user_id)
        .filter((v): v is string => !!v);
      if (!authIds.length) return;
      const { data: profs } = await supabase
        .from("profiles")
        .select("id,user_id")
        .in("user_id", authIds);
      const recipients = ((profs ?? []) as { id: string }[]).map((p) => p.id);
      if (!recipients.length) return;
      const { notifyMany } = await import("@/lib/notifications");
      await notifyMany(recipients, {
        type: "collaborator.invited",
        title: "You were invited to collaborate",
        body: `You've been added as ${input.accessUi.toLowerCase()} on a ${input.resourceType}.`,
        resourceType: input.resourceType,
        resourceId: input.resourceId,
        link:
          input.resourceType === "analysis"
            ? `/analysis?id=${input.resourceId}`
            : input.resourceType === "report"
              ? `/reports/${input.resourceId}`
              : "/dashboard",
      });
    } catch (e) {
      console.warn("notify(collaborator.invited) failed", e);
    }
  })();

  return rows.length;
}
