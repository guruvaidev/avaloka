import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { displayNameOf, fetchDisplayProfiles, indexDisplayProfiles } from "@/lib/display-profiles";
import { requireCurrentProfileId, getCurrentProfileId } from "@/lib/current-profile";
import { notify } from "@/lib/notifications";
import { getCurrentOrgId } from "@/lib/current-org";
import { getCachedAuthUser, getCachedAuthUserId } from "@/lib/auth-user";
import { useEffect, useState } from "react";

async function notifySelf(title: string, body?: string) {
  const pid = await getCurrentProfileId();
  if (!pid) return;
  await notify({ recipientId: pid, type: "generic", title, body });
}


export type Analysis = {
  id: string;
  project_id: string | null;
  name: string;
  team_label: string;
  team_count: number;
  created_by_name: string | null;
  created_by_role: string | null;
  created_by_initials: string | null;
  created_by_avatar: string | null;
  created_at: string;
  is_bookmarked: boolean;
  is_archived: boolean;
  is_pinned: boolean;
};

async function uid() {
  const data = { user: await getCachedAuthUser() };
  return data.user?.id ?? null;
}

export const analysesKey = (projectId: string | null) => ["analyses", projectId] as const;


const analysisSelect =
  "id,project_id,owner_id,name,team_label,team_count,created_at,is_bookmarked,is_archived,is_pinned";

function initialsOf(name: string) {
  const parts = name.trim().split(/\s+/);
  if (!parts[0]) return undefined;
  return (parts[0][0] + (parts[1]?.[0] ?? "")).toUpperCase();
}

/** Fill in owner name / avatar for a list of analyses from the profiles table. */
async function enrichOwners(rows: Analysis[], ownerIds: (string | null)[]): Promise<Analysis[]> {
  const ids = Array.from(new Set(ownerIds.filter(Boolean) as string[]));
  if (ids.length === 0) return rows;

  // Owners of shared analyses may be in other organisations: display fields only.
  const data = await fetchDisplayProfiles(supabase, ids);

  const byId = indexDisplayProfiles(data);

  return rows.map((r, i) => {
    const owner = ownerIds[i] ? byId.get(ownerIds[i] as string) : undefined;
    const name = displayNameOf(owner);
    if (!name && !owner?.avatar_url) return r;
    return {
      ...r,
      created_by_name: name || r.created_by_name,
      created_by_initials: name ? initialsOf(name) ?? null : r.created_by_initials,
      created_by_avatar: owner?.avatar_url ?? null,
    };
  });
}

function normalizeAnalysis(row: {
  id: string;
  project_id: string | null;
  name: string;
  team_label: string | null;
  team_count: number | null;
  created_at: string;
  is_bookmarked?: boolean | null;
  is_archived?: boolean | null;
  is_pinned?: boolean | null;
}): Analysis {
  return {
    id: row.id,
    project_id: row.project_id,
    name: row.name,
    team_label: row.team_label ?? "Team",
    team_count: row.team_count ?? 0,
    created_by_name: null,
    created_by_role: null,
    created_by_initials: null,
    created_by_avatar: null,
    created_at: row.created_at,
    is_bookmarked: !!row.is_bookmarked,
    is_archived: !!row.is_archived,
    is_pinned: !!row.is_pinned,
  };
}





/** Organization owners also see analyses created by their members (no invite needed). */
async function orgMemberAnalysisRows(projectId: string | null): Promise<any[]> {
  try {
    const { hasSupabaseSession } = await import("@/lib/has-session");
    if (!(await hasSupabaseSession())) return [];
    const { listOrgMemberAnalyses } = await import("@/lib/org-owner-access.functions");
    return ((await listOrgMemberAnalyses({ data: { projectId } })) ?? []) as any[];
  } catch {
    return [];
  }
}

function mergeAnalysisRows(a: any[], b: any[]): any[] {
  const byId = new Map<string, any>();
  for (const row of [...a, ...b]) if (!byId.has(row.id)) byId.set(row.id, row);
  return Array.from(byId.values()).sort((x, y) =>
    String(y.created_at).localeCompare(String(x.created_at)),
  );
}

export function useAnalyses(projectId: string | null) {
  return useQuery({
    queryKey: analysesKey(projectId),
    refetchInterval: 15_000,
    enabled: !!projectId,
    queryFn: async (): Promise<Analysis[]> => {
      if (!projectId) return [];

      const { data, error } = await supabase
        .from("analyses")
        .select(analysisSelect)
        .eq("project_id", projectId)
        .is("parent_analysis_id", null)
        .order("created_at", { ascending: false });

      if (error) throw error;
      const list = mergeAnalysisRows(data ?? [], await orgMemberAnalysisRows(projectId));
      return enrichOwners(list.map(normalizeAnalysis), list.map((r: { owner_id: string | null }) => r.owner_id));
    },
  });
}

/** Cache key for the sidebar Analysis list; scoped per signed-in auth user. */
export function standaloneAnalysesKey(authUserId: string | null) {
  return ["analyses", "standalone", authUserId ?? "anonymous"] as const;
}

/** Tracks the signed-in auth user id so per-user caches never leak across accounts. */
function useAuthUserId(): string | null {
  const [id, setId] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    void getCachedAuthUserId().then((v) => alive && setId(v));
    const { data } = supabase.auth.onAuthStateChange((_e, session) => setId(session?.user?.id ?? null));
    return () => {
      alive = false;
      data.subscription.unsubscribe();
    };
  }, []);
  return id;
}

/** Enterprise keeps its existing owner-only standalone Analysis list. */
export async function fetchOwnedStandaloneAnalyses(
  client: typeof supabase,
  profileId: string | null,
): Promise<any[]> {
  if (!profileId) return [];
  const { data, error } = await client
    .from("analyses")
    .select(analysisSelect)
    .eq("owner_id", profileId)
    .is("project_id", null)
    .is("parent_analysis_id", null)
    .order("created_at", { ascending: false });
  if (error) throw error;
  return ((data ?? []) as any[]).filter((r) => r.owner_id === profileId);
}

/** Include standalone analyses shared with the caller, including team grants.
 * The authenticated client's row policies enforce the recipient's access.
 * Analyses within projects continue to appear under their project.
 */
export async function fetchVisibleStandaloneAnalyses(
  client: typeof supabase,
  profileId: string | null,
): Promise<any[]> {
  if (!profileId) return [];
  const { data: tier, error: planError } = await client.rpc("current_plan_tier");
  if (planError) throw planError;
  if (tier === "enterprise") return fetchOwnedStandaloneAnalyses(client, profileId);

  const { data, error } = await client
    .from("analyses")
    .select(analysisSelect)
    .is("project_id", null)
    .is("parent_analysis_id", null)
    .order("created_at", { ascending: false });
  if (error) throw error;
  return data ?? [];
}

export function useStandaloneAnalyses() {
  const authUserId = useAuthUserId();
  return useQuery({
    queryKey: standaloneAnalysesKey(authUserId),
    refetchInterval: 15_000,
    enabled: !!authUserId,
    queryFn: async (): Promise<Analysis[]> => {
      const profileId = await getCurrentProfileId();
      const list = await fetchVisibleStandaloneAnalyses(supabase, profileId);
      return enrichOwners(list.map(normalizeAnalysis), list.map((r: { owner_id: string | null }) => r.owner_id));
    },
  });
}

export function useAnalysisMutations(projectId: string | null) {
  const qc = useQueryClient();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: analysesKey(projectId) });
    qc.invalidateQueries({ queryKey: ["analyses", "standalone"] });
  };

  const create = useMutation({
    mutationFn: async (input: { name: string; project_id?: string | null }) => {
      const owner_id = await requireCurrentProfileId();
      const organization_id = await getCurrentOrgId();
      const { data, error } = await supabase
        .from("analyses")
        .insert({
          owner_id,
          organization_id,
          project_id: input.project_id ?? null,
          name: input.name,
          team_label: "Team",
          team_count: 0,
        })
        .select("*")
        .single();
      if (error) throw error;
      await notifySelf("Analysis created", `“${input.name}” was created.`);
      return data as Analysis;
    },
    onSuccess: invalidate,
  });

  const rename = useMutation({
    mutationFn: async ({ id, name }: { id: string; name: string }) => {
      const { error } = await supabase.from("analyses").update({ name }).eq("id", id);
      if (error) throw error;
      await notifySelf("Analysis renamed", `Analysis renamed to “${name}”.`);
    },
    onSuccess: invalidate,
  });

  const softDelete = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from("analyses").delete().eq("id", id);
      if (error) throw error;
      await notifySelf("Analysis deleted");
    },
    onSuccess: invalidate,
  });
  const setFlag = useMutation({
    mutationFn: async ({
      id,
      field,
      value,
    }: {
      id: string;
      field: "is_bookmarked" | "is_archived" | "is_pinned";
      value: boolean;
    }) => {
      const { error } = await supabase
        .from("analyses")
        .update({ [field]: value })
        .eq("id", id);
      if (error) throw error;
      const labels: Record<typeof field, [string, string]> = {
        is_bookmarked: ["Analysis bookmarked", "Analysis bookmark removed"],
        is_pinned: ["Analysis pinned", "Analysis unpinned"],
        is_archived: ["Analysis archived", "Analysis unarchived"],
      };
      const [on, off] = labels[field];
      await notifySelf(value ? on : off);
    },
    onSuccess: invalidate,
  });

  const moveToProject = useMutation({
    mutationFn: async ({ id, project_id }: { id: string; project_id: string | null }) => {
      const { error } = await supabase.from("analyses").update({ project_id }).eq("id", id).select("id").single();
      if (error) throw error;
      await notifySelf(project_id ? "Analysis moved to project" : "Analysis moved out of project");
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["analyses"] });
    },
  });

  return { create, rename, softDelete, setFlag, moveToProject };
}
