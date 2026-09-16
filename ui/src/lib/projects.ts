import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { requireCurrentProfileId, getCurrentProfileId } from "@/lib/current-profile";
import { getCurrentOrgId } from "@/lib/current-org";
import { notify } from "@/lib/notifications";

async function notifySelf(title: string, body?: string) {
  const pid = await getCurrentProfileId();
  if (!pid) return;
  await notify({ recipientId: pid, type: "generic", title, body });
}

export type Project = {
  id: string;
  name: string;
  description: string | null;
  bookmarked: boolean;
  archived: boolean;
  pinned: boolean;
};



async function fetchProjects(): Promise<Project[]> {
  const { data, error } = await supabase
    .from("projects")
    .select("id,name,description,bookmarked,archived,pinned")
    .is("deleted_at", null)
    .order("pinned", { ascending: false })
    .order("created_at", { ascending: false });
  if (error) throw error;
  const own = (data ?? []) as Project[];

  // Organization owners also see projects created by their members (no invite needed).
  let orgRows: Project[] = [];
  try {
    const { hasSupabaseSession } = await import("@/lib/has-session");
    const { listOrgMemberProjects } = await import("@/lib/org-owner-access.functions");
    orgRows = (await hasSupabaseSession())
      ? (((await listOrgMemberProjects()) ?? []) as Project[])
      : [];
  } catch {
    orgRows = [];
  }

  const byId = new Map<string, Project>();
  for (const row of [...own, ...orgRows]) if (!byId.has(row.id)) byId.set(row.id, row);
  return Array.from(byId.values()).sort(
    (a, b) => Number(b.pinned) - Number(a.pinned),
  );
}

export const projectsKey = ["projects"] as const;

/** Create a project and its first analysis (e.g. after a dashboard file upload). */
export async function createProjectWithAnalysis(name: string): Promise<Project> {
  const profileId = await requireCurrentProfileId();
  const organizationId = await getCurrentOrgId();
  const { data: project, error: projectError } = await supabase
    .from("projects")
    .insert({ name, owner_id: profileId, organization_id: organizationId } as never)
    .select("id,name,description,bookmarked,archived,pinned")
    .single();
  if (projectError) throw projectError;

  const { error: analysisError } = await supabase.from("analyses").insert({
    owner_id: profileId,
    organization_id: organizationId,
    project_id: project.id,
    name,
    team_label: "Team",
    team_count: 0,
  } as never);
  if (analysisError) throw analysisError;

  return project as Project;
}


export function useProjects() {
  return useQuery({
    queryKey: projectsKey,
    queryFn: async () => {
      return fetchProjects();
    },
  });
}

export function useProjectMutations() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: projectsKey });

  const create = useMutation({
    mutationFn: async (input: string | { name: string; description?: string | null }) => {
      const name = typeof input === "string" ? input : input.name;
      const description = typeof input === "string" ? null : (input.description ?? null);
      const profileId = await requireCurrentProfileId();
      const organizationId = await getCurrentOrgId();
      const { data, error } = await supabase
        .from("projects")
        .insert({
          name,
          description: description ?? null,
          owner_id: profileId,
          organization_id: organizationId,
        } as never)
        .select("id,name,description,bookmarked,archived,pinned")
        .single();
      if (error) throw error;
      await notifySelf("Project created", `“${name}” was created.`);
      return data as Project;
    },
    onSuccess: invalidate,
  });

  const updateDescription = useMutation({
    mutationFn: async ({ id, description }: { id: string; description: string | null }) => {
      const { error } = await supabase
        .from("projects")
        .update({ description } as never)
        .eq("id", id);
      if (error) throw error;
    },
    onSuccess: invalidate,
  });

  const rename = useMutation({
    mutationFn: async ({ id, name }: { id: string; name: string }) => {
      const { error } = await supabase.from("projects").update({ name }).eq("id", id);
      if (error) throw error;
      await notifySelf("Project renamed", `Project renamed to “${name}”.`);
    },
    onSuccess: invalidate,
  });

  const toggleFlag = useMutation({
    mutationFn: async ({
      id,
      field,
      value,
    }: {
      id: string;
      field: "bookmarked" | "archived" | "pinned";
      value: boolean;
    }) => {
      const patch: Record<string, boolean> = { [field]: value };
      const { error } = await supabase.from("projects").update(patch as never).eq("id", id);
      if (error) throw error;
      const labels: Record<typeof field, [string, string]> = {
        bookmarked: ["Project bookmarked", "Project bookmark removed"],
        pinned: ["Project pinned", "Project unpinned"],
        archived: ["Project archived", "Project unarchived"],
      };
      const [on, off] = labels[field];
      await notifySelf(value ? on : off);
    },
    onSuccess: invalidate,
  });

  const softDelete = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase
        .from("projects")
        .update({ deleted_at: new Date().toISOString() })
        .eq("id", id);
      if (error) throw error;
      await notifySelf("Project deleted");
    },
    onSuccess: invalidate,
  });

  return { create, rename, updateDescription, toggleFlag, softDelete };
}
