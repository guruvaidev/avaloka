import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

/**
 * Extra rows an organization OWNER should see even without being invited:
 * everything created by members of their organization.
 */

export const listOrgMemberProjects = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const { resolveOrgOwnerScope } = await import("./org-owner-access.server");
    const admin = supabaseAdmin as any;
    const scope = await resolveOrgOwnerScope(admin, context.userId as string);
    if (!scope.orgIds.length) return [] as any[];

    const { data, error } = await admin
      .from("projects")
      .select("id,name,description,bookmarked,archived,pinned,created_at")
      .in("organization_id", scope.orgIds)
      .is("deleted_at", null)
      .order("created_at", { ascending: false });
    if (error) throw error;
    return (data ?? []) as any[];
  });


export const listOrgMemberAnalyses = createServerFn({ method: "GET" })
  .inputValidator((data: { projectId?: string | null }) => ({
    projectId: data?.projectId ?? null,
  }))
  .middleware([requireSupabaseAuth])
  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const { resolveOrgOwnerScope } = await import("./org-owner-access.server");
    const admin = supabaseAdmin as any;
    const scope = await resolveOrgOwnerScope(admin, context.userId as string);
    if (!scope.orgIds.length) return [] as any[];

    let query = admin
      .from("analyses")
      .select(
        "id,project_id,owner_id,name,team_label,team_count,created_at,updated_at,filename,dataset_id,parent_analysis_id",
      )
      .in("organization_id", scope.orgIds)
      .is("parent_analysis_id", null)
      .order("created_at", { ascending: false });


    query = data.projectId
      ? query.eq("project_id", data.projectId)
      : query.is("project_id", null);

    const { data: rows, error } = await query;
    if (error) throw error;
    return (rows ?? []) as any[];
  });

export const listOrgMemberReports = createServerFn({ method: "GET" })
  .inputValidator((data: { projectId?: string | null }) => ({
    projectId: data?.projectId ?? null,
  }))
  .middleware([requireSupabaseAuth])
  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const { resolveOrgOwnerScope } = await import("./org-owner-access.server");
    const admin = supabaseAdmin as any;
    const scope = await resolveOrgOwnerScope(admin, context.userId as string);
    if (!scope.orgIds.length) return [] as any[];

    let query = admin
      .from("reports")
      .select(
        "*, analyses:analysis_id (name, team_label, team_count), projects:project_id (id, name)",
      )
      .in("organization_id", scope.orgIds)
      .order("created_at", { ascending: false });
    if (data.projectId) query = query.eq("project_id", data.projectId);

    const { data: rows, error } = await query;
    if (error) throw error;
    return (rows ?? []) as any[];
  });
