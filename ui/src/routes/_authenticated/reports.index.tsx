import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { toast } from "sonner";
import { ReportsSidebar, type ReportProject } from "@/components/dashboard/ReportsSidebar";
import { ReportsView, type ReportCategory, type ReportRow } from "@/components/dashboard/ReportsView";
import { deleteReport, fetchReportProjects, fetchReports, type ReportRecord } from "@/lib/reports";
import { supabase } from "@/integrations/supabase/client";
import { PlanGate } from "@/components/dashboard/PlanGate";
import { getCachedAuthUser } from "@/lib/auth-user";

export const Route = createFileRoute("/_authenticated/reports/")({
  head: () => ({
    meta: [
      { title: "Reports · Avaloka AI" },
      {
        name: "description",
        content: "Browse, organize, and share your Avaloka AI reports across projects.",
      },
    ],
  }),
  component: () => (
    <PlanGate>
      <ReportsPage />
    </PlanGate>
  ),
});

const KNOWN_CATEGORIES: ReportCategory[] = [
  "Revenue Dataset",
  "Finance Dataset",
  "Accounts Dataset",
];

function toCategory(label: string | null | undefined): ReportCategory {
  if (label && (KNOWN_CATEGORIES as string[]).includes(label)) {
    return label as ReportCategory;
  }
  return "Revenue Dataset";
}

function initialsFrom(name: string | null | undefined): string {
  if (!name) return "";
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]?.toUpperCase() ?? "")
    .join("");
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

function toRow(r: ReportRecord, currentUserId: string | null, isAdmin: boolean): ReportRow {
  const analysis = r.analyses ?? null;
  const creator = r.creator ?? null;
  const createdById = r.created_by_id ?? (typeof r.created_by === "string" ? r.created_by : r.created_by.id);
  const createdByAuthId = typeof r.created_by === "string" ? null : r.created_by.auth_user_id;
  const isSelf = !!currentUserId && (createdById === currentUserId || createdByAuthId === currentUserId);
  const creatorName = creator?.name || "Unknown";
  const creatorRole = creator?.role || "Analyst";
  const creatorInitials = creator?.initials || initialsFrom(creatorName) || "U";
  const shared = r.shared_members ?? [];
  // Always include creator alongside shared collaborators.
  const creatorMember = { id: createdById, name: creatorName, initials: creatorInitials };
  const teamMembersRaw = [creatorMember, ...shared];
  const seen = new Set<string>();
  const teamMembers = teamMembersRaw.filter((m) => {
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
  const members = teamMembers.map((m) => ({ initials: m.initials }));
  const teamCount = teamMembers.length || analysis?.team_count || 1;
  return {
    id: r.id,
    title: r.title,
    category: toCategory(r.dataset_label),
    teamCount,
    members,
    createdOn: formatDate(r.created_at),
    createdBy: {
      name: creatorName,
      role: creatorRole,
      initials: creatorInitials,
      src: creator?.avatar_url ?? undefined,
    },
    canDelete: isAdmin || isSelf,
  };
}


function ReportsPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  const { data: projectList = [] } = useQuery({
    queryKey: ["reports", "projects"],
    queryFn: fetchReportProjects,
  });

  const projects = useMemo<ReportProject[]>(
    () => projectList.map((p) => ({ id: p.id, label: p.name })),
    [projectList],
  );

  // Auto-select the first available project once the list loads.
  useEffect(() => {
    if (projects.length > 0 && (selectedProjectId === null || !projects.some((p) => p.id === selectedProjectId))) {
      setSelectedProjectId(projects[0].id);
    }
  }, [projects, selectedProjectId]);


  const projectFilter = selectedProjectId ?? null;

  const { data: records = [], isLoading: reportsLoading, isFetching: reportsFetching } = useQuery({
    queryKey: ["reports", "byProject", projectFilter ?? "__none__"],
    enabled: !!projectFilter,
    queryFn: () => fetchReports(projectFilter),
  });


  const { data: currentUserId = null } = useQuery({
    queryKey: ["auth", "user-id"],
    queryFn: async () => (await getCachedAuthUser())?.id ?? null,
  });
  const { data: isAdmin = false } = useQuery({
    queryKey: ["auth", "is-admin", currentUserId],
    enabled: !!currentUserId,
    queryFn: async () => {
      const { data } = await supabase.rpc("has_role", {
        _user_id: currentUserId as string,
        _role: "admin",
      });
      return !!data;
    },
  });

  const rows = useMemo(
    () => records.map((r: ReportRecord) => toRow(r, currentUserId, isAdmin)),
    [records, currentUserId, isAdmin],
  );
  const bookmarkedIds = useMemo(() => new Set<string>(), []);
  const toggleReportBookmark = () => {};

  const qc = useQueryClient();
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteReport(id),
    onSuccess: () => {
      toast.success("Report deleted");
      qc.invalidateQueries({ queryKey: ["reports"] });
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Could not delete report");
    },
  });

  const handleDelete = (id: string) => {
    if (!window.confirm("Delete this report?")) return;
    deleteMutation.mutate(id);
  };

  const showLoading = reportsLoading || reportsFetching;

  return (
    <div className="flex min-h-0 min-w-0 flex-1 bg-primary text-primary">
      <ReportsSidebar
        projects={projects}
        selectedId={selectedProjectId}
        onSelect={(id) => setSelectedProjectId(id)}
      />
      <main className="relative flex flex-1 flex-col overflow-hidden">
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-full bg-[radial-gradient(150%_100%_at_42.83%_20.65%,#4680e400_43.87%,#1565EF_150%,#fff)]" />
        <div className="scrollbar-hide relative flex flex-1 flex-col overflow-y-auto">
          <div className="px-6 pt-4" />

          {showLoading ? (
            <div className="flex flex-1 items-center justify-center py-16">
              <div className="flex flex-col items-center gap-3 text-sm text-tertiary">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-secondary border-t-[#1565EF]" />
                <span>Loading reports…</span>
              </div>
            </div>
          ) : (
            <ReportsView rows={rows} bookmarkedIds={bookmarkedIds} onToggleBookmark={toggleReportBookmark} onDelete={handleDelete} />
          )}
        </div>
      </main>
    </div>
  );
}

