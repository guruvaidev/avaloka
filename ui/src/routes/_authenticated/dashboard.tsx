import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Plus } from "@untitledui/icons";
import { UploadDropzone } from "@/components/dashboard/UploadDropzone";
import { SecurityFooter } from "@/components/dashboard/SecurityFooter";
import { AdminDashboardShell } from "@/components/dashboard/AdminDashboardShell";
import { EditableProjectHeader } from "@/components/dashboard/EditableProjectHeader";
import { DataSourceGrid } from "@/components/dashboard/DataSourceGrid";
import { Button } from "@/components/base/buttons/button";
import type { ProjectItem } from "@/components/dashboard/ProjectsSidebar";
import { useProjects, useProjectMutations } from "@/lib/projects";
import { useAnalyses, useAnalysisMutations } from "@/lib/analyses";
import { clearPendingDashboardAnalysis, getPendingDashboardAnalysis } from "@/lib/pending-dashboard-analysis";
import { ProjectDetailsView, type AnalysisRow } from "@/components/dashboard/ProjectDetailsView";
import { ReportsModal } from "../analysis";
import { useProjectsPerms } from "@/lib/use-user-mgmt-perms";
import { useUpgradeGate } from "@/components/dashboard/UpgradeGate";


export const Route = createFileRoute("/_authenticated/dashboard")({
  head: () => ({
    meta: [
      { title: "Dashboard · Avaloka AI" },
      {
        name: "description",
        content: "Avaloka AI workspace — upload data, connect databases, and generate instant insights.",
      },
    ],
  }),
  component: DashboardPage,
});


const SAMPLE_MEMBERS = [{ initials: "AB" }, { initials: "CD" }, { initials: "EF" }, { initials: "GH" }];

function toAnalysisRows(
  list: {
    id: string;
    name: string;
    team_count: number;
    team_label: string;
    created_by_name: string | null;
    created_by_role: string | null;
    created_by_initials: string | null;
    created_by_avatar?: string | null;
    created_at: string;
  }[],
  projectName: string,
): AnalysisRow[] {
  return list.map((a) => ({
    id: a.id,
    name: a.name,
    projectName,
    teamCount: a.team_count ?? 0,
    teamLabel: a.team_label ?? "Team",
    createdOn: new Date(a.created_at).toLocaleDateString(undefined, {
      day: "2-digit",
      month: "short",
      year: "numeric",
    }),
    createdBy: {
      name: a.created_by_name ?? "—",
      role: a.created_by_role ?? "",
      initials: a.created_by_initials ?? undefined,
      src: a.created_by_avatar ?? undefined,
    },

    members: SAMPLE_MEMBERS,
  }));
}

function DashboardPage() {
  const navigate = useNavigate();
  const perms = useProjectsPerms();
  const canView = perms.canView;
  const canCreate = perms.canCreate;
  const canEdit = perms.canEdit;
  const { blocked, dialog: upgradeDialog } = useUpgradeGate();



  useEffect(() => {
    if (!perms.loading && !perms.canView) {
      navigate({ to: "/analysis" });
    }
  }, [perms.loading, perms.canView, navigate]);

  const [creatingNewProject, setCreatingNewProject] = useState(false);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [newProjectId, setNewProjectId] = useState<string | null>(null);
  const [reportsOpen, setReportsOpen] = useState(false);
  const [pendingAnalysisId, setPendingAnalysisId] = useState<string | null>(() => getPendingDashboardAnalysis());

  const { data: dbProjects } = useProjects();
  const { create, rename, updateDescription, toggleFlag, softDelete } = useProjectMutations();
  const { data: dbAnalyses } = useAnalyses(selectedProjectId);
  const analysisMut = useAnalysisMutations(selectedProjectId);

  const newProjectName = newProjectId ? (dbProjects?.find((p) => p.id === newProjectId)?.name ?? null) : null;

  const sidebarProjects: ProjectItem[] = (dbProjects ?? []).map((p) => ({
    id: p.id,
    label: p.name,
    bookmarked: p.bookmarked,
    archived: p.archived,
    pinned: p.pinned,
  }));

  const selectedProject = selectedProjectId ? sidebarProjects.find((p) => p.id === selectedProjectId) : null;
  const selectedProjectDescription = selectedProjectId
    ? (dbProjects?.find((p) => p.id === selectedProjectId)?.description ?? null)
    : null;

  const attachPendingAnalysisToProject = async (projectId: string) => {
    const pendingId = pendingAnalysisId ?? getPendingDashboardAnalysis();
    if (!pendingId) return;
    try {
      await analysisMut.moveToProject.mutateAsync({
        id: pendingId,
        project_id: projectId,
      });
      clearPendingDashboardAnalysis();
      setPendingAnalysisId(null);
    } catch (err) {
      console.error("Failed to attach pending analysis to project", err);
    }
  };

  const hasProjects = (sidebarProjects.length ?? 0) > 0;
  const selectedAnalyses = dbAnalyses ?? [];
  const selectedProjectHasAnalyses = selectedAnalyses.length > 0;

  if (!perms.loading && !canView) return null;

  return (
    <AdminDashboardShell
      projects={sidebarProjects}
      selectedProjectId={selectedProjectId}
      canCreate={canCreate}
      canEdit={canEdit}
      onNewProject={
        canCreate
          ? () => {
              if (blocked("Creating projects")) return;
              setSelectedProjectId(null);
              setNewProjectId(null);
              setCreatingNewProject(true);
            }
          : undefined
      }
      onSelectProject={async (id) => {
        await attachPendingAnalysisToProject(id);
        setSelectedProjectId(id);
        setNewProjectId(null);
        setCreatingNewProject(false);
      }}
      onRenameProject={
        canEdit
          ? (id, name) => {
              if (blocked("Renaming projects")) return;
              rename.mutate({ id, name });
            }
          : undefined
      }
      onToggleBookmark={(id, next) => {
        if (blocked("Managing projects")) return;
        toggleFlag.mutate({ id, field: "bookmarked", value: next });
      }}
      onToggleArchive={(id, next) => {
        if (blocked("Managing projects")) return;
        toggleFlag.mutate({ id, field: "archived", value: next });
        if (next && selectedProjectId === id) setSelectedProjectId(null);
        if (next && newProjectId === id) {
          setNewProjectId(null);
          setCreatingNewProject(false);
        }
      }}
      onTogglePin={(id, next) => {
        if (blocked("Managing projects")) return;
        toggleFlag.mutate({ id, field: "pinned", value: next });
      }}
      onDeleteProject={
        canEdit
          ? (id) => {
              if (blocked("Deleting projects")) return;
              softDelete.mutate(id);
              if (selectedProjectId === id) setSelectedProjectId(null);
              if (newProjectId === id) {
                setNewProjectId(null);
                setCreatingNewProject(false);
              }
            }
          : undefined
      }

    >
      {selectedProject ? (
        selectedProjectHasAnalyses ? (
          <ProjectDetailsView
            projectId={selectedProject.id}
            projectName={selectedProject.label}
            projectDescription={selectedProjectDescription}
            rows={toAnalysisRows(selectedAnalyses, selectedProject.label)}
            onRenameAnalysis={
              canEdit
                ? (id, name) => {
                    if (blocked("Renaming analyses")) return;
                    analysisMut.rename.mutate({ id, name });
                  }
                : undefined
            }
            onDeleteAnalysis={
              canEdit
                ? (id) => {
                    if (blocked("Deleting analyses")) return;
                    analysisMut.softDelete.mutate(id);
                  }
                : undefined
            }

          />
        ) : (
          <div className="flex min-h-screen flex-1 flex-col">
            <div className="px-8 pt-8">
              <div className="mx-auto w-full max-w-[760px]">
                <EditableProjectHeader
                  projectId={selectedProject.id}
                  defaultName={selectedProject.label}
                  defaultDescription={selectedProjectDescription ?? ""}
                  onNameCommit={canEdit ? (name) => rename.mutate({ id: selectedProject.id, name }) : undefined}
                  onDescriptionCommit={
                    canEdit
                      ? (description) => updateDescription.mutate({ id: selectedProject.id, description })
                      : undefined
                  }
                  readOnly={!canEdit}
                />
              </div>
            </div>
            {canCreate && (
              <div className="flex flex-1 flex-col items-center justify-center px-8 py-10">
                <div className="mx-auto w-full max-w-[760px]">
                  <section>
                    <UploadDropzone
                      projectId={selectedProject.id}
                      projectName={selectedProject.label}
                    />
                  </section>
                  <div className="my-8 flex items-center gap-4">
                    <div className="h-px flex-1 bg-border-secondary" />
                    <span className="my-8 text-sm text-tertiary">Or</span>
                    <div className="h-px flex-1 bg-border-secondary" />
                  </div>
                  <section>
                    <DataSourceGrid />
                  </section>
                </div>
              </div>
            )}
            <div className="mt-auto">
              <SecurityFooter />
            </div>
          </div>
        )
      ) : (
        <LockedProjectWorkspace
          hasProjects={hasProjects}
          pendingAnalysisNotice={!!pendingAnalysisId}
          canCreate={canCreate}
          onCreateProject={(name, description) => {
            if (blocked("Creating projects")) return;
            create.mutate({ name, description }, {
              onSuccess: async (p) => {
                if (pendingAnalysisId) {
                  await attachPendingAnalysisToProject(p.id);
                }
                setNewProjectId(p.id);
                setCreatingNewProject(false);
                setSelectedProjectId(p.id);
              },
            });
          }}
        />
      )}

      <ReportsModal open={reportsOpen} onOpenChange={setReportsOpen} />
      {upgradeDialog}

    </AdminDashboardShell>
  );
}

function LockedProjectWorkspace({
  hasProjects,
  pendingAnalysisNotice,
  canCreate,
  onCreateProject,
}: {
  hasProjects: boolean;
  pendingAnalysisNotice: boolean;
  canCreate: boolean;
  onCreateProject: (name: string, description: string | null) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const canSave = canCreate && name.trim().length > 0;

  const commit = () => {
    if (!canSave) return;
    onCreateProject(name.trim(), description.trim() || null);
  };

  return (
    <div className="flex min-h-screen flex-1 flex-col">
      <div className="px-8 pt-8">
        <div className="mx-auto w-full max-w-[760px]">
          <div className="flex w-full items-start justify-between gap-4 border-b border-secondary px-2 pb-5">
            <div className="flex-1 space-y-1">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && commit()}
                disabled={!canCreate}
                placeholder="Add Project Name"
                className="w-full bg-transparent text-md font-semibold text-primary outline-none placeholder:text-primary disabled:cursor-not-allowed"
              />
              <input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && commit()}
                disabled={!canCreate}
                placeholder="Add Description"
                className="w-full bg-transparent text-sm text-tertiary outline-none placeholder:text-tertiary disabled:cursor-not-allowed"
              />
            </div>
            {canCreate && (
              <button
                type="button"
                onClick={commit}
                disabled={!canSave}
                aria-label="Create project"
                title="Create project"
                className="grid size-10 shrink-0 place-items-center rounded-full border border-secondary bg-primary text-tertiary transition hover:border-brand hover:text-brand-secondary disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Plus className="size-5" />
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center px-8 py-10">
        <div className="mx-auto w-full max-w-[760px]">
          {pendingAnalysisNotice && (
            <div className="mb-6 rounded-xl border border-[#b2ceff] bg-[#eff5ff] px-4 py-3 text-sm text-[#175cd3]">
              Your analysis is ready. Create a new project or select an existing one to save it.
            </div>
          )}

          <div className="mb-4 text-center text-sm text-tertiary">
            {hasProjects
              ? "Select a project from the sidebar, or name a new one above to upload data."
              : "Name your project above to enable uploads and data source connections."}
          </div>


          <section>
            <UploadDropzone disabled />
          </section>
          <div className="my-8 flex items-center gap-4">
            <div className="h-px flex-1 bg-border-secondary" />
            <span className="text-sm text-tertiary">Or</span>
            <div className="h-px flex-1 bg-border-secondary" />
          </div>
          <section>
            <DataSourceGrid disabled />
          </section>
        </div>
      </div>

      <div className="mt-auto">
        <SecurityFooter />
      </div>
    </div>
  );
}

