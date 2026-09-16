import { useState, type ReactNode } from "react";
import { PanelLeftOpen } from "lucide-react";

import { ProjectsSidebar, type ProjectItem } from "./ProjectsSidebar";

export function AdminDashboardShell({
  children,
  projects,
  selectedProjectId,
  onNewProject,
  onSelectProject,
  onRenameProject,
  onToggleBookmark,
  onToggleArchive,
  onTogglePin,
  onDeleteProject,
  canCreate = true,
  canEdit = true,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onRailItemClick: _onRailItemClick,
}: {
  children: ReactNode;
  projects?: ProjectItem[];
  selectedProjectId?: string | null;
  onNewProject?: () => void;
  onSelectProject?: (id: string) => void;
  onRenameProject?: (id: string, name: string) => void;
  onToggleBookmark?: (id: string, next: boolean) => void;
  onToggleArchive?: (id: string, next: boolean) => void;
  onTogglePin?: (id: string, next: boolean) => void;
  onDeleteProject?: (id: string) => void;
  canCreate?: boolean;
  canEdit?: boolean;
  /** Deprecated: the icon rail now lives in the authenticated layout. */
  onRailItemClick?: (id: string) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 bg-primary text-primary">
      {!collapsed && (
        <ProjectsSidebar
          projects={projects}
          onNewProject={onNewProject}
          onSelectProject={onSelectProject}
          onRenameProject={onRenameProject}
          onToggleBookmark={onToggleBookmark}
          onToggleArchive={onToggleArchive}
          onTogglePin={onTogglePin}
          onDeleteProject={onDeleteProject}
          onCollapse={() => setCollapsed(true)}
          canCreate={canCreate}
          canEdit={canEdit}
        />
      )}
      <main className="relative flex flex-1 flex-col overflow-hidden">
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-full bg-[radial-gradient(150%_100%_at_42.83%_20.65%,#4680e400_43.87%,#1565EF_150%,#fff)]" />
        <div className="scrollbar-hide relative flex flex-1 flex-col overflow-y-auto">
          <div className="flex items-center justify-between px-6 pt-4">
            {collapsed ? (
              <button
                type="button"
                onClick={() => setCollapsed(false)}
                aria-label="Open sidebar"
                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-background/80 text-foreground backdrop-blur-sm transition-colors hover:bg-accent"
              >
                <PanelLeftOpen className="h-4 w-4" />
              </button>
            ) : (
              <span />
            )}
          </div>

          {children}
        </div>
      </main>
    </div>
  );
}
