import { SearchLg, BarChart03, LayoutAlt01 } from "@untitledui/icons";
import { Input } from "@/components/base/input/input";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { cx } from "@/lib/utils/cx";

export type ReportProject = {
  id: string;
  label: string;
};

interface ReportsSidebarProps {
  projects?: ReportProject[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}

export function ReportsSidebar({
  projects = [],
  selectedId,
  onSelect,
}: ReportsSidebarProps) {
  return (
    <aside className="flex h-full w-[300px] shrink-0 flex-col border-r border-secondary bg-primary">
      <div className="flex items-center justify-between px-4 py-4">
        <h2 className="text-lg font-semibold text-primary">Reports</h2>
        <ButtonUtility size="sm" color="tertiary" icon={LayoutAlt01} tooltip="Collapse sidebar" />
      </div>

      <div className="px-4">
        <Input aria-label="Search" placeholder="Search" size="md" icon={SearchLg} shortcut="⌘K" />
      </div>

      <div className="mt-5 flex flex-1 flex-col gap-0.5 px-2 pb-2">
        {projects.length === 0 ? (
          <div className="px-4 pt-6 text-center text-xs text-tertiary">No projects yet.</div>
        ) : (
          projects.map((p) => {
            const isActive = selectedId === p.id;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => onSelect?.(p.id)}
                className={cx(
                  "flex h-9 items-center gap-2 rounded-md px-2.5 text-left text-sm font-semibold text-secondary outline-focus-ring hover:bg-primary_hover focus-visible:outline-2",
                  isActive && "bg-secondary",
                )}
              >
                <BarChart03 className="size-4 shrink-0 text-fg-quaternary" />
                <span className="flex-1 truncate">{p.label}</span>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
