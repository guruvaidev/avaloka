import { useEffect, useMemo, useRef, useState } from "react";
import {
  SearchLg,
  Bookmark,
  Archive,
  BarChart03,
  Plus,
  LayoutAlt01,
  BookmarkCheck,
  Trash01,
  Edit01,
  Pin01,
  Check,
  X,
} from "@untitledui/icons";
import { Button as AriaButton } from "react-aria-components";
import { Input } from "@/components/base/input/input";
import { Button } from "@/components/base/buttons/button";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { Dropdown } from "@/components/base/dropdown/dropdown";
import { Tooltip } from "@/components/base/tooltip/tooltip";
import { cx } from "@/lib/utils/cx";
const noActivity = { url: "/assets/dashboard/no_activity.png" };
import { DeleteAnalysisModal, DeletedToastModal } from "./DeleteAnalysisModal";

const chips = [
  { id: "all", label: "All", icon: null },
  { id: "bookmark", label: "Bookmark", icon: Bookmark },
  { id: "archive", label: "Archive", icon: Archive },
] as const;

export type ProjectItem = {
  id: string;
  label: string;
  bookmarked?: boolean;
  archived?: boolean;
  pinned?: boolean;
};
type ChipId = (typeof chips)[number]["id"];

interface ProjectsSidebarProps {
  projects?: ProjectItem[];
  onNewProject?: () => void;
  onSelectProject?: (id: string) => void;
  onRenameProject?: (id: string, name: string) => void;
  onToggleBookmark?: (id: string, next: boolean) => void;
  onToggleArchive?: (id: string, next: boolean) => void;
  onTogglePin?: (id: string, next: boolean) => void;
  onDeleteProject?: (id: string) => void;
  onCollapse?: () => void;
  canCreate?: boolean;
  canEdit?: boolean;
}

export function ProjectsSidebar({
  projects = [],
  onNewProject,
  onSelectProject,
  onRenameProject,
  onToggleBookmark,
  onToggleArchive,
  onTogglePin,
  onDeleteProject,
  onCollapse,
  canCreate = true,
  canEdit = true,
}: ProjectsSidebarProps) {
  const [activeChip, setActiveChip] = useState<ChipId>("all");
  const [search, setSearch] = useState("");
  // Local fallback state when no server callbacks are provided
  const [bookmarkedLocal, setBookmarkedLocal] = useState<Set<string>>(new Set());
  const [archivedLocal, setArchivedLocal] = useState<Set<string>>(new Set());
  const [pinnedLocal, setPinnedLocal] = useState<Set<string>>(new Set());
  const [deletedLocal, setDeletedLocal] = useState<Set<string>>(new Set());
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);
  const [deletedToastLabel, setDeletedToastLabel] = useState<string | null>(null);
  const [labelsLocal, setLabelsLocal] = useState<Record<string, string>>({});
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  const isBookmarked = (p: ProjectItem) =>
    p.bookmarked ?? bookmarkedLocal.has(p.id);
  const isArchived = (p: ProjectItem) => p.archived ?? archivedLocal.has(p.id);
  const isPinned = (p: ProjectItem) => p.pinned ?? pinnedLocal.has(p.id);
  const labelOf = (p: ProjectItem) => labelsLocal[p.id] ?? p.label;

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  const startRename = (id: string) => {
    const p = projects.find((x) => x.id === id);
    setRenameValue(labelsLocal[id] ?? p?.label ?? "");
    setRenamingId(id);
  };
  const commitRename = () => {
    if (!renamingId) return;
    const val = renameValue.trim();
    if (val) {
      if (onRenameProject) onRenameProject(renamingId, val);
      else setLabelsLocal((prev) => ({ ...prev, [renamingId!]: val }));
    }
    setRenamingId(null);
  };
  const cancelRename = () => setRenamingId(null);

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    const notDeleted = projects.filter((p) => !deletedLocal.has(p.id));
    let list: ProjectItem[];
    if (activeChip === "bookmark") list = notDeleted.filter(isBookmarked);
    else if (activeChip === "archive") list = notDeleted.filter(isArchived);
    else list = notDeleted.filter((p) => !isArchived(p));
    if (q) list = list.filter((p) => labelOf(p).toLowerCase().includes(q));
    return [...list].sort(
      (a, b) => (isPinned(b) ? 1 : 0) - (isPinned(a) ? 1 : 0),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projects, activeChip, search, bookmarkedLocal, archivedLocal, pinnedLocal, deletedLocal, labelsLocal]);

  const toggleSet = (setter: React.Dispatch<React.SetStateAction<Set<string>>>, id: string) =>
    setter((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const handleToggleBookmark = (p: ProjectItem) => {
    const next = !isBookmarked(p);
    if (onToggleBookmark) onToggleBookmark(p.id, next);
    else toggleSet(setBookmarkedLocal, p.id);
  };
  const handleToggleArchive = (p: ProjectItem) => {
    const next = !isArchived(p);
    if (onToggleArchive) onToggleArchive(p.id, next);
    else toggleSet(setArchivedLocal, p.id);
  };
  const handleTogglePin = (p: ProjectItem) => {
    const next = !isPinned(p);
    if (onTogglePin) onTogglePin(p.id, next);
    else toggleSet(setPinnedLocal, p.id);
  };



  return (
    <aside className="flex h-full w-[300px] shrink-0 flex-col border-r border-secondary bg-primary">
      <div className="flex items-center justify-between px-4 py-4">
        <h2 className="text-lg font-semibold text-primary">Projects</h2>
        <ButtonUtility size="sm" color="tertiary" icon={LayoutAlt01} tooltip="Collapse sidebar" onClick={onCollapse} />
      </div>

      <div className="px-4">
        <Input
          aria-label="Search"
          placeholder="Search"
          size="md"
          icon={SearchLg}
          shortcut="⌘K"
          value={search}
          onChange={setSearch}
        />
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5 px-4">
        {chips.map((c) => {
          const Icon = c.icon;
          const isActive = activeChip === c.id;
          return (
            <button
              key={c.id}
              type="button"
              onClick={() => setActiveChip(c.id)}
              className={cx(
                "inline-flex h-7 items-center gap-1 rounded-full px-3 text-xs font-medium ring-1 ring-inset transition-colors",
                isActive
                  ? "bg-brand-primary_alt text-brand-secondary ring-brand"
                  : "bg-primary text-tertiary ring-primary hover:bg-primary_hover",
              )}
            >
              {Icon ? <Icon className="size-3 stroke-[2.25px]" /> : null}
              {c.label}
            </button>
          );
        })}
      </div>

      {canCreate && (
        <div className="mt-4 flex flex-col gap-0.5 px-2">
          <div className="group relative flex items-center">
            <Button
              color="tertiary"
              size="md"
              iconLeading={BarChart03}
              onClick={onNewProject}
              className="w-full justify-start *:data-text:flex-1 *:data-text:text-left"
            >
              New Project
            </Button>
            <ButtonUtility
              size="xs"
              color="tertiary"
              icon={Plus}
              tooltip="New project"
              onClick={onNewProject}
              className="absolute right-2 top-1/2 -translate-y-1/2"
            />
          </div>
        </div>
      )}

      <div className="my-4 mx-3 border-b bg-border-secondary" />


      <div className="mt-2 flex min-h-0 flex-1 flex-col pb-2">
        <p className="px-4 pb-1 text-xs font-medium text-tertiary">
          {activeChip === "bookmark" ? "Bookmarks" : activeChip === "archive" ? "Archived" : "Recent"}
        </p>
        {visible.length === 0 ? (
          <RecentEmptyState activeChip={activeChip} />
        ) : (
          <div className="flex flex-1 flex-col gap-0.5 overflow-y-auto">

            {visible.map((p) => {
              const bookmarked = isBookmarked(p);
              const archived = isArchived(p);
              const pinned = isPinned(p);
              const isRenaming = renamingId === p.id;
              const label = labelOf(p);
              return (
                <div key={p.id} className="group relative flex items-center px-2">
                  {isRenaming ? (
                    <div className="flex h-9 flex-1 items-center gap-1 rounded-md px-2.5 ring-1 ring-inset ring-primary bg-primary">
                      <BarChart03 className="size-4 shrink-0 text-fg-quaternary" />
                      <input
                        ref={renameInputRef}
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitRename();
                          if (e.key === "Escape") cancelRename();
                        }}
                        className="min-w-0 flex-1 bg-transparent text-sm font-semibold text-secondary outline-none"
                      />
                      <ButtonUtility size="xs" color="tertiary" icon={Check} tooltip="Save" onClick={commitRename} />
                      <ButtonUtility size="xs" color="tertiary" icon={X} tooltip="Cancel" onClick={cancelRename} />
                    </div>
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => onSelectProject?.(p.id)}
                        className={cx(
                          "flex h-9 min-w-0 flex-1 items-center gap-2 rounded-md px-2.5 pr-9 text-left text-sm font-semibold text-secondary outline-focus-ring hover:bg-primary_hover focus-visible:outline-2",
                          pinned && "bg-secondary",
                        )}
                      >
                        {activeChip === "bookmark" ? (
                          <BookmarkCheck className="size-4 shrink-0 text-fg-brand-primary" />
                        ) : (
                          <BarChart03 className="size-4 shrink-0 text-fg-quaternary" />
                        )}
                        <span className="min-w-0 flex-1 overflow-hidden">
                          <Tooltip title={label} placement="top" delay={150}>
                            <span className="block truncate">{label}</span>
                          </Tooltip>
                        </span>
                      </button>
                      <Dropdown.Root>
                        {pinned ? (
                          <AriaButton
                            aria-label={`More options for ${label}`}
                            className="absolute right-3 top-1/2 -translate-y-1/2 cursor-pointer rounded-md p-1 text-fg-quaternary outline-focus-ring hover:text-fg-quaternary_hover"
                          >
                            <Pin01 className="size-4" />
                          </AriaButton>
                        ) : activeChip === "bookmark" ? (
                          <AriaButton
                            aria-label={`More options for ${label}`}
                            className="absolute right-3 top-1/2 -translate-y-1/2 cursor-pointer rounded-md p-1 text-fg-quaternary outline-focus-ring hover:text-fg-quaternary_hover"
                          >
                            <Bookmark className="size-4" />
                          </AriaButton>
                        ) : activeChip === "archive" ? (
                          <AriaButton
                            aria-label={`More options for ${label}`}
                            className="absolute right-3 top-1/2 -translate-y-1/2 cursor-pointer rounded-md p-1 text-fg-quaternary outline-focus-ring hover:text-fg-quaternary_hover"
                          >
                            <Archive className="size-4" />
                          </AriaButton>
                        ) : (
                          <Dropdown.DotsButton
                            aria-label={`More options for ${label}`}
                            className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1"
                          />
                        )}
                        <Dropdown.Popover placement="right top" offset={8} className="w-56">
                          <Dropdown.Menu>
                            <Dropdown.Section>
                              {canEdit && (
                                <Dropdown.Item icon={Edit01} onAction={() => startRename(p.id)}>
                                  Rename
                                </Dropdown.Item>
                              )}
                              <Dropdown.Item icon={Archive} onAction={() => handleToggleArchive(p)}>
                                {archived ? "Unarchive" : "Archive"}
                              </Dropdown.Item>
                              <Dropdown.Item
                                icon={bookmarked ? BookmarkCheck : Bookmark}
                                onAction={() => handleToggleBookmark(p)}
                              >
                                {bookmarked ? "Remove Bookmark" : "Bookmark"}
                              </Dropdown.Item>
                              <Dropdown.Item icon={Pin01} onAction={() => handleTogglePin(p)}>
                                {pinned ? "Unpin" : "Pin"}
                              </Dropdown.Item>
                            </Dropdown.Section>
                            {canEdit && (
                              <>
                                <Dropdown.Separator />
                                <Dropdown.Section>
                                  <Dropdown.Item icon={Trash01} onAction={() => setDeleteTargetId(p.id)}>
                                    Delete
                                  </Dropdown.Item>
                                </Dropdown.Section>
                              </>
                            )}
                          </Dropdown.Menu>
                        </Dropdown.Popover>
                      </Dropdown.Root>
                    </>
                  )}
                </div>
              );
            })}

          </div>
        )}
      </div>

      <DeleteAnalysisModal
        open={deleteTargetId !== null}
        onOpenChange={(o) => !o && setDeleteTargetId(null)}
        onConfirm={() => {
          if (!deleteTargetId) return;
          const proj = projects.find((p) => p.id === deleteTargetId);
          const lbl = labelsLocal[deleteTargetId] ?? proj?.label ?? "";
          if (onDeleteProject) onDeleteProject(deleteTargetId);
          else setDeletedLocal((prev) => new Set(prev).add(deleteTargetId));
          setDeleteTargetId(null);
          setDeletedToastLabel(lbl);
        }}
      />
      <DeletedToastModal
        open={deletedToastLabel !== null}
        onOpenChange={(o) => !o && setDeletedToastLabel(null)}
        label={deletedToastLabel ?? ""}
      />
    </aside>
  );
}

function RecentEmptyState({ activeChip }: { activeChip: ChipId }) {
  const copy =
    activeChip === "bookmark"
      ? { title: "No bookmarks yet.", sub: "Add projects to the Bookmark tab from the menu." }
      : activeChip === "archive"
      ? { title: "Nothing archived.", sub: "Archived projects appear here." }
      : { title: "No recent Projects.", sub: "Create one and begin your analysis." };

  return (
    <div className="mt-6 flex flex-col items-center px-4 text-center">
      <img src={noActivity.url} alt="" aria-hidden className="h-auto w-auto max-w-[140px] select-none object-contain" />
      <p className="mt-3 text-sm text-primary">{copy.title}</p>
      <p className="text-sm text-tertiary">{copy.sub}</p>
    </div>
  );
}
