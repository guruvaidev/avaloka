import { useEffect, useMemo, useRef, useState } from "react";
import {
  SearchLg,
  Bookmark,
  Archive,
  MessageCircle01,
  Calendar,
  Plus,
  LayoutAlt01,
  BookmarkCheck,
  Trash01,
  Edit01,
  Pin01,
  Check,
  X,
  ChevronRight,
  File02,
  RefreshCcw01,

} from "@untitledui/icons";
import { Button as AriaButton } from "react-aria-components";
import { useNavigate } from "@tanstack/react-router";
import { Input } from "@/components/base/input/input";
import { Button } from "@/components/base/buttons/button";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { Dropdown } from "@/components/base/dropdown/dropdown";
import { Tooltip } from "@/components/base/tooltip/tooltip";
import { cx } from "@/lib/utils/cx";
const noActivity = { url: "/assets/dashboard/no_activity.png" };
import { ScheduleAnalysisModal } from "./ScheduleAnalysisModal";
import { DeleteAnalysisModal, DeletedToastModal } from "./DeleteAnalysisModal";
import { UploadModal } from "./UploadModal";
import { useStandaloneAnalyses, useAnalysisMutations } from "@/lib/analyses";
import { useQueryClient } from "@tanstack/react-query";
import { useAnalysesDatasets, analysisDatasetLabel, analysisGroupName, type AnalysisDataset } from "@/lib/analysis-datasets";

const chips = [
  { id: "all", label: "All", icon: null },
  { id: "bookmark", label: "Bookmark", icon: Bookmark },
  { id: "archive", label: "Archive", icon: Archive },
] as const;

export type AnalysisItem = {
  id: string;
  label: string;
  is_bookmarked?: boolean;
  is_archived?: boolean;
  is_pinned?: boolean;
};

interface AnalysisSidebarProps {
  analyses?: AnalysisItem[];
}

type ChipId = (typeof chips)[number]["id"];

export function AnalysisSidebar({ analyses: _analysesProp }: AnalysisSidebarProps = {}) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { data: dbAnalyses, refetch: refetchAnalyses, isFetching: isFetchingAnalyses } = useStandaloneAnalyses();
  const { rename: renameMutation, softDelete: deleteMutation, setFlag } = useAnalysisMutations(null);
  const analyses = useMemo<AnalysisItem[]>(
    () =>
      (dbAnalyses ?? []).map((a) => ({
        id: a.id,
        label: a.name,
        is_bookmarked: a.is_bookmarked,
        is_archived: a.is_archived,
        is_pinned: a.is_pinned,
      })),
    [dbAnalyses],
  );
  const [activeChip, setActiveChip] = useState<ChipId>("all");
  const [collapsed, setCollapsed] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [deleted, setDeleted] = useState<Set<string>>(new Set());
  const [scheduleOpen, setScheduleOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);
  const [deletedToastLabel, setDeletedToastLabel] = useState<string | null>(null);
  const [labels, setLabels] = useState<Record<string, string>>({});
  useEffect(() => {
    setLabels((prev) => {
      const next = { ...prev };
      for (const a of analyses) if (next[a.id] === undefined) next[a.id] = a.label;
      return next;
    });
  }, [analyses]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  const startRename = (id: string) => {
    setRenameValue(labels[id] ?? "");
    setRenamingId(id);
  };

  const commitRename = () => {
    if (!renamingId) return;
    const val = renameValue.trim();
    if (val) {
      const id = renamingId;
      setLabels((prev) => ({ ...prev, [id]: val }));
      renameMutation.mutate(
        { id, name: val },
        { onSuccess: () => qc.invalidateQueries({ queryKey: ["analyses"] }) },
      );
    }
    setRenamingId(null);
  };

  const cancelRename = () => setRenamingId(null);

  const flags = useMemo(() => {
    const m = new Map<string, { b: boolean; a: boolean; p: boolean }>();
    for (const it of analyses) {
      m.set(it.id, {
        b: !!it.is_bookmarked,
        a: !!it.is_archived,
        p: !!it.is_pinned,
      });
    }
    return m;
  }, [analyses]);

  const visible = useMemo(() => {
    let list: AnalysisItem[];
    const notDeleted = analyses.filter((a) => !deleted.has(a.id));
    if (activeChip === "bookmark") list = notDeleted.filter((a) => flags.get(a.id)?.b);
    else if (activeChip === "archive") list = notDeleted.filter((a) => flags.get(a.id)?.a);
    else list = notDeleted.filter((a) => !flags.get(a.id)?.a);
    const q = searchQuery.trim().toLowerCase();
    if (q) {
      list = list.filter((a) => (labels[a.id] ?? a.label).toLowerCase().includes(q));
    }
    return [...list].sort((a, b) => {
      const ap = flags.get(a.id)?.p ? 1 : 0;
      const bp = flags.get(b.id)?.p ? 1 : 0;
      return bp - ap;
    });
  }, [analyses, activeChip, flags, deleted, searchQuery, labels]);

  const visibleIds = useMemo(() => visible.map((a) => a.id), [visible]);
  const { byAnalysis, isLoading: datasetsLoading } = useAnalysesDatasets(visibleIds);
  const datasetsFor = (id: string): AnalysisDataset[] => byAnalysis[id] ?? [];

  const toggleBookmark = (id: string) => {

    const cur = !!flags.get(id)?.b;
    setFlag.mutate({ id, field: "is_bookmarked", value: !cur });
  };

  const toggleArchive = (id: string) => {
    const cur = !!flags.get(id)?.a;
    setFlag.mutate({ id, field: "is_archived", value: !cur });
  };

  const togglePin = (id: string) => {
    const cur = !!flags.get(id)?.p;
    setFlag.mutate({ id, field: "is_pinned", value: !cur });
  };

  return (
    <>
      {collapsed && (
        <div className="flex h-full shrink-0 items-start p-2">
          <ButtonUtility
            size="sm"
            color="tertiary"
            icon={LayoutAlt01}
            tooltip="Expand sidebar"
            onClick={() => setCollapsed(false)}
          />
        </div>
      )}
      <aside
        className={cx(
          "flex h-full shrink-0 flex-col border-r border-secondary bg-primary",
          collapsed && "hidden",
        )}
        style={collapsed ? undefined : { width: 300 }}
      >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-4">
        <h2 className="text-lg font-semibold text-primary">Analysis</h2>
        <ButtonUtility
          size="sm"
          color="tertiary"
          icon={LayoutAlt01}
          tooltip="Collapse sidebar"
          onClick={() => setCollapsed(true)}
        />
      </div>

      {/* Search */}
      <div className="px-4">
        <Input
          aria-label="Search"
          placeholder="Search"
          size="md"
          icon={SearchLg}
          shortcut="⌘K"
          value={searchQuery}
          onChange={setSearchQuery}
        />
      </div>

      {/* Filter chips */}
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

      {/* Sections */}
      <div className="mt-4 flex flex-col gap-0.5 px-2">
        <div className="group relative flex items-center">
          <Button
            color="tertiary"
            size="md"
            iconLeading={MessageCircle01}
            onClick={() => setUploadOpen(true)}
            className="w-full justify-start *:data-text:flex-1 *:data-text:text-left"
          >
            New Analysis
          </Button>
          <ButtonUtility
            size="xs"
            color="tertiary"
            icon={Plus}
            tooltip="New analysis"
            onClick={() => setUploadOpen(true)}
            className="absolute right-2 top-1/2 -translate-y-1/2"
          />
        </div>
        <div className="group relative flex items-center">
          <Button
            color="tertiary"
            size="md"
            iconLeading={Calendar}
            onClick={() => navigate({ to: "/scheduled-analysis" })}
            className="w-full justify-start *:data-text:flex-1 *:data-text:text-left"
          >
            Schedule analysis
          </Button>
          <ButtonUtility
            size="xs"
            color="tertiary"
            icon={Plus}
            tooltip="New schedule"
            onClick={() => setScheduleOpen(true)}
            className="absolute right-2 top-1/2 -translate-y-1/2"
          />
        </div>
      </div>

      <div className="my-4 mx-3 border-b bg-border-secondary"></div>

      {/* Recent / Analysis list */}
      <div className="mt-4 flex flex-1 min-h-0 flex-col pb-2">
        {(() => {
          const sectionTitle =
            activeChip === "bookmark" ? "Bookmarks" : activeChip === "archive" ? "Archived" : "Recent";
          const refreshButton = (
            <Tooltip title="Refresh" placement="right">
              <ButtonUtility
                size="xs"
                color="tertiary"
                icon={RefreshCcw01}
                aria-label="Refresh recent analyses"
                className={cx(isFetchingAnalyses && "animate-spin")}
                onClick={() => {
                  void qc.invalidateQueries({ queryKey: ["analyses"] });
                  void refetchAnalyses();
                }}
              />
            </Tooltip>
          );
          return visible.length === 0 ? (
          <div className="px-2">
            <div className="flex items-center justify-between pr-2">
              <p className="px-2 text-xs font-medium text-tertiary">{sectionTitle}</p>
              {refreshButton}
            </div>
            <RecentEmptyState activeChip={activeChip} />
          </div>
        ) : (
          <div className="flex flex-col gap-0.5 min-h-0 flex-1">
            <div className="flex items-center justify-between px-4 pb-1 pr-2">
              <p className="text-xs font-medium text-tertiary">{sectionTitle}</p>
              {refreshButton}
            </div>
            <div className="flex flex-col gap-0.5 overflow-y-auto pr-1">
            {visible.map((a) => {
              const f = flags.get(a.id);
              const isBookmarked = !!f?.b;
              const isArchived = !!f?.a;
              const isPinned = !!f?.p;
              const isRenaming = renamingId === a.id;
              const label = analysisGroupName(labels[a.id] ?? a.label, datasetsFor(a.id).length);
              const isExpanded = expandedId === a.id;
              return (
                <div key={a.id}>
                <div className="group relative flex items-center px-2">
                  {!isRenaming && (
                    <button
                      type="button"
                      aria-label={isExpanded ? `Hide versions for ${label}` : `Show versions for ${label}`}
                      onClick={() => setExpandedId(isExpanded ? null : a.id)}
                      className="mr-0.5 rounded-md p-1 text-fg-quaternary hover:bg-primary_hover hover:text-fg-quaternary_hover"
                    >
                      <ChevronRight className={cx("size-3.5 transition-transform", isExpanded && "rotate-90")} />
                    </button>
                  )}

                  {isRenaming ? (
                    <div className="flex h-9 flex-1 items-center gap-1 rounded-md px-2.5 ring-1 ring-inset ring-primary bg-primary">
                      {activeChip === "bookmark" ? (
                        <BookmarkCheck className="size-4 shrink-0 text-fg-brand-primary" />
                      ) : (
                        <MessageCircle01 className="size-4 shrink-0 text-fg-quaternary" />
                      )}
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
                        onClick={() => navigate({ to: "/analysis", search: { aid: a.id } as any })}
                        className={cx(
                          "flex h-9 min-w-0 flex-1 items-center gap-2 rounded-md px-2.5 pr-9 text-left text-sm font-semibold text-secondary outline-focus-ring hover:bg-primary_hover focus-visible:outline-2",
                          isPinned && "bg-secondary",
                        )}
                      >
                        {activeChip === "bookmark" ? (
                          <BookmarkCheck className="size-4 shrink-0 text-fg-brand-primary" />
                        ) : (
                          <MessageCircle01 className="size-4 shrink-0 text-fg-quaternary" />
                        )}
                        <span className="min-w-0 flex-1 overflow-hidden">
                          <Tooltip title={label} placement="top" delay={150}>
                            <span className="block truncate">{label}</span>
                          </Tooltip>
                        </span>
                        {datasetsFor(a.id).length > 1 ? (
                          <span className="shrink-0 rounded-full bg-secondary px-1.5 text-[10px] font-medium text-tertiary ring-1 ring-inset ring-secondary">
                            {datasetsFor(a.id).length} files
                          </span>
                        ) : null}
                      </button>

                      <Dropdown.Root>
                        {isPinned ? (
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
                              <Dropdown.Item icon={Edit01} onAction={() => startRename(a.id)}>
                                Rename
                              </Dropdown.Item>
                              <Dropdown.Item icon={Archive} onAction={() => toggleArchive(a.id)}>
                                {isArchived ? "Unarchive" : "Archive"}
                              </Dropdown.Item>
                              <Dropdown.Item
                                icon={isBookmarked ? BookmarkCheck : Bookmark}
                                onAction={() => toggleBookmark(a.id)}
                              >
                                {isBookmarked ? "Remove Bookmark" : "Bookmark"}
                              </Dropdown.Item>
                              <Dropdown.Item icon={Pin01} onAction={() => togglePin(a.id)}>
                                {isPinned ? "Unpin" : "Pin"}
                              </Dropdown.Item>
                            </Dropdown.Section>
                            <Dropdown.Separator />
                            <Dropdown.Section>
                              <Dropdown.Item icon={Trash01} onAction={() => setDeleteTargetId(a.id)}>
                                Delete
                              </Dropdown.Item>
                            </Dropdown.Section>
                          </Dropdown.Menu>
                        </Dropdown.Popover>
                      </Dropdown.Root>

                    </>
                  )}
                </div>
                {isExpanded && !isRenaming ? (
                  <div className="px-2 pb-1">
                    <AnalysisDatasetList
                      datasets={datasetsFor(a.id)}
                      loading={datasetsLoading}
                      onSelect={(datasetId: string) =>
                        navigate({ to: "/analysis", search: { aid: a.id, ds: datasetId } as any })
                      }

                    />
                  </div>
                ) : null}

                </div>
              );

            })}
            </div>
          </div>
          );
        })()}
      </div>

      <ScheduleAnalysisModal open={scheduleOpen} onOpenChange={setScheduleOpen} />
      <UploadModal open={uploadOpen} onOpenChange={setUploadOpen} />
      <DeleteAnalysisModal
        open={deleteTargetId !== null}
        onOpenChange={(o) => !o && setDeleteTargetId(null)}
        onConfirm={() => {
          if (!deleteTargetId) return;
          const id = deleteTargetId;
          const lbl = labels[id] ?? analyses.find((a) => a.id === id)?.label ?? "";
          setDeleted((prev) => new Set(prev).add(id));
          deleteMutation.mutate(id, {
            onSuccess: () => qc.invalidateQueries({ queryKey: ["analyses"] }),
          });
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
    </>
  );
}

function AnalysisDatasetList({
  datasets,
  loading,
  onSelect,
}: {
  datasets: AnalysisDataset[];
  loading?: boolean;
  onSelect: (datasetId: string) => void;
}) {
  return (
    <div className="ml-6 flex flex-col gap-0.5 border-l border-secondary pl-2">
      {loading && datasets.length === 0 ? (
        <div className="flex flex-col gap-1 py-1">
          {[0, 1].map((i) => (
            <div key={i} className="h-7 animate-pulse rounded-md bg-secondary" />
          ))}
        </div>
      ) : datasets.length === 0 ? (
        <p className="px-2 py-2 text-xs text-tertiary">No files in this analysis.</p>
      ) : (
        datasets.map((d) => {
          const label = analysisDatasetLabel(d);
          return (
            <button
              key={d.dataset_id}
              type="button"
              onClick={() => onSelect(d.dataset_id)}
              className="flex min-h-7 items-center gap-1.5 rounded-md px-2 py-1.5 text-left hover:bg-primary_hover"
            >
              <File02 className="size-3.5 shrink-0 text-fg-quaternary" />
              <Tooltip title={label} placement="top" delay={150}>
                <span className="min-w-0 flex-1 truncate text-xs font-medium text-secondary">{label}</span>
              </Tooltip>
            </button>
          );
        })
      )}
    </div>
  );
}


function RecentEmptyState({ activeChip }: { activeChip: ChipId }) {
  const copy =
    activeChip === "bookmark"
      ? { title: "No bookmarks yet.", sub: "Add analyses to the Bookmark tab from the menu." }
      : activeChip === "archive"
      ? { title: "Nothing archived.", sub: "Archived analyses appear here." }
      : { title: "No recent activity.", sub: "Upload data and begin your analysis." };

  return (
    <div className="mt-6 flex flex-col items-center text-center">
      <img src={noActivity.url} alt="" aria-hidden className="h-auto w-auto max-w-[140px] select-none object-contain" />
      <p className="mt-3 text-sm text-primary">{copy.title}</p>
      <p className="text-sm text-tertiary">{copy.sub}</p>
    </div>
  );
}
