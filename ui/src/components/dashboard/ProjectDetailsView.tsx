import { useMemo, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import {
  SearchLg,
  Calendar,
  ChevronDown,

  Grid01,
  Rows01,
  File02,
  Edit01,
  Archive,
  Pin01,
  Trash01,
  ArrowLeft,
  ArrowRight,
  Monitor01,
  Plus,
} from "@untitledui/icons";
import { Input } from "@/components/base/input/input";
import { Button } from "@/components/base/buttons/button";
import { Avatar } from "@/components/base/avatar/avatar";
import { Dropdown } from "@/components/base/dropdown/dropdown";
import { cx } from "@/lib/utils/cx";
import { InviteCollaboratorsTrigger } from "./InviteCollaboratorsTrigger";
import { DeleteAnalysisModal, DeletedToastModal } from "./DeleteAnalysisModal";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { UploadDropzone } from "./UploadDropzone";
import { DataSourceGrid } from "./DataSourceGrid";
import { CircularLoaderWithLabel } from "@/components/ui/circular-loader";

export type AnalysisRow = {
  id: string;
  name: string;
  projectName: string;
  teamCount: number;
  teamLabel?: string;
  createdOn: string;
  createdBy: { name: string; role: string; initials?: string; src?: string };
  members: { initials?: string; src?: string }[];
};

interface ProjectDetailsViewProps {
  projectId?: string | null;
  projectName: string;
  projectDescription?: string | null;
  rows: AnalysisRow[];
  onRenameAnalysis?: (id: string, name: string) => void;
  onDeleteAnalysis?: (id: string) => void;
}

const PAGES = [1, 2, 3, "...", 8, 9, 10] as const;

type DateFilterId = "all" | "today" | "week" | "month" | "90d" | "year";

const DATE_FILTERS: { id: DateFilterId; label: string; days: number | null }[] = [
  { id: "all", label: "All time", days: null },
  { id: "today", label: "Today", days: 1 },
  { id: "week", label: "This week", days: 7 },
  { id: "month", label: "This month", days: 30 },
  { id: "90d", label: "Last 90 days", days: 90 },
  { id: "year", label: "This year", days: 365 },
];

function withinDateFilter(createdOn: string, filter: DateFilterId) {
  const def = DATE_FILTERS.find((f) => f.id === filter);
  if (!def || def.days === null) return true;
  const ts = Date.parse(createdOn);
  if (Number.isNaN(ts)) return true;
  const cutoff = Date.now() - def.days * 24 * 60 * 60 * 1000;
  return ts >= cutoff;
}


function AnalysisDropdown({ onRename, onDelete }: { onRename?: () => void; onDelete?: () => void }) {
  return (
    <Dropdown.Root>
      <Dropdown.DotsButton aria-label="More options" className="rounded-md p-1" />
      <Dropdown.Popover placement="bottom right" offset={6} className="w-44">
        <Dropdown.Menu>
          <Dropdown.Item icon={Edit01} onAction={onRename}>
            Rename
          </Dropdown.Item>
          <Dropdown.Item icon={Archive}>Archive</Dropdown.Item>
          <Dropdown.Item icon={Pin01}>Pin</Dropdown.Item>
          <Dropdown.Separator />
          <Dropdown.Item icon={Trash01} onAction={onDelete}>
            Delete
          </Dropdown.Item>
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown.Root>
  );
}

function MemberStack({ members, extra }: { members: { initials?: string; src?: string }[]; extra?: number }) {
  return (
    <div className="flex -space-x-2">
      {members.slice(0, 4).map((m, i) => (
        <Avatar key={i} size="xs" initials={m.initials} src={m.src} contrastBorder />
      ))}
      {extra && extra > 0 ? (
        <div className="flex size-6 items-center justify-center rounded-full border-2 border-primary bg-secondary text-[10px] font-semibold text-secondary">
          +{extra}
        </div>
      ) : null}
    </div>
  );
}

export function ProjectDetailsView({
  projectId,
  projectName,
  projectDescription,
  rows,
  onRenameAnalysis,
  onDeleteAnalysis,
}: ProjectDetailsViewProps) {
  const navigate = useNavigate();
  const [view, setView] = useState<"grid" | "table">("grid");
  const [query, setQuery] = useState("");
  const [dateFilter, setDateFilter] = useState<DateFilterId>("all");

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [page, setPage] = useState(1);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<AnalysisRow | null>(null);
  const [deletedToast, setDeletedToast] = useState(false);
  const [openingAnalysisId, setOpeningAnalysisId] = useState<string | null>(null);

  const openAnalysis = (row: AnalysisRow) => {
    setOpeningAnalysisId(row.id);
    try {
      sessionStorage.setItem(
        "analysis:context",
        JSON.stringify({
          name: row.name,
          project: projectName,
          projectId: projectId ?? null,
          analysisId: row.id,
        }),
      );
    } catch {}
    navigate({ to: "/analysis", search: { aid: row.id } });
  };

  const filtered = useMemo(
    () =>
      rows.filter(
        (r) =>
          [r.name, r.projectName, r.createdBy.name].some((s) =>
            s.toLowerCase().includes(query.toLowerCase()),
          ) && withinDateFilter(r.createdOn, dateFilter),
      ),
    [rows, query, dateFilter],
  );


  const allSelected = filtered.length > 0 && filtered.every((r) => selected.has(r.id));

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const toggleAll = () =>
    setSelected((prev) => {
      if (allSelected) return new Set();
      return new Set(filtered.map((r) => r.id));
    });

  const handleRename = (r: AnalysisRow) => {
    const next = typeof window !== "undefined" ? window.prompt("Rename analysis", r.name) : null;
    if (next && next.trim() && next !== r.name) onRenameAnalysis?.(r.id, next.trim());
  };
  const handleDelete = (r: AnalysisRow) => {
    setDeleteTarget(r);
  };


  return (
    <div className="relative flex h-full flex-col">
      <header className="flex items-center justify-between gap-2 border-b border-secondary bg-primary px-6 py-4">
        <div className="flex items-start gap-2">
          <Monitor01 className="mt-0.5 size-5 text-fg-quaternary" />
          <div>
            <h2 className="text-md font-semibold text-primary">{projectName}</h2>
            {projectDescription ? (
              <p className="mt-0.5 text-sm text-tertiary">{projectDescription}</p>
            ) : null}
          </div>
        </div>
        {projectId ? <InviteCollaboratorsTrigger projectId={projectId} /> : null}
      </header>

      <div className="flex items-center justify-between gap-3 px-6 pt-5">
        <div className="flex flex-1 items-center gap-3">
          <div className="w-[360px]">
            <Input
              aria-label="Search"
              placeholder="Search for trades"
              size="md"
              icon={SearchLg}
              shortcut="⌘K"
              value={query}
              onChange={setQuery}
            />
          </div>
          <Dropdown.Root>
            <Button color="secondary" size="md" iconLeading={Calendar} iconTrailing={ChevronDown}>
              {DATE_FILTERS.find((f) => f.id === dateFilter)?.label ?? "All time"}
            </Button>
            <Dropdown.Popover placement="bottom left" offset={6} className="w-48">
              <Dropdown.Menu
                selectionMode="single"
                selectedKeys={[dateFilter]}
                onSelectionChange={(keys) => {
                  const next = Array.from(keys as Set<string>)[0];
                  if (next) setDateFilter(next as DateFilterId);
                }}
              >
                {DATE_FILTERS.map((f) => (
                  <Dropdown.Item key={f.id} id={f.id} label={f.label}>
                    {f.label}
                  </Dropdown.Item>
                ))}
              </Dropdown.Menu>
            </Dropdown.Popover>
          </Dropdown.Root>


        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setUploadOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#1565ef] px-3 py-2 text-sm font-semibold text-white shadow-xs hover:bg-[#1257d6]"
          >
            <Plus className="size-4" />
            New Analysis
          </button>

          <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-0.5 shadow-xs">
            <button
              type="button"
              onClick={() => setView("grid")}
              className={cx(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-semibold",
                view === "grid" ? "bg-secondary text-primary shadow-xs" : "text-tertiary hover:text-primary",
              )}
            >
              <Grid01 className="size-4" />
              Grid
            </button>
            <button
              type="button"
              onClick={() => setView("table")}
              className={cx(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-semibold",
                view === "table" ? "bg-secondary text-primary shadow-xs" : "text-tertiary hover:text-primary",
              )}
            >
              <Rows01 className="size-4" />
              Table
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {view === "grid" ? (
          <GridView
            rows={filtered}
            selected={selected}
            openingId={openingAnalysisId}
            onToggle={toggle}
            onOpen={openAnalysis}
            onRename={handleRename}
            onDelete={handleDelete}
          />
        ) : (
          <TableView
            rows={filtered}
            selected={selected}
            openingId={openingAnalysisId}
            onToggle={toggle}
            allSelected={allSelected}
            onToggleAll={toggleAll}
            onOpen={openAnalysis}
            onRename={handleRename}
            onDelete={handleDelete}
          />
        )}
      </div>

      <footer className="flex items-center justify-between border-t border-secondary bg-primary px-6 py-3">
        <Button color="secondary" size="sm" iconLeading={ArrowLeft} onClick={() => setPage((p) => Math.max(1, p - 1))}>
          Previous
        </Button>
        <div className="flex items-center gap-1">
          {PAGES.map((p, i) =>
            p === "..." ? (
              <span key={i} className="px-2 text-sm text-tertiary">
                ...
              </span>
            ) : (
              <button
                key={i}
                type="button"
                onClick={() => setPage(p as number)}
                className={cx(
                  "flex size-8 items-center justify-center rounded-md text-sm font-medium",
                  page === p ? "bg-secondary text-primary" : "text-tertiary hover:bg-primary_hover",
                )}
              >
                {p}
              </button>
            ),
          )}
        </div>
        <Button color="secondary" size="sm" iconTrailing={ArrowRight} onClick={() => setPage((p) => p + 1)}>
          Next
        </Button>
      </footer>

      <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
        <DialogContent className="max-w-2xl bg-white p-6">
          <DialogHeader>
            <DialogTitle>New Analysis</DialogTitle>
          </DialogHeader>
          <section>
            <UploadDropzone
              projectId={projectId ?? null}
              projectName={projectName}
            />
          </section>
          <div className="my-6 flex items-center gap-4">
            <div className="h-px flex-1 bg-border-secondary" />
            <span className="text-sm text-tertiary">Or</span>
            <div className="h-px flex-1 bg-border-secondary" />
          </div>
          <section>
            <DataSourceGrid />
          </section>
        </DialogContent>
      </Dialog>

      <DeleteAnalysisModal
        open={!!deleteTarget}
        onOpenChange={(o) => {
          if (!o) setDeleteTarget(null);
        }}
        onConfirm={() => {
          if (deleteTarget) {
            onDeleteAnalysis?.(deleteTarget.id);
            setDeleteTarget(null);
            setDeletedToast(true);
          }
        }}
      />
      <DeletedToastModal open={deletedToast} onOpenChange={setDeletedToast} label="Analysis deleted" />

    </div>
  );
}

function GridView({
  rows,
  selected,
  openingId,
  onToggle,
  onOpen,
  onRename,
  onDelete,
}: {
  rows: AnalysisRow[];
  selected: Set<string>;
  openingId: string | null;
  onToggle: (id: string) => void;
  onOpen: (row: AnalysisRow) => void;
  onRename: (r: AnalysisRow) => void;
  onDelete: (r: AnalysisRow) => void;
}) {
  return (
    <div className="grid grid-cols-3 gap-4">
      {rows.map((r) => {
        const isSelected = selected.has(r.id);
        const isOpening = openingId === r.id;
        const extra = Math.max(0, r.teamCount - 4);
        return (
          <div
            key={r.id}
            className={cx(
              "relative flex flex-col gap-4 rounded-xl border bg-primary p-4 shadow-xs",
              isSelected ? "border-brand ring-1 ring-brand" : "border-secondary",
              isOpening && "opacity-90",
            )}
          >
            {isOpening && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 rounded-xl bg-primary/80 backdrop-blur-sm">
                <CircularLoaderWithLabel label="Opening..." size={40} />
              </div>
            )}
            <div className="flex items-start justify-between">
              <button
                type="button"
                onClick={() => onToggle(r.id)}
                aria-label={`Select ${r.name}`}
                className={cx(
                  "flex size-9 items-center justify-center rounded-lg",
                  isSelected ? "bg-brand-secondary" : "bg-success-secondary",
                )}
              >
                <File02 className={cx("size-5", isSelected ? "text-fg-brand-primary" : "text-fg-success-primary")} />
              </button>
              <AnalysisDropdown onRename={() => onRename(r)} onDelete={() => onDelete(r)} />
            </div>

            <button type="button" onClick={() => onOpen(r)} className="text-left">
              <p className="text-md font-semibold text-primary hover:underline">{r.name}</p>
              <p className="mt-2 text-sm text-tertiary">{r.teamLabel ?? "Team"}</p>
              <div className="mt-2 flex items-center gap-2">
                <MemberStack members={r.members} extra={extra} />
                <span className="text-sm text-tertiary">{r.teamCount} users</span>
              </div>
            </button>

            <div className="-mx-4 mt-2 flex items-center justify-between gap-3 border-t border-secondary px-4 pt-3">
              <div className="flex min-w-0 items-center gap-2">
                <Avatar size="sm" initials={r.createdBy.initials} src={r.createdBy.src} alt={r.createdBy.name} />
                <span className="truncate text-sm font-medium text-primary">{r.createdBy.name}</span>
              </div>
              <span className="shrink-0 text-xs text-tertiary">Created on {r.createdOn}</span>
            </div>

          </div>
        );
      })}
    </div>
  );
}

function TableView({
  rows,
  selected,
  openingId,
  onToggle,
  allSelected,
  onToggleAll,
  onOpen,
  onRename,
  onDelete,
}: {
  rows: AnalysisRow[];
  selected: Set<string>;
  openingId: string | null;
  onToggle: (id: string) => void;
  allSelected: boolean;
  onToggleAll: () => void;
  onOpen: (row: AnalysisRow) => void;
  onRename: (r: AnalysisRow) => void;
  onDelete: (r: AnalysisRow) => void;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-secondary bg-primary shadow-xs">
      <table className="w-full">
        <thead className="border-b border-secondary bg-secondary/40">
          <tr className="text-left">
            <th className="w-10 px-4 py-3">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={onToggleAll}
                aria-label="Select all"
                className="size-4 cursor-pointer rounded border-secondary"
              />
            </th>
            <th className="px-4 py-3 text-xs font-medium text-tertiary">Analysis Name</th>
            <th className="px-4 py-3 text-xs font-medium text-tertiary">Project Name</th>
            <th className="px-4 py-3 text-xs font-medium text-tertiary">Team</th>
            <th className="px-4 py-3 text-xs font-medium text-tertiary">Owner</th>
            <th className="px-4 py-3 text-xs font-medium text-tertiary">Created On</th>

            <th className="px-4 py-3 text-xs font-medium text-tertiary">Create By</th>
            <th className="w-12 px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isSelected = selected.has(r.id);
            const isOpening = openingId === r.id;
            const extra = Math.max(0, r.teamCount - 4);
            if (isOpening) {
              return (
                <tr key={r.id} className="border-b border-secondary last:border-b-0">
                  <td colSpan={8} className="bg-secondary/30 px-4 py-6">
                    <div className="flex items-center justify-center gap-3">
                      <CircularLoaderWithLabel label="Opening analysis..." size={32} />
                    </div>
                  </td>
                </tr>
              );
            }
            return (
              <tr key={r.id} className="border-b border-secondary last:border-b-0">
                <td className="px-4 py-3">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => onToggle(r.id)}
                    aria-label={`Select ${r.name}`}
                    className="size-4 cursor-pointer rounded border-secondary"
                  />
                </td>
                <td className="px-4 py-3 text-sm font-medium text-primary">
                  <button type="button" onClick={() => onOpen(r)} className="hover:underline">
                    {r.name}
                  </button>
                </td>
                <td className="px-4 py-3 text-sm text-secondary">{r.projectName}</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-secondary">{String(r.teamCount).padStart(2, "0")} users</span>
                    <MemberStack members={r.members} extra={extra} />
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <Avatar size="xs" initials={r.createdBy.initials} src={r.createdBy.src} alt={r.createdBy.name} />
                    <span className="text-sm font-medium text-primary">{r.createdBy.name}</span>
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-secondary">{r.createdOn}</td>

                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <Avatar size="sm" initials={r.createdBy.initials} src={r.createdBy.src} alt={r.createdBy.name} />
                    <div className="leading-tight">
                      <p className="text-sm font-medium text-primary">{r.createdBy.name}</p>
                      <p className="text-xs text-tertiary">{r.createdBy.role}</p>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <AnalysisDropdown onRename={() => onRename(r)} onDelete={() => onDelete(r)} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
