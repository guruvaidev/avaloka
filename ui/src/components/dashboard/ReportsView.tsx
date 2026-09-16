import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  SearchLg,
  File02,
  ArrowLeft,
  ArrowRight,
  ChevronSelectorVertical,
  Edit01,
  Archive,
  Bookmark,
  Pin01,
  Trash01,
} from "@untitledui/icons";
import { Input } from "@/components/base/input/input";
import { Button } from "@/components/base/buttons/button";
import { Avatar } from "@/components/base/avatar/avatar";
import { Dropdown } from "@/components/base/dropdown/dropdown";
import { cx } from "@/lib/utils/cx";

export type ReportCategory = "Revenue Dataset" | "Finance Dataset" | "Accounts Dataset";

export type ReportRow = {
  id: string;
  title: string;
  category: ReportCategory;
  teamCount: number;
  members: { initials?: string; src?: string }[];
  createdOn: string;
  createdBy: { name: string; role: string; initials?: string; src?: string };
  canDelete?: boolean;
};


const TABS = ["View all", "Your files", "Shared files"] as const;
type TabId = (typeof TABS)[number];

const PAGE_SIZE = 10;

function buildPageList(current: number, total: number): (number | "...")[] {
  if (total <= 1) return [1];
  const pages = new Set<number>([1, total, current, current - 1, current + 1]);
  const sorted = Array.from(pages).filter((p) => p >= 1 && p <= total).sort((a, b) => a - b);
  const result: (number | "...")[] = [];
  let prev = 0;
  for (const p of sorted) {
    if (prev && p - prev > 1) result.push("...");
    result.push(p);
    prev = p;
  }
  return result;
}

const categoryStyles: Record<ReportCategory, string> = {
  "Revenue Dataset":
    "bg-success-secondary text-fg-success-primary ring-success-subtle",
  "Finance Dataset":
    "bg-brand-secondary text-fg-brand-secondary ring-brand-subtle",
  "Accounts Dataset":
    "bg-utility-blue-50 text-utility-blue-700 ring-utility-blue-200",
};


function MemberStack({
  members,
  extra,
}: {
  members: { initials?: string; src?: string }[];
  extra?: number;
}) {
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

function SortHeader({ label }: { label: string }) {
  return (
    <button
      type="button"
      className="inline-flex items-center gap-1 text-xs font-medium text-tertiary hover:text-primary"
    >
      {label}
      <ChevronSelectorVertical className="size-3" />
    </button>
  );
}

export function ReportsView({
  rows,
  bookmarkedIds,
  onToggleBookmark,
  onDelete,
}: {
  rows: ReportRow[];
  bookmarkedIds?: Set<string>;
  onToggleBookmark?: (id: string) => void;
  onDelete?: (id: string) => void;
}) {
  const navigate = useNavigate();
  const [tab, setTab] = useState<TabId>("View all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);

  const openReport = (reportId: string) => {
    navigate({ to: "/reports/$reportId", params: { reportId } });
  };

  const filtered = useMemo(
    () =>
      rows.filter((r) =>
        [r.title, r.category, r.createdBy.name].some((s) =>
          s.toLowerCase().includes(query.toLowerCase()),
        ),
      ),
    [rows, query],
  );

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageRows = useMemo(
    () => filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE),
    [filtered, currentPage],
  );
  const pageList = useMemo(() => buildPageList(currentPage, totalPages), [currentPage, totalPages]);

  // Reset to first page when filter/tab changes.
  useEffect(() => {
    setPage(1);
  }, [query, tab, rows.length]);


  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-2 border-b border-secondary bg-primary px-6 py-4">
        <File02 className="size-5 text-fg-quaternary" />
        <h2 className="text-md font-semibold text-primary">Reports</h2>
      </header>

      <div className="flex items-center justify-between gap-3 px-6 pt-5">
        <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-0.5 shadow-xs">
          {TABS.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={cx(
                "rounded-md px-3 py-1.5 text-sm font-semibold",
                tab === t
                  ? "bg-secondary text-primary shadow-xs"
                  : "text-tertiary hover:text-primary",
              )}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="w-[320px]">
          <Input
            aria-label="Search"
            placeholder="Search"
            size="md"
            icon={SearchLg}
            shortcut="⌘K"
            value={query}
            onChange={setQuery}
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="overflow-hidden rounded-xl border border-secondary bg-primary shadow-xs">
          <table className="w-full">
            <thead className="border-b border-secondary bg-secondary/40">
              <tr className="text-left">
                <th className="px-6 py-3"><SortHeader label="Title" /></th>
                <th className="px-6 py-3"><SortHeader label="Category" /></th>
                <th className="px-6 py-3 text-xs font-medium text-tertiary">Team</th>
                <th className="px-6 py-3"><SortHeader label="Created On" /></th>
                <th className="px-6 py-3"><SortHeader label="Create By" /></th>
                <th className="w-12 px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {pageRows.map((r) => {
                const extra = Math.max(0, r.teamCount - 4);
                return (
                  <tr
                    key={r.id}
                    role="link"
                    tabIndex={0}
                    onClick={() => openReport(r.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        openReport(r.id);
                      }
                    }}
                    className="cursor-pointer border-b border-secondary last:border-b-0 hover:bg-primary_hover focus:outline-none focus-visible:bg-primary_hover"
                  >
                    <td className="px-6 py-4 text-sm font-medium text-primary">
                      <span className="inline-flex items-center gap-1.5">
                        <Link
                          to="/reports/$reportId"
                          params={{ reportId: r.id }}
                          className="hover:underline"
                        >
                          {r.title}
                        </Link>
                        {bookmarkedIds?.has(r.id) ? (
                          <Bookmark className="size-3.5 fill-brand-solid text-brand-solid" />
                        ) : null}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={cx(
                          "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset",
                          categoryStyles[r.category],
                        )}
                      >
                        {r.category}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-secondary">
                          {String(r.teamCount).padStart(2, "0")} users
                        </span>
                        <MemberStack members={r.members} extra={extra} />
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-secondary">{r.createdOn}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <Avatar
                          size="sm"
                          initials={r.createdBy.initials}
                          src={r.createdBy.src}
                          alt={r.createdBy.name}
                        />
                        <div className="leading-tight">
                          <p className="text-sm font-medium text-primary">{r.createdBy.name}</p>
                          <p className="text-xs text-tertiary">{r.createdBy.role}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4 text-right" onClick={(event) => event.stopPropagation()}>
                      <Dropdown.Root>
                        <Dropdown.DotsButton aria-label={`Options for ${r.title}`} className="rounded-md p-1" />
                        <Dropdown.Popover placement="bottom right" offset={6} className="w-44">
                          <Dropdown.Menu>
                            <Dropdown.Item icon={Edit01}>Rename</Dropdown.Item>
                            <Dropdown.Item icon={Archive}>Archive</Dropdown.Item>
                            <Dropdown.Item
                              icon={Bookmark}
                              onAction={() => onToggleBookmark?.(r.id)}
                            >
                              {bookmarkedIds?.has(r.id) ? "Remove Bookmark" : "Bookmark"}
                            </Dropdown.Item>
                            <Dropdown.Item icon={Pin01}>Pin</Dropdown.Item>
                            {r.canDelete !== false ? (
                              <>
                                <Dropdown.Separator />
                                <Dropdown.Item icon={Trash01} onAction={() => onDelete?.(r.id)}>Delete</Dropdown.Item>
                              </>
                            ) : null}

                          </Dropdown.Menu>
                        </Dropdown.Popover>
                      </Dropdown.Root>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <footer className="flex items-center justify-between border-t border-secondary bg-primary px-6 py-3">
        <Button
          color="secondary"
          size="sm"
          iconLeading={ArrowLeft}
          isDisabled={currentPage <= 1}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
        >
          Previous
        </Button>
        <div className="flex items-center gap-1">
          {pageList.map((p, i) =>
            p === "..." ? (
              <span key={`gap-${i}`} className="px-2 text-sm text-tertiary">
                ...
              </span>
            ) : (
              <button
                key={p}
                type="button"
                onClick={() => setPage(p)}
                className={cx(
                  "flex size-8 items-center justify-center rounded-md text-sm font-medium",
                  currentPage === p ? "bg-secondary text-primary" : "text-tertiary hover:bg-primary_hover",
                )}
              >
                {p}
              </button>
            ),
          )}
        </div>
        <Button
          color="secondary"
          size="sm"
          iconTrailing={ArrowRight}
          isDisabled={currentPage >= totalPages}
          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
        >
          Next
        </Button>
      </footer>

    </div>
  );
}
