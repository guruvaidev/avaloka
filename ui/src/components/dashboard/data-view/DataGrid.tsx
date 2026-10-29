// BG38–BG41: data table with sort, filters, column search, page size, pinned columns
// and type-based alignment. Works only on the rows passed in; never fetches.

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, BarChart3, Filter, Pin, PinOff, Search, X } from "lucide-react";
import { cx } from "@/lib/utils/cx";
import {
  applyFilters,
  cellAlign,
  choosePinnedColumn,
  columnKinds,
  filterChipText,
  formatValue,
  isFilterActive,
  sortRows,
  topValues,
  type ColumnFilter,
  type ColumnKind,
  type ColumnProfile,
  type SortDir,
} from "@/lib/data-view";

type Row = Record<string, unknown>;
export const PAGE_SIZE_KEY = "dataView:pageSize";
const PAGE_SIZES = [10, 25, 50, 100];
const NUMBER_COL_WIDTH = 48;

function readPageSize(): number {
  try {
    const saved = Number(localStorage.getItem(PAGE_SIZE_KEY));
    return PAGE_SIZES.includes(saved) ? saved : 25;
  } catch {
    return 25;
  }
}

export type HighlightRequest = { name: string; nonce: number } | null;

export function DataGrid({
  rows,
  columns,
  profiles = [],
  previewOf,
  emptyMessage = "No table data saved for this analysis yet.",
  highlight,
  onChartColumn,
}: {
  rows: Row[];
  columns: string[];
  profiles?: ColumnProfile[];
  /** Total dataset rows when `rows` is only a preview. */
  previewOf?: number;
  emptyMessage?: string;
  highlight?: HighlightRequest;
  onChartColumn?: (column: string, kind: ColumnKind) => void;
}) {
  const kinds = useMemo(() => columnKinds(rows, columns, profiles), [rows, columns, profiles]);
  const [sort, setSort] = useState<{ column: string; dir: SortDir } | null>(null);
  const [filters, setFilters] = useState<Record<string, ColumnFilter>>({});
  const [pageSize, setPageSize] = useState(25);
  const [page, setPage] = useState(1);
  const [pinOverride, setPinOverride] = useState<string | null | undefined>(undefined);
  const [menu, setMenu] = useState<{ column: string; top: number; left: number } | null>(null);
  const [find, setFind] = useState("");
  const [findOpen, setFindOpen] = useState(false);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [scroll, setScroll] = useState({ left: false, right: false });
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const headerRefs = useRef<Record<string, HTMLTableCellElement | null>>({});
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => setPageSize(readPageSize()), []);

  const pinned = pinOverride === undefined ? choosePinnedColumn(columns, profiles) : pinOverride;
  const ordered = useMemo(() => (pinned && columns.includes(pinned) ? [pinned, ...columns.filter((c) => c !== pinned)] : columns), [columns, pinned]);

  const filtered = useMemo(() => applyFilters(rows, filters), [rows, filters]);
  const sorted = useMemo(() => (sort ? sortRows(filtered, sort.column, sort.dir, kinds[sort.column] ?? "text") : filtered), [filtered, sort, kinds]);
  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * pageSize;
  const pageRows = sorted.slice(start, start + pageSize);
  const activeFilters = Object.entries(filters).filter(([, f]) => isFilterActive(f));

  const updateScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    setScroll({ left: el.scrollLeft > 0, right: el.scrollLeft + el.clientWidth < el.scrollWidth - 1 });
  };
  useEffect(() => {
    updateScroll();
    window.addEventListener("resize", updateScroll);
    return () => window.removeEventListener("resize", updateScroll);
  }, [ordered.length, pageRows.length]);

  useEffect(() => {
    if (!menu) return;
    const close = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) setMenu(null);
    };
    const esc = (event: KeyboardEvent) => event.key === "Escape" && setMenu(null);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [menu]);

  const jumpTo = (column: string) => {
    setHighlighted(column);
    setFindOpen(false);
    requestAnimationFrame(() => headerRefs.current[column]?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" }));
  };
  useEffect(() => {
    if (highlight?.name && columns.includes(highlight.name)) {
      setFind(highlight.name);
      jumpTo(highlight.name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlight?.nonce]);

  const matches = find.trim() ? columns.filter((c) => c.toLowerCase().includes(find.trim().toLowerCase())) : [];

  const cycleSort = (column: string) => {
    setPage(1);
    setSort((prev) => (prev?.column !== column ? { column, dir: "asc" } : prev.dir === "asc" ? { column, dir: "desc" } : null));
  };
  const setFilter = (column: string, next: ColumnFilter) => {
    setPage(1);
    setFilters((prev) => ({ ...prev, [column]: next }));
  };
  const choosePageSize = (size: number) => {
    setPageSize(size);
    setPage(1);
    try {
      localStorage.setItem(PAGE_SIZE_KEY, String(size));
    } catch {
      /* preference is optional */
    }
  };

  const stickyStyle = (column: string | "#", header = false): CSSProperties | undefined => {
    if (column === "#") return { position: "sticky", left: 0, zIndex: header ? 4 : 2, width: NUMBER_COL_WIDTH, minWidth: NUMBER_COL_WIDTH };
    if (column === pinned) return { position: "sticky", left: NUMBER_COL_WIDTH, zIndex: header ? 4 : 2, boxShadow: scroll.left ? "6px 0 6px -4px rgba(16,24,40,0.18)" : undefined };
    return undefined;
  };

  return (
    <div className="flex h-[calc(100dvh-220px)] min-h-[380px] flex-col overflow-hidden rounded-xl border border-secondary bg-primary">
      <div className="flex shrink-0 flex-col gap-2 border-b border-secondary px-3 py-2">
        <div className="relative w-full max-w-xs">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-fg-quaternary" />
          <input
            aria-label="Find column"
            placeholder="Find column…"
            value={find}
            onChange={(event) => {
              const value = event.target.value;
              setFind(value);
              const found = value.trim() ? columns.filter((c) => c.toLowerCase().includes(value.trim().toLowerCase())) : [];
              setFindOpen(found.length > 1);
              if (found.length) jumpTo(found[0]);
              else setHighlighted(null);
              if (found.length > 1) setFindOpen(true);
            }}
            onFocus={() => setFindOpen(matches.length > 1)}
            onBlur={() => setTimeout(() => setFindOpen(false), 150)}
            className="h-8 w-full rounded-lg border border-secondary bg-primary pl-8 pr-2 text-sm text-primary outline-none focus:border-[#1a5fe0]"
          />
          {findOpen && matches.length > 1 && (
            <ul role="listbox" className="absolute left-0 right-0 top-9 z-30 max-h-56 overflow-auto rounded-lg border border-secondary bg-primary py-1 shadow-lg">
              {matches.map((column) => (
                <li key={column}>
                  <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => jumpTo(column)} className={cx("w-full px-3 py-1.5 text-left text-sm hover:bg-secondary", highlighted === column ? "text-[#1a5fe0]" : "text-primary")}>
                    {column}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        {activeFilters.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            {activeFilters.map(([column, filter]) => (
              <span key={column} className="inline-flex items-center gap-1 rounded-full bg-[#eaf1fd] px-2.5 py-0.5 text-xs font-medium text-[#1a5fe0]">
                {filterChipText(column, filter)}
                <button type="button" aria-label={`Remove ${column} filter`} onClick={() => setFilter(column, {})} className="rounded-full p-0.5 hover:bg-[#d4e3fb]">
                  <X className="size-3" />
                </button>
              </span>
            ))}
            <button type="button" onClick={() => { setFilters({}); setPage(1); }} className="text-xs font-semibold text-tertiary underline-offset-2 hover:underline">
              Clear all filters
            </button>
          </div>
        )}
        {(activeFilters.length > 0 || sort) && previewOf != null && previewOf > rows.length && (
          <p className="text-xs text-tertiary">Sorting and filters apply to the {rows.length.toLocaleString()} loaded rows.</p>
        )}
      </div>

      <div className="relative min-h-0 flex-1">
        <div ref={scrollRef} onScroll={updateScroll} className="h-full overflow-auto">
          <table className="w-full border-separate border-spacing-0 text-[13px]">
            <thead>
              <tr>
                <th style={stickyStyle("#", true)} className="sticky top-0 border-b border-secondary bg-secondary px-3 py-2 text-left font-medium text-tertiary">#</th>
                {ordered.map((column) => {
                  const kind = kinds[column] ?? "text";
                  const align = cellAlign(kind);
                  const sorted = sort?.column === column ? sort.dir : null;
                  const filterOn = isFilterActive(filters[column]);
                  return (
                    <th
                      key={column}
                      ref={(el) => { headerRefs.current[column] = el; }}
                      aria-sort={sorted === "asc" ? "ascending" : sorted === "desc" ? "descending" : "none"}
                      style={{ zIndex: 3, ...stickyStyle(column, true) }}
                      className={cx(
                        "group sticky top-0 whitespace-nowrap border-b border-secondary px-2 py-1.5 font-medium",
                        highlighted === column ? "bg-[#eaf1fd] text-[#1a5fe0]" : "bg-secondary text-tertiary",
                      )}
                    >
                      <div className={cx("flex items-center gap-1", align === "right" && "flex-row-reverse justify-start")}>
                        <button type="button" onClick={() => cycleSort(column)} className="relative inline-flex items-center rounded py-0.5 hover:text-primary">
                          <span>{column}</span>
                          <span className="pointer-events-none absolute -top-2 right-0">{sorted === "asc" ? <ArrowUp className="size-3 text-[#1a5fe0]" aria-hidden /> : sorted === "desc" ? <ArrowDown className="size-3 text-[#1a5fe0]" aria-hidden /> : <ArrowUpDown className="size-3 opacity-0 transition group-hover:opacity-50" aria-hidden />}</span>
                        </button>
                        <button
                          type="button"
                          aria-label={`Column options for ${column}`}
                          onClick={(event) => {
                            const rect = event.currentTarget.getBoundingClientRect();
                            setMenu(menu?.column === column ? null : { column, top: rect.bottom + 4, left: Math.min(rect.left, window.innerWidth - 300) });
                          }}
                          className={cx("shrink-0 rounded hover:bg-primary_hover", filterOn ? "text-brand-secondary" : "text-fg-quaternary")}
                        >
                          <Filter className="size-3.5" fill={filterOn ? "currentColor" : "none"} />
                        </button>
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {pageRows.length === 0 ? (
                <tr>
                  <td colSpan={ordered.length + 1} className="px-4 py-10 text-center text-sm text-tertiary">
                    {rows.length && activeFilters.length ? "No rows match the current filters." : emptyMessage}
                  </td>
                </tr>
              ) : (
                pageRows.map((row, index) => (
                  <tr key={start + index} className="hover:[&>td]:bg-secondary">
                    <td style={stickyStyle("#")} className="border-b border-secondary bg-primary px-3 py-1.5 leading-5 text-tertiary tabular-nums">{start + index + 1}</td>
                    {ordered.map((column) => {
                      const kind = kinds[column] ?? "text";
                      const right = cellAlign(kind) === "right";
                      const text = formatValue(row?.[column], kind, column);
                      return (
                        <td
                          key={column}
                          style={stickyStyle(column)}
                          className={cx(
                            "max-w-[260px] truncate whitespace-nowrap border-b border-secondary px-3 py-1.5 leading-5",
                            right ? "text-right font-medium text-primary tabular-nums" : "text-left text-secondary",
                            highlighted === column ? "bg-[#f3f7fe]" : "bg-primary",
                          )}
                          title={text}
                        >
                          {text}
                        </td>
                      );
                    })}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        {scroll.right && <div aria-hidden className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l from-[rgba(16,24,40,0.10)] to-transparent" />}
      </div>

      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-t border-secondary px-4 py-2">
        <p className="text-xs text-tertiary">
          {sorted.length === 0 ? "Showing 0" : `Showing ${start + 1}–${Math.min(start + pageSize, sorted.length)}`} of {sorted.length.toLocaleString()}
          {activeFilters.length ? ` (filtered from ${rows.length.toLocaleString()})` : ""}
          {!activeFilters.length && previewOf != null && previewOf > rows.length ? ` (preview · ${previewOf.toLocaleString()} rows total)` : ""}
        </p>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs text-tertiary">
            Rows per page
            <select aria-label="Rows per page" value={pageSize} onChange={(e) => choosePageSize(Number(e.target.value))} className="h-7 rounded-md border border-secondary bg-primary px-1.5 text-xs text-primary">
              {PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}
            </select>
          </label>
          <div className="flex items-center gap-1">
            <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={safePage === 1} className="rounded-md border border-secondary bg-primary px-2.5 py-1 text-xs font-semibold text-primary disabled:cursor-not-allowed disabled:opacity-50">Previous</button>
            <span className="px-1.5 text-xs text-tertiary">{safePage} / {totalPages}</span>
            <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={safePage === totalPages} className="rounded-md border border-secondary bg-primary px-2.5 py-1 text-xs font-semibold text-primary disabled:cursor-not-allowed disabled:opacity-50">Next</button>
          </div>
        </div>
      </div>

      {menu && (
        <div ref={menuRef} role="dialog" aria-label={`${menu.column} options`} style={{ position: "fixed", top: menu.top, left: Math.max(8, menu.left) }} className="z-[60] w-72 rounded-xl border border-secondary bg-primary p-3 shadow-xl">
          <ColumnMenu
            column={menu.column}
            kind={kinds[menu.column] ?? "text"}
            rows={rows}
            filter={filters[menu.column] ?? {}}
            onFilter={(next) => setFilter(menu.column, next)}
            pinned={pinned === menu.column}
            onPin={() => { setPinOverride(pinned === menu.column ? null : menu.column); setMenu(null); }}
            onChart={onChartColumn ? () => { onChartColumn(menu.column, kinds[menu.column] ?? "text"); setMenu(null); } : undefined}
          />
        </div>
      )}
    </div>
  );
}

function ColumnMenu({ column, kind, rows, filter, onFilter, pinned, onPin, onChart }: {
  column: string; kind: ColumnKind; rows: Row[]; filter: ColumnFilter;
  onFilter: (next: ColumnFilter) => void; pinned: boolean; onPin: () => void; onChart?: () => void;
}) {
  const options = useMemo(() => topValues(rows, column, 20), [rows, column]);
  const selected = new Set(filter.values ?? []);
  const toggle = (value: string) => {
    const next = new Set(selected);
    if (next.has(value)) next.delete(value); else next.add(value);
    onFilter({ ...filter, values: [...next] });
  };
  const num = (value: string) => (value.trim() === "" || Number.isNaN(Number(value)) ? undefined : Number(value));
  const input = "h-8 w-full rounded-md border border-secondary bg-primary px-2 text-sm text-primary outline-none focus:border-[#1a5fe0]";
  return (
    <div className="flex flex-col gap-2 text-sm">
      <p className="truncate text-xs font-semibold uppercase tracking-wide text-tertiary">Filter {column}</p>
      {kind === "number" ? (
        <div className="flex gap-2">
          <input aria-label="Minimum" type="number" placeholder="Min" defaultValue={filter.min ?? ""} onChange={(e) => onFilter({ ...filter, min: num(e.target.value) })} className={input} />
          <input aria-label="Maximum" type="number" placeholder="Max" defaultValue={filter.max ?? ""} onChange={(e) => onFilter({ ...filter, max: num(e.target.value) })} className={input} />
        </div>
      ) : kind === "date" ? (
        <div className="flex gap-2">
          <input aria-label="From" type="date" defaultValue={filter.from ?? ""} onChange={(e) => onFilter({ ...filter, from: e.target.value || undefined })} className={input} />
          <input aria-label="To" type="date" defaultValue={filter.to ?? ""} onChange={(e) => onFilter({ ...filter, to: e.target.value || undefined })} className={input} />
        </div>
      ) : (
        <>
          <input aria-label="Contains" placeholder="Contains…" defaultValue={filter.contains ?? ""} onChange={(e) => onFilter({ ...filter, contains: e.target.value })} className={input} />
          <div className="flex justify-between text-xs">
            <button type="button" className="font-semibold text-[#1a5fe0]" onClick={() => onFilter({ ...filter, values: options.map((o) => o.value) })}>Select all</button>
            <button type="button" className="font-semibold text-tertiary" onClick={() => onFilter({ ...filter, values: [] })}>Clear</button>
          </div>
          <ul className="max-h-48 overflow-auto rounded-md border border-secondary">
            {options.map((option) => (
              <li key={option.value}>
                <label className="flex cursor-pointer items-center gap-2 px-2 py-1 hover:bg-secondary">
                  <input type="checkbox" checked={selected.has(option.value)} onChange={() => toggle(option.value)} />
                  <span className="min-w-0 flex-1 truncate">{option.value || "(empty)"}</span>
                  <span className="text-xs text-tertiary tabular-nums">{option.count.toLocaleString()}</span>
                </label>
              </li>
            ))}
          </ul>
        </>
      )}
      <div className="mt-1 flex flex-col border-t border-secondary pt-2">
        <button type="button" onClick={onPin} className="flex items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-secondary">
          {pinned ? <PinOff className="size-4" /> : <Pin className="size-4" />} {pinned ? "Unpin column" : "Pin column"}
        </button>
        {onChart && (
          <button type="button" onClick={onChart} className="flex items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-secondary">
            <BarChart3 className="size-4" /> Chart this column
          </button>
        )}
      </div>
    </div>
  );
}
