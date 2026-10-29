import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import type { FindingGroup } from "../../../lib/multi-dataset-view";

function Collapsible({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-secondary bg-primary">
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-semibold text-primary">
        <span className="min-w-0 flex-1 truncate">{title}</span>
        <ChevronDown className={`size-4 shrink-0 text-tertiary transition-transform motion-reduce:transition-none ${open ? "rotate-180" : ""}`} />
      </button>
      {open ? <div className="border-t border-secondary px-3 py-3">{children}</div> : null}
    </div>
  );
}

/** "Key findings" for multi-dataset analyses: the shared top picks, then everything grouped by sheet behind "View more by sheet". */
export function TopKeyFindings({ picks, groups }: { picks: { text: string; datasetName: string }[]; groups: FindingGroup[] }) {
  const [expanded, setExpanded] = useState(false);
  const nonEmpty = groups.filter((group) => group.findings.length);
  return (
    <div className="my-2" data-testid="top-key-findings">
      <p className="text-sm font-semibold text-primary">Key findings</p>
      <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-primary">
        {picks.map((pick, index) => (
          <li key={index}>
            {pick.text} <span className="text-tertiary">· {pick.datasetName}</span>
          </li>
        ))}
      </ul>
      {nonEmpty.length ? (
        <button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)} className="mt-2 text-xs font-semibold text-brand-secondary hover:underline">
          {expanded ? "View less" : "View more by sheet"}
        </button>
      ) : null}
      {expanded ? (
        <div className="mt-2 space-y-2">
          {nonEmpty.map((group) => (
            <Collapsible key={group.id} title={group.name}>
              <ul className="list-disc space-y-1 pl-5 text-sm text-primary">
                {group.findings.map((text, index) => <li key={index}>{text}</li>)}
              </ul>
            </Collapsible>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export type ChartGroupView = { id: string; name: string; count: number; emptyText: string; cards: ReactNode };

/** Chart tab for multi-dataset analyses: top charts first, every chart grouped by sheet behind "View all charts by sheet". */
export function TopChartsSection({ heading, topCards, groups, total }: { heading: string; topCards: ReactNode; groups: ChartGroupView[]; total: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="space-y-3" data-testid="top-charts-section">
      <h3 className="text-sm font-semibold text-primary">{heading}</h3>
      {topCards}
      <button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)} className="text-xs font-semibold text-brand-secondary hover:underline">
        {expanded ? "View less" : `View all charts by sheet (${total})`}
      </button>
      {expanded ? (
        <div className="space-y-2">
          {groups.map((group) => (
            <Collapsible key={group.id} title={`${group.name} · ${group.count} chart${group.count === 1 ? "" : "s"}`}>
              {group.count ? group.cards : <p className="text-sm text-tertiary">{group.emptyText}</p>}
            </Collapsible>
          ))}
        </div>
      ) : null}
    </div>
  );
}
