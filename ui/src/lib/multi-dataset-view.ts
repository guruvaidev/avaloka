/** Pure presentation helpers for analyses with several selected sheets/files. No requests. */
import { normalizeFinding } from "./workspace-findings";

/** More than 4 chips → first 3 plus a "+{n} more" chip, unless expanded. */
export function visibleChips<T>(chips: T[], expanded: boolean): { shown: T[]; hidden: number } {
  if (expanded || chips.length <= 4) return { shown: chips, hidden: 0 };
  return { shown: chips.slice(0, 3), hidden: chips.length - 3 };
}

/** Workbook sheets are labelled by their sheet name; files by their name. */
export function chipLabel(chip: { name: string; sheetName?: string | null; sourceWorkbook?: string | null }): string {
  return chip.sourceWorkbook && chip.sheetName ? chip.sheetName : chip.name;
}

/** The selection after toggling one chip; the last selected chip cannot be switched off. */
export function toggleSelection(selected: string[], id: string): string[] {
  if (selected.includes(id)) return selected.length <= 1 ? selected : selected.filter((x) => x !== id);
  return [...selected, id];
}

export type FindingGroup = { id: string; name: string; findings: string[] };

/** Findings per group with normalised dedupe across all groups, so nothing is duplicated. */
export function dedupeFindingGroups(groups: FindingGroup[]): FindingGroup[] {
  const seen = new Set<string>();
  return groups.map((group) => ({
    ...group,
    findings: group.findings.filter((text) => {
      const key = normalizeFinding(text);
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    }),
  }));
}

/** Text shown when a selected sheet has no charts. */
export function noChartsText(chartsLeftOut: unknown): string {
  const reason = Array.isArray(chartsLeftOut) ? chartsLeftOut[0]?.reason : null;
  return typeof reason === "string" && reason.trim() ? `No charts: ${reason.trim()}` : "No charts for this sheet.";
}
