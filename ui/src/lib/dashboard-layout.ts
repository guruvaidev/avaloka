/** Dashboard grid: row 1 = 3 slots, rows 2–3 = 2 slots each (7 total). */
export const DASHBOARD_SLOT_COUNT = 7;

export type DashboardSlotLayout = {
  position: number;
  colSpan: 2 | 3;
};

export const DASHBOARD_SLOT_LAYOUT: DashboardSlotLayout[] = [
  { position: 0, colSpan: 2 },
  { position: 1, colSpan: 2 },
  { position: 2, colSpan: 2 },
  { position: 3, colSpan: 3 },
  { position: 4, colSpan: 3 },
  { position: 5, colSpan: 3 },
  { position: 6, colSpan: 3 },
];

export function dashboardSlotColClass(colSpan: 2 | 3) {
  return colSpan === 2
    ? "col-span-1 sm:col-span-2 lg:col-span-2"
    : "col-span-1 sm:col-span-2 lg:col-span-3";
}
