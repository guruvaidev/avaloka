import type { DragEvent } from "react";
import { Plus } from "@untitledui/icons";
import { cx } from "@/lib/utils/cx";
import { dashboardSlotColClass } from "@/lib/dashboard-layout";

function SlotChartIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-7 text-[#98a2b3]" aria-hidden>
      <rect x="4" y="12" width="3" height="8" rx="0.5" fill="currentColor" />
      <rect x="10.5" y="8" width="3" height="12" rx="0.5" fill="currentColor" />
      <rect x="17" y="5" width="3" height="15" rx="0.5" fill="currentColor" />
    </svg>
  );
}

type EmptySlotProps = {
  colSpan: 2 | 3;
  dragOver?: boolean;
  saving?: boolean;
  disabled?: boolean;
  onDragOver?: (event: DragEvent<Element>) => void;
  onDragLeave?: () => void;
  onDrop?: (event: DragEvent<Element>) => void;
};

export function DashboardEmptySlot({
  colSpan,
  dragOver,
  saving,
  disabled,
  onDragOver,
  onDragLeave,
  onDrop,
}: EmptySlotProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={cx(
        dashboardSlotColClass(colSpan),
        "group relative flex min-h-[180px] items-center justify-center rounded-xl border-2 border-dashed border-[#d0d5dd] bg-primary/60 transition hover:border-[#1565ef]/60",
        dragOver && "border-[#1565ef] bg-[#1565ef]/5 ring-1 ring-[#1565ef]/30",
        saving && "opacity-60",
      )}
    >
      <div className="relative flex flex-col items-center">
        <div className="grid size-[72px] place-items-center rounded-full bg-[#f2f4f7]">
          <SlotChartIcon />
        </div>
        <span className="absolute bottom-0 grid size-7 translate-y-1/2 place-items-center rounded-full bg-[#475467] text-white shadow-sm group-hover:bg-[#1565ef]">
          <Plus className="size-4" />
        </span>
      </div>
    </button>
  );
}
