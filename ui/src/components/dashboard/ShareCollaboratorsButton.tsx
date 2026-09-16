import { useEffect, useRef, useState } from "react";
import { User01 } from "@untitledui/icons";
import { InviteCollaboratorsPopover } from "./InviteCollaboratorsPopover";
import { ReportInviteCollaboratorsPopover } from "./ReportInviteCollaboratorsPopover";
import { AnalysisInviteCollaboratorsPopover } from "./AnalysisInviteCollaboratorsPopover";
import { usePlanFeatures } from "@/lib/use-plan-features";


interface ShareCollaboratorsButtonProps {
  analysisId?: string | null;
  projectId?: string | null;
  reportId?: string | null;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onCollaborationChange?: () => void;
}

export function ShareCollaboratorsButton({
  analysisId = null,
  projectId = null,
  reportId = null,
  open: openProp,
  onOpenChange,
  onCollaborationChange,
}: ShareCollaboratorsButtonProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const open = openProp ?? internalOpen;
  const setOpen = onOpenChange ?? setInternalOpen;
  const wrapRef = useRef<HTMLDivElement>(null);
  const { canShare } = usePlanFeatures();

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (wrapRef.current?.contains(target)) return;
      if ((target as Element).closest?.(".invite-collaborators-dropdown")) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, setOpen]);

  if (!canShare) return null;

  return (
    <div ref={wrapRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-2 rounded-lg bg-[#1565ef] px-3 py-2 text-sm font-semibold text-white shadow-xs hover:bg-[#1257d6]"
      >
        <User01 className="size-4" />
        Share
      </button>
      {reportId ? (
        <ReportInviteCollaboratorsPopover
          open={open}
          onClose={() => setOpen(false)}
          reportId={reportId}
          onCollaborationChange={onCollaborationChange}
        />
      ) : analysisId ? (
        <AnalysisInviteCollaboratorsPopover
          open={open}
          onClose={() => setOpen(false)}
          analysisId={analysisId}
          onCollaborationChange={onCollaborationChange}
        />
      ) : (
        <InviteCollaboratorsPopover
          open={open}
          onClose={() => setOpen(false)}
          analysisId={analysisId}
          projectId={projectId}
          reportId={reportId}
          onCollaborationChange={onCollaborationChange}
        />
      )}

    </div>
  );
}
