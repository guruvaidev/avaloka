import { useEffect, useRef, useState } from "react";
import { Plus } from "@untitledui/icons";
import { cx } from "@/lib/utils/cx";
import { InviteCollaboratorsPopover } from "./InviteCollaboratorsPopover";
import { AnalysisInviteCollaboratorsPopover } from "./AnalysisInviteCollaboratorsPopover";
import { usePlanFeatures } from "@/lib/use-plan-features";

interface InviteCollaboratorsTriggerProps {
  projectId?: string | null;
  analysisId?: string | null;
  className?: string;
}

export function InviteCollaboratorsTrigger({
  projectId = null,
  analysisId = null,
  className,
}: InviteCollaboratorsTriggerProps) {
  const [inviteOpen, setInviteOpen] = useState(false);
  const popoverWrapRef = useRef<HTMLDivElement>(null);
  const { canShare } = usePlanFeatures();

  useEffect(() => {
    if (!inviteOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (popoverWrapRef.current?.contains(target)) return;
      if ((target as Element).closest?.(".invite-collaborators-dropdown")) return;
      setInviteOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [inviteOpen]);

  if (!canShare) return null;

  return (
    <div ref={popoverWrapRef} className={cx("relative", className)}>
      <button
        type="button"
        aria-label="Invite collaborators"
        onClick={() => setInviteOpen((v) => !v)}
        className="flex size-10 cursor-pointer items-center justify-center rounded-full border border-brand text-fg-brand-primary hover:bg-brand-primary_alt"
      >
        <Plus className="size-5" />
      </button>
      {analysisId ? (
        <AnalysisInviteCollaboratorsPopover
          open={inviteOpen}
          onClose={() => setInviteOpen(false)}
          analysisId={analysisId}
        />
      ) : (
        <InviteCollaboratorsPopover
          open={inviteOpen}
          onClose={() => setInviteOpen(false)}
          projectId={projectId}
          analysisId={analysisId}
        />
      )}
    </div>
  );
}
