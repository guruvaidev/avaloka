import { useCallback, useState, type ReactNode } from "react";
import { useNavigate } from "@tanstack/react-router";
import { Diamond01, X as XIcon } from "@untitledui/icons";
import { Button } from "@/components/base/buttons/button";
import { useActivePlan } from "@/lib/use-active-plan";

/**
 * Free-plan action gate.
 *
 * Free users can now *open* premium sections, but any write/premium action is
 * intercepted with an "Upgrade plan" popup that routes to Settings → Billing.
 *
 * Usage:
 *   const { blocked, dialog } = useUpgradeGate();
 *   <Button onClick={() => { if (blocked("Creating projects")) return; doThing(); }} />
 *   {dialog}
 */
export function useUpgradeGate() {
  const { data, isLoading } = useActivePlan();
  const isFree = !isLoading && (data?.plan ?? "free") === "free";
  const [feature, setFeature] = useState<string | null>(null);

  const blocked = useCallback(
    (featureLabel?: string) => {
      if (!isFree) return false;
      setFeature(featureLabel ?? null);
      return true;
    },
    [isFree],
  );

  const dialog = (
    <UpgradePlanDialog
      open={feature !== null}
      feature={feature}
      onClose={() => setFeature(null)}
    />
  );

  return { isFree, blocked, dialog, openUpgrade: () => setFeature(null) };
}

export function UpgradePlanDialog({
  open,
  feature,
  onClose,
}: {
  open: boolean;
  feature?: string | null;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 p-4">
      <div className="relative w-full max-w-[420px] rounded-2xl border border-secondary bg-primary p-6 shadow-xl">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="absolute right-4 top-4 rounded-md p-1 text-fg-secondary hover:bg-secondary"
        >
          <XIcon className="size-4" />
        </button>

        <div className="grid size-11 place-items-center rounded-full bg-[#eff5ff] text-[#1565ef]">
          <Diamond01 className="size-5" />
        </div>

        <h2 className="mt-4 text-lg font-semibold text-primary">Upgrade your plan</h2>
        <p className="mt-1.5 text-sm text-tertiary">
          {feature ? `${feature} is` : "This feature is"} available on paid plans. Upgrade
          to unlock projects, datasets, insights and team collaboration.
        </p>

        <div className="mt-6 flex justify-end gap-2">
          <Button color="secondary" size="md" onClick={onClose}>
            Not now
          </Button>
          <Button
            color="primary"
            size="md"
            onClick={() => {
              onClose();
              navigate({ to: "/settings", search: { tab: "billing" } as any });
            }}
          >
            Upgrade plan
          </Button>
        </div>
      </div>
    </div>
  );
}

/** Wrapper kept for call sites that only need the dialog imperatively. */
export function UpgradeGateProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
