import { useEffect, type ReactNode } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useActivePlan } from "@/lib/use-active-plan";
import { normalizePlan, planIncludes } from "@/lib/edition-comparison";
import type { EditionCapability } from "@/lib/edition-matrix.gen";

/**
 * Guards a route/section so free-plan users cannot access it (even via URL).
 * Redirects to /settings (billing/upgrade) while the plan loads or if free.
 */
export function PlanGate({
  children,
  blockProfessional = false,
  capability,
}: {
  children: ReactNode;
  /** Also block Professional-plan users (Enterprise-only sections). */
  blockProfessional?: boolean;
  /**
   * Gate on this edition-matrix capability instead of on "is Free", so the
   * section and the Billing tab's comparison read the same flag.
   */
  capability?: EditionCapability;
}) {
  const navigate = useNavigate();
  const { data, isLoading } = useActivePlan();
  const tier = data?.plan;
  const plan = normalizePlan(tier);
  const isFree = capability ? plan !== null && !planIncludes(plan, capability) : !!tier && tier === "free";
  const isBlockedPro = blockProfessional && tier === "professional";

  useEffect(() => {
    if (isLoading) return;
    if (isFree) {
      navigate({ to: "/settings", search: { tab: "billing" } as any, replace: true });
    } else if (isBlockedPro) {
      navigate({ to: "/proanalysis", replace: true });
    }
  }, [isLoading, isFree, isBlockedPro, navigate]);

  if (isLoading || isFree || isBlockedPro) return null;
  return <>{children}</>;
}

/**
 * Blocks Professional-plan users only (Free keeps its existing behavior of
 * opening the section with per-action upgrade prompts).
 */
export function NoProfessionalGate({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const { data, isLoading } = useActivePlan();
  const isProfessional = !isLoading && data?.plan === "professional";

  useEffect(() => {
    if (isProfessional) navigate({ to: "/proanalysis", replace: true });
  }, [isProfessional, navigate]);

  if (isProfessional) return null;
  return <>{children}</>;
}
