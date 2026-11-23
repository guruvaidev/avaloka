import { createElement, useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { supabase } from "@/integrations/supabase/client";
import { UpgradePlanDialog } from "@/components/dashboard/UpgradeGate";
import { toast } from "sonner";
import { openRayDashboard, rayApi, RayRequestError, stopRayDashboardSessions } from "./ray-dashboard-session";

type Access = "hidden" | "allowed" | "locked";
let access: Access = "hidden";
let initialized = false;
let userId: string | null | undefined;
let revision = 0;
const listeners = new Set<() => void>();
function publish(next: Access) { access = next; for (const listener of listeners) listener(); }
function check(id: string | null) {
  if (id === userId) return;
  userId = id;
  const version = ++revision;
  publish("hidden");
  stopRayDashboardSessions();
  if (!id) return;
  void rayApi.status().then(() => {
    if (version === revision) publish("allowed");
  }).catch((error) => {
    if (version === revision) publish(error instanceof RayRequestError && error.status === 403 && error.reason === "plan_upgrade_required" ? "locked" : "hidden");
  });
}
function initialize() {
  if (initialized) return;
  initialized = true;
  let changed = false;
  supabase.auth.onAuthStateChange((_event, session) => {
    changed = true;
    // Defer work outside the auth callback's lock.
    window.setTimeout(() => check(session?.user.id ?? null), 0);
  });
  void supabase.auth.getSession().then(({ data }) => {
    if (!changed) check(data.session?.user.id ?? null);
  }).catch(() => { if (!changed) check(null); });
}
function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** Shared entitlement: mounting chat links must not generate additional status requests. */
export function useRayDashboardAccess() {
  useEffect(initialize, []);
  const state = useSyncExternalStore(subscribe, () => access, () => "hidden" as Access);
  return { allowed: state === "allowed", locked: state === "locked", visible: state !== "hidden" };
}

export function useRayDashboardLauncher() {
  const entitlement = useRayDashboardAccess();
  const [upgrade, setUpgrade] = useState(false);
  const open = useCallback((jobId?: string) => {
    if (entitlement.locked) { setUpgrade(true); return; }
    if (!entitlement.allowed) return;
    openRayDashboard((error) => {
      if (error instanceof RayRequestError && error.status === 403) { setUpgrade(true); return; }
      if (error instanceof RayRequestError && error.status === 401) {
        toast.error("Your session has expired — please sign in again");
        return;
      }
      toast.error(error instanceof Error ? error.message : "Could not open Ray dashboard. Try again from Avaloka.");
    }, jobId);
  }, [entitlement.allowed, entitlement.locked]);
  return { ...entitlement, open, dialog: createElement(UpgradePlanDialog, { open: upgrade, feature: "Ray dashboard", onClose: () => setUpgrade(false) }) };
}