import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { getMyActivePlan, type ActivePlan } from "@/lib/plan.functions";
import { getCachedAuthUser } from "@/lib/auth-user";

export const ACTIVE_PLAN_KEY = (userId: string | null) =>
  ["active-plan", userId ?? "anon"] as const;

function useAuthUserId() {
  const [id, setId] = useState<string | null>(null);
  useEffect(() => {
    getCachedAuthUser().then((u) => setId(u?.id ?? null));
    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => {
      setId(session?.user?.id ?? null);
    });
    return () => sub.subscription.unsubscribe();
  }, []);
  return id;
}

export function useActivePlan() {
  const userId = useAuthUserId();
  const fetchPlan = useServerFn(getMyActivePlan);
  return useQuery<ActivePlan>({
    queryKey: ACTIVE_PLAN_KEY(userId),
    queryFn: () => fetchPlan(),
    enabled: !!userId,
    staleTime: 60_000,
  });
}

export function useInvalidateActivePlan() {
  const qc = useQueryClient();
  return (userId?: string | null) =>
    qc.invalidateQueries({
      queryKey: userId ? ACTIVE_PLAN_KEY(userId) : ["active-plan"],
    });
}
