import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { usePlanFeatures } from "@/lib/use-plan-features";

/** Professional sharing checks. Enterprise keeps its existing module permissions. */
export function useResourceAccess(type: "project" | "analysis", id?: string | null) {
  const { isEnterprise, isProfessional, loading } = usePlanFeatures();
  const [access, setAccess] = useState<{ key: string; level: number } | null>(null);
  const key = `${type}:${id}:${isProfessional}:${isEnterprise}`;
  useEffect(() => {
    if (!id || loading || isEnterprise) return;
    let cancelled = false;
    const refresh = async () => {
      try {
        const { data, error } = await (supabase as any).rpc(`${type}_share_access_level`, { [`p_${type}_id`]: id });
        if (!cancelled) setAccess({ key, level: error ? 0 : Number(data ?? 0) });
      } catch {
        if (!cancelled) setAccess({ key, level: 0 });
      }
    };
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 15_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [type, id, key, loading, isEnterprise, isProfessional]);
  const level = access?.key === key ? access.level : 0;
  return {
    canEdit: isEnterprise || (!loading && level >= 3),
    canComment: isEnterprise || (!loading && isProfessional && level >= 2),
  };
}
