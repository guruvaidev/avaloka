import { useQuery } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { supabase } from "@/integrations/supabase/client";
import { listAppUsers } from "@/lib/configurations.functions";
import { getCachedAuthUser } from "@/lib/auth-user";

export type ModulePerms = {
  canView: boolean;
  canCreate: boolean;
  canEdit: boolean;
  canApprove: boolean;
  loading: boolean;
};

export type UserMgmtPerms = ModulePerms;

const findKey = (obj: any, key: string): string | undefined =>
  Object.keys(obj ?? {}).find((k) => k.toLowerCase() === key.toLowerCase());

/**
 * Returns the current authenticated user's permissions for a given module
 * (case-insensitive). If no app_users row is found for the user or no
 * permissions are configured, treats the user as privileged (full access).
 */
export function useModulePerms(moduleName: string): ModulePerms {
  const listAppUsersFn = useServerFn(listAppUsers);

  const uidQ = useQuery({
    queryKey: ["auth", "uid"],
    queryFn: async () => (await getCachedAuthUser())?.id ?? null,
    staleTime: 5 * 60_000,
  });

  const usersQ = useQuery({
    queryKey: ["config", "users"],
    queryFn: async () => (await listAppUsersFn()) as any[],
    enabled: !!uidQ.data,
    staleTime: 5 * 60_000,
  });

  const loading = uidQ.isLoading || usersQ.isLoading;
  const authUid = uidQ.data;
  const users = usersQ.data ?? [];
  const me = authUid ? users.find((u: any) => u.auth_user_id === authUid) : null;

  const perms = (me?.permissions ?? null) as Record<
    string,
    Record<string, boolean>
  > | null;

  // No app_users row or no permissions set → treat as privileged.
  const privileged = !me || !perms;

  const modKey = perms ? findKey(perms, moduleName) : undefined;
  const mod = modKey && perms ? perms[modKey] : {};
  const val = (name: string): boolean => {
    const k = findKey(mod, name);
    return k ? Boolean((mod as any)[k]) : false;
  };

  return {
    canView: privileged || val("View"),
    canCreate: privileged || val("Create"),
    canEdit: privileged || val("Edit"),
    canApprove: privileged || val("Approve"),
    loading,
  };
}

export function useUserMgmtPerms(): UserMgmtPerms {
  return useModulePerms("User Management");
}

export function useProjectsPerms(): ModulePerms {
  return useModulePerms("Projects");
}

export function useDashboardPerms(): ModulePerms {
  return useModulePerms("Dashboard");
}
