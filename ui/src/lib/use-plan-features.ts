import { useActivePlan } from "@/lib/use-active-plan";

/**
 * Feature availability per plan tier.
 *
 * Professional plan (data is never deleted — only access is disabled, so
 * upgrading back to Enterprise restores everything):
 *  - Configurations: Models section only (view)
 *  - No creating/managing Roles, Branches, Divisions, Users or Teams
 *  - Pro Analysis, Projects and Project Analysis are available
 *  - Sharing limited to email results + public link (no collaborator/team invites)
 *
 * Free-plan behavior is unchanged (handled by PlanGate / useUpgradeGate).
 */
export function usePlanFeatures() {
  const { data, isLoading } = useActivePlan();
  const tier = data?.plan ?? "free";
  const isProfessional = !isLoading && tier === "professional";
  const isEnterprise = tier === "enterprise";

  return {
    loading: isLoading,
    tier,
    isProfessional,
    isEnterprise,
    /** Projects section (rail item + /dashboard route). */
    canUseProjects: true,
    /** Teams / team members configuration. */
    canUseTeams: !isProfessional,
    /** Creating or managing roles, branches, divisions, users, teams. */
    canManageOrgEntities: !isProfessional,
    /** Share panel access (email results + public link). */
    canShare: true,
    /** Inviting team members/teams as collaborators on a resource. */
    canInviteCollaborators: !isProfessional,
    /** All configuration sections; professional gets Models only. */
    canUseAllConfigSections: !isProfessional,
  };
}
