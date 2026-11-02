import { useActivePlan } from "@/lib/use-active-plan";

/**
 * Feature availability per plan tier.
 *
 * Owner rulings (NEW-BG-01, revised 2026-11-01; supersedes the 2026-10-31
 * Free/Pro rule): collaboration is a paid offering, and Professional gets
 * referrals.
 *  - Free: shares reports (email results + public link). No teams, no
 *    collaborator invites.
 *  - Professional: invites collaborators onto its own resources, including
 *    Free users (the referral path); forms teams whose members are Professional
 *    or higher. Configurations: Models and Teams.
 *  - Enterprise: everything, including departments, roles, branches and users.
 * Every flag fails closed: until the plan has loaded, or with no plan, nobody
 * gets a paid feature.
 *
 * Professional data is never deleted when a plan lapses -- only access is
 * disabled, so upgrading again restores everything.
 */
export function usePlanFeatures() {
  const { data, isLoading } = useActivePlan();
  const tier = data?.plan ?? "free";
  const isProfessional = !isLoading && tier === "professional";
  const isEnterprise = !isLoading && tier === "enterprise";
  const isProfessionalOrHigher = isProfessional || isEnterprise;

  return {
    loading: isLoading,
    tier,
    isProfessional,
    isEnterprise,
    /** Projects section (rail item + /dashboard route). */
    canUseProjects: true,
    /** Teams (Professional teams take Professional-or-higher members only). */
    canUseTeams: isProfessionalOrHigher,
    /** Creating or managing roles, branches, divisions and users. */
    canManageOrgEntities: isEnterprise,
    /** Share panel access (email results + public link): every plan. */
    canShare: true,
    /** Inviting collaborators onto a resource; the invitee may be on any plan. */
    canInviteCollaborators: isProfessionalOrHigher,
    /** All configuration sections; Professional gets Models and Teams. */
    canUseAllConfigSections: !isProfessional,
  };
}
