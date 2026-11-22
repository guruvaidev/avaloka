import { useActivePlan } from "@/lib/use-active-plan";

/** Shown where a comment box would be, for a plan without commenting. */
export const COMMENTS_NEED_PRO = "Commenting is included in the Professional and Enterprise plans.";
/** Shown where the analysis share-link controls would be, for a plan without them. */
export const ANALYSIS_LINKS_NEED_PRO =
  "Share links to an analysis are included in the Professional and Enterprise plans. You can still email the results.";

/**
 * Feature availability per plan tier.
 *
 * Owner rulings (NEW-BG-01, revised 2026-11-01; supersedes the 2026-10-31
 * Free/Pro rule): collaboration is a paid offering, and Professional gets
 * referrals.
 *  - Free: can view Teams and Models in Configurations, but cannot create a
 *    team or invite collaborators.
 *  - Professional: invites collaborators onto its own resources, including
 *    Free users (the referral path); forms teams. A team member may be on any
 *    plan, Free, Professional or Enterprise (later owner ruling); creating a
 *    team stays Professional and above (canCreateTeams). Configurations: Models
 *    and Teams.
 *  - Enterprise: everything, including departments, roles, branches and users.
 *  - Commenting, and notifying other people, are Professional and Enterprise
 *    (owner ruling 2026-11-01; app/core/editions.py comment_on_analysis and
 *    notifications). Mentions are sent from the comment box, so canComment
 *    gates them too. Notifications sent TO a Free user still reach them, so the
 *    bell and the notification settings stay on every plan.
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
    /** Teams and Models are visible in Configurations on every plan. */
    canUseTeams: true,
    /** Creating teams and inviting members needs a paid collaboration plan. */
    canCreateTeams: isProfessionalOrHigher,
    /** Creating or managing roles, branches, divisions and users. */
    canManageOrgEntities: isEnterprise,
    /** Share panel access (email results; report public links): every plan. */
    canShare: true,
    /** Creating or changing a share link to an analysis (owner ruling 2026-11-02). */
    canShareAnalysisLink: isProfessionalOrHigher,
    /** Inviting collaborators onto a resource; the invitee may be on any plan. */
    canInviteCollaborators: isProfessionalOrHigher,
    /** Writing or replying to comments on analyses and reports; reading stays open. */
    canComment: isProfessionalOrHigher,
    /** Enterprise gets organization configuration; Free/Professional get Models and Teams. */
    canUseAllConfigSections: isEnterprise,
  };
}
