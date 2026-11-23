// Owner ruling: a Professional team's members may be on any plan (Free,
// Professional or Enterprise) but need an Avaloka account; an Enterprise team
// takes any member of the organisation. The server answers per app user with a
// boolean only (getTeamMemberEligibility) -- never the member's plan -- and the
// database enforces the same rule on write (public.team_member_allowed).

export const TEAM_MEMBERS_NEED_ACCOUNT = "Team members need an Avaloka account.";

export type TeamMemberEligibility = Record<string, boolean>;

/** May this app user be added to the caller's team? Fails closed: an unknown
 *  answer (not loaded, lookup failed) means no, except on Enterprise. */
export function canAddTeamMember(
  appUserId: string,
  isEnterprise: boolean,
  eligibility: TeamMemberEligibility | undefined,
): boolean {
  if (isEnterprise) return true;
  return eligibility?.[appUserId] === true;
}
