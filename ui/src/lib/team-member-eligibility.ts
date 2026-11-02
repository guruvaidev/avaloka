// NEW-BG-01 (owner ruling 2026-11-01): a Professional team's members must be on
// Professional or higher; an Enterprise team takes any member of the
// organisation. The server answers per app user with a boolean only
// (getTeamMemberEligibility) -- never the member's plan -- and the database
// enforces the same rule on write.

export const TEAM_MEMBERS_NEED_PRO = "Team members need a Professional plan or higher.";

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
