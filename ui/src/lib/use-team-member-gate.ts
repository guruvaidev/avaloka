import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { getTeamMemberEligibility } from "@/lib/configurations.functions";
import { usePlanFeatures } from "@/lib/use-plan-features";
import { TEAM_MEMBERS_NEED_ACCOUNT, canAddTeamMember, type TeamMemberEligibility } from "@/lib/team-member-eligibility";

/**
 * The team-member picker's rule (owner ruling): on a Professional team a member
 * on any plan may be added, as long as they have an Avaloka account; on
 * Enterprise any member of the organisation. `allowed(id)` fails closed until
 * the server has answered; `tryAdd(id)` returns false and sets the message when
 * the member may not be added.
 */
export function useTeamMemberGate(open: boolean, candidateIds: string[]) {
  const { isEnterprise } = usePlanFeatures();
  const eligibilityFn = useServerFn(getTeamMemberEligibility);
  const ids = [...candidateIds].sort();
  const q = useQuery<TeamMemberEligibility>({
    queryKey: ["config", "team-member-eligibility", ids],
    queryFn: async () => (await eligibilityFn({ data: { appUserIds: ids } })) as TeamMemberEligibility,
    enabled: open && !isEnterprise && ids.length > 0,
  });
  const [error, setError] = useState<string | null>(null);
  const allowed = (id: string) => canAddTeamMember(id, isEnterprise, q.data);
  const tryAdd = (id: string) => {
    if (allowed(id)) {
      setError(null);
      return true;
    }
    setError(TEAM_MEMBERS_NEED_ACCOUNT);
    return false;
  };
  return { allowed, tryAdd, error, clearError: () => setError(null) };
}
