import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";
import { claimPendingProfessionalTeamInvitations } from "@/lib/configurations.functions";

/** Reconcile invites even when signup redirects past the login page. */
export function TeamInvitationClaim() {
  const [user, setUser] = useState<Session["user"] | null>(null);
  const claim = useServerFn(claimPendingProfessionalTeamInvitations);
  const queryClient = useQueryClient();

  useEffect(() => {
    let active = true;
    // Auth callbacks only update state; requests run after the auth lock releases.
    const { data: subscription } = supabase.auth.onAuthStateChange((_event, session) => {
      if (active) setUser(session?.user ?? null);
    });
    void supabase.auth.getSession().then(({ data }) => {
      if (active) setUser(data.session?.user ?? null);
    });
    return () => {
      active = false;
      subscription.subscription.unsubscribe();
    };
  }, []);

  const result = useQuery({
    // Email verification can complete without changing the signed-in user ID.
    // Include it in the key so an earlier unverified result cannot mask signup.
    queryKey: ["claim-team-invitations", user?.id, user?.email, user?.email_confirmed_at],
    enabled: Boolean(user?.id && user.email_confirmed_at),
    queryFn: () => claim(),
    retry: 3,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (!result.data?.attached) return;
    void queryClient.invalidateQueries({ queryKey: ["config", "teams"] });
    window.dispatchEvent(new Event("app-notifications:refresh"));
  }, [queryClient, result.data, result.dataUpdatedAt]);

  return null;
}
