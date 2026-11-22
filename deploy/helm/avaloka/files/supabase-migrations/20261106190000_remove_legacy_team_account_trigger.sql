-- Team ownership and admin access now come from owner_profile_id and
-- lead_profile_id. The legacy account-based auto-join trigger references the
-- removed teams.customer_account_id and team_memberships.role columns and
-- prevents every team insert on upgraded installations.
DROP TRIGGER IF EXISTS on_team_created_join_enterprise_admin ON public.teams;
DROP FUNCTION IF EXISTS public.auto_join_enterprise_admin();
