-- Team members may be on any plan; creating a team stays Professional or higher.
--
-- Owner ruling: a user on any plan (Free, Professional or Enterprise) may be
-- invited to and added as a member of a Professional team. Creating a team stays
-- Professional and above, the same rule as the UI's canCreateTeams
-- (ui/src/lib/use-plan-features.ts).
--
-- Replaces, from earlier migrations that are left untouched:
--   avaloka_meta.enforce_team_invitee_plan()   (20261106150000) refused an
--       Enterprise invitee. It now needs only an Avaloka account.
--   public.team_member_allowed(app_user_id)    (20261101140000) needed the member
--       to be Professional or higher. On a Professional team the member now needs
--       only an Avaloka account; on Enterprise any member of the organisation, as
--       before.
--   avaloka_meta.enforce_app_users_plan()      (20261101140000) applied the same
--       member rule to app_users.team_id. It now needs only an account; the rest
--       (below Enterprise only team_id may change) is unchanged.
-- Adds:
--   "pro plan to create teams" on public.teams, a RESTRICTIVE INSERT policy with
--       public.collab_features_enabled() (Professional or higher, or the audited
--       app-admin bypass), the same check organization_teams has. Before this, a
--       signed-in browser on any plan could insert into public.teams directly;
--       only the server function and the UI refused.
--   the audit_org_feature_bypass row trigger on public.teams INSERT, so a team
--       created only through the bypass is recorded, as on organization_teams.
--
-- Not changed: who may invite (the team owner or lead, on Professional or higher);
-- SELECT, UPDATE and DELETE on public.teams; existing rows.
--
-- Idempotent (CREATE OR REPLACE, DROP ... IF EXISTS).

CREATE OR REPLACE FUNCTION avaloka_meta.enforce_team_invitee_plan()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_auth_user_id uuid;
BEGIN
  IF TG_OP = 'UPDATE' AND NEW.status <> 'accepted' THEN
    RETURN NEW;
  END IF;
  IF TG_OP = 'INSERT' AND NEW.invited_profile_id IS NULL AND NEW.status = 'pending' THEN
    RETURN NEW;
  END IF;

  SELECT user_id INTO v_auth_user_id
    FROM public.profiles
   WHERE id = NEW.invited_profile_id;
  IF v_auth_user_id IS NULL THEN
    RAISE EXCEPTION 'The invited user must have an Avaloka account.' USING ERRCODE = '42501';
  END IF;
  -- Any plan may join (owner ruling): no plan check on the invitee.
  RETURN NEW;
END;
$$;
REVOKE ALL ON FUNCTION avaloka_meta.enforce_team_invitee_plan() FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS enforce_team_invitee_plan ON public.team_invitations;
CREATE TRIGGER enforce_team_invitee_plan
  BEFORE INSERT OR UPDATE OF invited_profile_id, status ON public.team_invitations
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.enforce_team_invitee_plan();

CREATE OR REPLACE FUNCTION public.team_member_allowed(p_app_user_id uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT public.org_features_enabled()
    OR (public.collab_features_enabled()
        AND EXISTS (SELECT 1 FROM public.app_users au
                     WHERE au.id = p_app_user_id
                       AND au.auth_user_id IS NOT NULL))
$$;
REVOKE ALL ON FUNCTION public.team_member_allowed(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.team_member_allowed(uuid) TO authenticated, service_role;

CREATE OR REPLACE FUNCTION avaloka_meta.enforce_app_users_plan()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_tier  text;
  v_admin boolean;
BEGIN
  IF auth.uid() IS NULL THEN
    RETURN NEW;    -- table owner / service role (seeding, server functions)
  END IF;
  v_tier := avaloka_meta.plan_tier_of(auth.uid());
  IF avaloka_meta.tier_meets(v_tier, 'enterprise') THEN
    RETURN NEW;    -- Enterprise: the organisation's own policies apply
  END IF;
  v_admin := EXISTS (SELECT 1 FROM public.user_roles r
                      WHERE r.user_id = auth.uid() AND r.role = 'admin'::public.app_role);

  -- Below Enterprise only team membership may change.
  IF (to_jsonb(NEW) - 'team_id' - 'updated_at') IS DISTINCT FROM (to_jsonb(OLD) - 'team_id' - 'updated_at') THEN
    IF v_admin THEN
      RETURN NEW;  -- bypass; the audit trigger records it
    END IF;
    RAISE EXCEPTION 'Departments, roles, branches and users need an Enterprise plan.'
      USING ERRCODE = '42501';
  END IF;

  -- Adding someone to a Professional team: the member may be on any plan, but
  -- needs an Avaloka account.
  IF NEW.team_id IS NOT NULL AND NEW.team_id IS DISTINCT FROM OLD.team_id
     AND NEW.auth_user_id IS NULL THEN
    IF v_admin THEN
      -- The audit trigger records a Free caller's bypass; record the member rule
      -- here only when the caller's own plan would not have been logged there.
      IF avaloka_meta.tier_meets(v_tier, 'professional') THEN
        PERFORM avaloka_meta.write_bypass_log(NEW.organization_id, 'rls', 'app_users.team_member', 'UPDATE');
      END IF;
      RETURN NEW;
    END IF;
    RAISE EXCEPTION 'Team members need an Avaloka account.'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END
$$;
REVOKE ALL ON FUNCTION avaloka_meta.enforce_app_users_plan() FROM PUBLIC;

DROP TRIGGER IF EXISTS enforce_app_users_plan ON public.app_users;
CREATE TRIGGER enforce_app_users_plan BEFORE UPDATE ON public.app_users
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.enforce_app_users_plan();

-- Creating a team: Professional or higher (canCreateTeams).
DROP POLICY IF EXISTS "pro plan to create teams" ON public.teams;
CREATE POLICY "pro plan to create teams" ON public.teams
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.collab_features_enabled());

DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.teams;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT ON public.teams
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('professional');
