-- NEW-BG-01, server side: collaboration and organisation features by plan.
--
-- SUPERSEDES 20261101120000_plan_enforced_org_features.sql (merged with #650 at
-- its pre-revision tip). That migration is left untouched -- it may already be
-- applied -- and this one replaces its functions (CREATE OR REPLACE) and its
-- four "enterprise plan to ..." policies on organization_teams and
-- resource_collaborators (DROP POLICY IF EXISTS), and replaces the policies it
-- left out. It is self-contained: it does not need the earlier file to have run,
-- which matters on Supabase-CLI projects where the duplicate version
-- 20261101120000 may have kept only one of its two files.
--
-- Owner rulings
--   2026-10-31  Free and Professional see no organisation or team features.
--   2026-11-01  All org structure is covered; the app-admin bypass is allowed if
--               every use is audited.
--   2026-11-01  (revision, supersedes the Free/Pro part above) Collaboration is a
--               paid offering and Professional gets referrals:
--                 FREE          shares reports (email results, public links); no
--                               teams; no collaborator invites.
--                 PROFESSIONAL  invites collaborators onto its own resources,
--                               including Free users (the referral path); forms
--                               teams whose members are Professional or higher.
--                 ENTERPRISE    everything: teams with any org member, and
--                               departments, roles, branches and app users.
--               Departments, config_roles, branches and app_users stay
--               Enterprise-only.
--
-- The UI hides what a plan does not include (usePlanFeatures), but a signed-in
-- browser can write to Supabase directly and the permissive policies on these
-- tables carry no plan check. This migration adds one.
--
-- Helpers (SECURITY DEFINER, empty search_path; nothing granted to anon)
--   avaloka_meta.plan_tier_of(uid)   the plan resolver, the same rules as the UI's
--       getMyActivePlan (ui/src/lib/plan.functions.ts): the latest subscription of
--       the user's organisation, else the latest one the user's profile owns;
--       active when trialing/active/trial or inside its trial window;
--       enterprise/professional by plan_type or name; anything else 'free'.
--       Fails closed. Not callable by any API role.
--   public.current_plan_tier()       the caller's tier.
--   public.user_is_pro_or_higher(uid) whether ANOTHER user is Professional or
--       higher -- a boolean only, never the plan, subscription or organisation.
--       Used for the team-member rule.
--   public.org_admin_bypass_active(required)  the caller is an app admin
--       (user_roles 'admin', the SQL side of ADMIN_EMAILS) whose own plan does not
--       meet `required`: exactly the case where only the bypass lets a write in.
--   public.collab_features_enabled() Professional or higher, or the bypass.
--   public.org_features_enabled()    Enterprise, or the bypass.
--   public.team_member_allowed(app_user_id)  Enterprise (org policy applies), or
--       Professional with a member who is Professional or higher, or the bypass.
--
-- Enforcement
--   RESTRICTIVE policies (ANDed with every existing permissive policy, so they
--   hold whichever permissive policy admits the row; existing policies untouched):
--     organization_teams       INSERT, UPDATE   collab_features_enabled()
--     resource_collaborators   INSERT, UPDATE   collab_features_enabled()  (the
--                              inviter's plan; the invitee may be on any plan)
--     departments, config_roles, branches       INSERT, UPDATE   org_features_enabled()
--     app_users                INSERT           org_features_enabled()
--     app_users                UPDATE           collab_features_enabled(), narrowed
--                              by the trigger below
--   Team membership is app_users.team_id (organization_teams). A BEFORE UPDATE
--   trigger on app_users lets a caller below Enterprise change ONLY team_id, and
--   only to a team member who is Professional or higher ("Team members need a
--   Professional plan or higher."). On Enterprise the existing org policies apply.
--
-- Audit of the app-admin bypass
--   avaloka_meta.org_feature_bypass_log, append-only; no API role can reach it.
--   Policies cannot reliably write, so an AFTER INSERT OR UPDATE row trigger on
--   each gated table records a write that succeeded only through the bypass, and
--   the app_users trigger records a team-member rule the bypass overrode. Server
--   functions that write with the service role record through
--   public.log_org_feature_bypass(action, required). Recorded: who (auth user id),
--   what (table and operation, or action name), which organisation, when. No
--   names, emails, row contents or column values.
--
-- Not changed: SELECT and DELETE, so existing rows stay readable and removable; no
-- existing row is modified; share links and email results; org seeding, which
-- runs as the table owner or the service role (no auth.uid(), not subject to RLS).
--
-- Idempotent (CREATE OR REPLACE, IF NOT EXISTS, DROP ... IF EXISTS). Not in the
-- frozen db-migrations.baseline: the chart's migrate hook (db-migrate.sh) runs it
-- once, in one transaction with ON_ERROR_STOP, and records it.

-- avaloka_meta is not exposed through PostgREST; no API role gets USAGE on it.
CREATE SCHEMA IF NOT EXISTS avaloka_meta;
REVOKE ALL ON SCHEMA avaloka_meta FROM PUBLIC;
REVOKE ALL ON SCHEMA avaloka_meta FROM anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION avaloka_meta.plan_tier_of(p_uid uuid)
RETURNS text
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_uid        uuid := p_uid;
  v_profile_id uuid;
  v_org_id     uuid;
  v_status     text;
  v_trial_ends timestamptz;
  v_plan_type  text;
  v_plan_name  text;
  v_found      boolean := false;
  v_active     boolean;
BEGIN
  IF v_uid IS NULL THEN
    RETURN 'free';
  END IF;

  SELECT p.id, p.organization_id
    INTO v_profile_id, v_org_id
    FROM public.profiles p
   WHERE p.user_id = v_uid OR p.id = v_uid
   ORDER BY (p.user_id = v_uid) DESC
   LIMIT 1;

  IF v_org_id IS NOT NULL THEN
    SELECT lower(s.status), s.trial_ends_at, lower(pl.plan_type::text), lower(coalesce(pl.name, ''))
      INTO v_status, v_trial_ends, v_plan_type, v_plan_name
      FROM public.subscriptions s
      LEFT JOIN public.plans pl ON pl.id = s.plan_id
     WHERE s.organization_id = v_org_id
     ORDER BY s.created_at DESC
     LIMIT 1;
    v_found := FOUND;
  END IF;

  -- Personal-scope subscriptions (Professional) are owned by the profile and may
  -- carry no organization_id even when the profile is linked to an organisation.
  IF NOT v_found THEN
    SELECT lower(s.status), s.trial_ends_at, lower(pl.plan_type::text), lower(coalesce(pl.name, ''))
      INTO v_status, v_trial_ends, v_plan_type, v_plan_name
      FROM public.subscriptions s
      LEFT JOIN public.plans pl ON pl.id = s.plan_id
     WHERE s.owner_profile_id = coalesce(v_profile_id, v_uid)
     ORDER BY s.created_at DESC
     LIMIT 1;
    v_found := FOUND;
  END IF;

  IF NOT v_found THEN
    RETURN 'free';
  END IF;

  v_active := v_status IN ('trialing', 'active', 'trial')
    OR (v_trial_ends IS NOT NULL AND v_trial_ends > now()
        AND v_status NOT IN ('canceled', 'cancelled', 'unpaid', 'incomplete_expired'));
  IF NOT coalesce(v_active, false) THEN
    RETURN 'free';
  END IF;

  IF v_plan_type = 'enterprise' OR v_plan_name LIKE '%enterprise%' THEN
    RETURN 'enterprise';
  ELSIF v_plan_type IN ('professional', 'pro') OR v_plan_name LIKE '%professional%' THEN
    RETURN 'professional';
  END IF;
  RETURN 'free';
END
$$;

REVOKE ALL ON FUNCTION avaloka_meta.plan_tier_of(uuid) FROM PUBLIC;

CREATE OR REPLACE FUNCTION avaloka_meta.tier_meets(p_tier text, p_required text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
SET search_path = ''
AS $$
  SELECT CASE p_required
           WHEN 'professional' THEN coalesce(p_tier IN ('professional', 'enterprise'), false)
           WHEN 'enterprise'   THEN coalesce(p_tier = 'enterprise', false)
           ELSE false
         END
$$;
REVOKE ALL ON FUNCTION avaloka_meta.tier_meets(text, text) FROM PUBLIC;

CREATE OR REPLACE FUNCTION public.current_plan_tier()
RETURNS text
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$ SELECT avaloka_meta.plan_tier_of(auth.uid()) $$;
REVOKE ALL ON FUNCTION public.current_plan_tier() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.current_plan_tier() TO authenticated, service_role;

CREATE OR REPLACE FUNCTION public.user_is_pro_or_higher(p_user uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$ SELECT avaloka_meta.tier_meets(avaloka_meta.plan_tier_of(p_user), 'professional') $$;
REVOKE ALL ON FUNCTION public.user_is_pro_or_higher(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.user_is_pro_or_higher(uuid) TO authenticated, service_role;

CREATE OR REPLACE FUNCTION public.org_admin_bypass_active(p_required text DEFAULT 'enterprise')
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT coalesce(
    NOT avaloka_meta.tier_meets(avaloka_meta.plan_tier_of(auth.uid()), p_required)
    AND EXISTS (SELECT 1 FROM public.user_roles r
                 WHERE r.user_id = auth.uid() AND r.role = 'admin'::public.app_role),
    false)
$$;
REVOKE ALL ON FUNCTION public.org_admin_bypass_active(text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.org_admin_bypass_active(text) TO authenticated, service_role;

CREATE OR REPLACE FUNCTION public.collab_features_enabled()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT avaloka_meta.tier_meets(avaloka_meta.plan_tier_of(auth.uid()), 'professional')
    -- APP-ADMIN BYPASS: allowed by the owner (2026-11-01); audited.
    OR public.org_admin_bypass_active('professional')
$$;
REVOKE ALL ON FUNCTION public.collab_features_enabled() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.collab_features_enabled() TO authenticated, service_role;

CREATE OR REPLACE FUNCTION public.org_features_enabled()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT avaloka_meta.tier_meets(avaloka_meta.plan_tier_of(auth.uid()), 'enterprise')
    -- APP-ADMIN BYPASS: allowed by the owner (2026-11-01); audited.
    OR public.org_admin_bypass_active('enterprise')
$$;
REVOKE ALL ON FUNCTION public.org_features_enabled() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.org_features_enabled() TO authenticated, service_role;

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
                       AND au.auth_user_id IS NOT NULL
                       AND public.user_is_pro_or_higher(au.auth_user_id)))
$$;
REVOKE ALL ON FUNCTION public.team_member_allowed(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.team_member_allowed(uuid) TO authenticated, service_role;

-- ---- audit log ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS avaloka_meta.org_feature_bypass_log (
  id              bigserial PRIMARY KEY,
  logged_at       timestamptz NOT NULL DEFAULT now(),
  actor_user_id   uuid NOT NULL,       -- auth.users.id of the admin
  organization_id uuid,                -- the organisation written to
  source          text NOT NULL CHECK (source IN ('rls', 'server')),
  action          text NOT NULL,       -- table name, rule name, or server action
  operation       text NOT NULL        -- INSERT | UPDATE | CALL
);
REVOKE ALL ON TABLE avaloka_meta.org_feature_bypass_log FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON SEQUENCE avaloka_meta.org_feature_bypass_log_id_seq FROM PUBLIC, anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION avaloka_meta.write_bypass_log(p_org uuid, p_source text, p_action text, p_operation text)
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = ''
AS $$
  INSERT INTO avaloka_meta.org_feature_bypass_log (actor_user_id, organization_id, source, action, operation)
  VALUES (auth.uid(), p_org, p_source, p_action, p_operation)
$$;
REVOKE ALL ON FUNCTION avaloka_meta.write_bypass_log(uuid, text, text, text) FROM PUBLIC;

-- Row trigger: TG_ARGV[0] is the plan the table needs ('professional' or
-- 'enterprise'). An app_users UPDATE that changes only team_id is team
-- membership, which needs 'professional'.
CREATE OR REPLACE FUNCTION avaloka_meta.audit_org_feature_bypass()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_required text := TG_ARGV[0];
  v_org      uuid;
BEGIN
  IF auth.uid() IS NULL THEN
    RETURN NULL;   -- table owner / service role: not a signed-in user's write
  END IF;
  IF TG_TABLE_NAME = 'app_users' AND TG_OP = 'UPDATE'
     AND (to_jsonb(NEW) - 'team_id' - 'updated_at') = (to_jsonb(OLD) - 'team_id' - 'updated_at') THEN
    v_required := 'professional';
  END IF;
  IF NOT public.org_admin_bypass_active(v_required) THEN
    RETURN NULL;
  END IF;
  v_org := nullif(to_jsonb(NEW) ->> 'organization_id', '')::uuid;
  IF v_org IS NULL AND TG_TABLE_NAME = 'resource_collaborators' THEN
    SELECT au.organization_id INTO v_org
      FROM public.app_users au
     WHERE au.id = nullif(to_jsonb(NEW) ->> 'app_user_id', '')::uuid;
  END IF;
  PERFORM avaloka_meta.write_bypass_log(v_org, 'rls', TG_TABLE_NAME, TG_OP);
  RETURN NULL;
END
$$;
REVOKE ALL ON FUNCTION avaloka_meta.audit_org_feature_bypass() FROM PUBLIC;

-- ---- team membership rule (app_users.team_id) --------------------------------
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

  -- Adding someone to a team: on a Professional team the member must be
  -- Professional or higher.
  IF NEW.team_id IS NOT NULL AND NEW.team_id IS DISTINCT FROM OLD.team_id
     AND NOT (NEW.auth_user_id IS NOT NULL AND public.user_is_pro_or_higher(NEW.auth_user_id)) THEN
    IF v_admin THEN
      -- The audit trigger records a Free caller's bypass; record the member rule
      -- here only when the caller's own plan would not have been logged there.
      IF avaloka_meta.tier_meets(v_tier, 'professional') THEN
        PERFORM avaloka_meta.write_bypass_log(NEW.organization_id, 'rls', 'app_users.team_member', 'UPDATE');
      END IF;
      RETURN NEW;
    END IF;
    RAISE EXCEPTION 'Team members need a Professional plan or higher.'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END
$$;
REVOKE ALL ON FUNCTION avaloka_meta.enforce_app_users_plan() FROM PUBLIC;

DROP TRIGGER IF EXISTS enforce_app_users_plan ON public.app_users;
CREATE TRIGGER enforce_app_users_plan BEFORE UPDATE ON public.app_users
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.enforce_app_users_plan();

-- ---- policies -------------------------------------------------------------------
-- Teams: Professional or higher.
DROP POLICY IF EXISTS "enterprise plan to add teams" ON public.organization_teams;
DROP POLICY IF EXISTS "enterprise plan to change teams" ON public.organization_teams;
DROP POLICY IF EXISTS "paid plan to add teams" ON public.organization_teams;
CREATE POLICY "paid plan to add teams" ON public.organization_teams
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.collab_features_enabled());
DROP POLICY IF EXISTS "paid plan to change teams" ON public.organization_teams;
CREATE POLICY "paid plan to change teams" ON public.organization_teams
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.collab_features_enabled())
  WITH CHECK (public.collab_features_enabled());

-- Collaborators: the inviter needs Professional or higher; the invitee may be on any plan.
DROP POLICY IF EXISTS "enterprise plan to invite collaborators" ON public.resource_collaborators;
DROP POLICY IF EXISTS "enterprise plan to change collaborators" ON public.resource_collaborators;
DROP POLICY IF EXISTS "paid plan to invite collaborators" ON public.resource_collaborators;
CREATE POLICY "paid plan to invite collaborators" ON public.resource_collaborators
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.collab_features_enabled());
DROP POLICY IF EXISTS "paid plan to change collaborators" ON public.resource_collaborators;
CREATE POLICY "paid plan to change collaborators" ON public.resource_collaborators
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.collab_features_enabled())
  WITH CHECK (public.collab_features_enabled());

-- Org structure: Enterprise.
DROP POLICY IF EXISTS "enterprise plan to add departments" ON public.departments;
CREATE POLICY "enterprise plan to add departments" ON public.departments
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.org_features_enabled());
DROP POLICY IF EXISTS "enterprise plan to change departments" ON public.departments;
CREATE POLICY "enterprise plan to change departments" ON public.departments
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.org_features_enabled())
  WITH CHECK (public.org_features_enabled());

DROP POLICY IF EXISTS "enterprise plan to add roles" ON public.config_roles;
CREATE POLICY "enterprise plan to add roles" ON public.config_roles
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.org_features_enabled());
DROP POLICY IF EXISTS "enterprise plan to change roles" ON public.config_roles;
CREATE POLICY "enterprise plan to change roles" ON public.config_roles
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.org_features_enabled())
  WITH CHECK (public.org_features_enabled());

DROP POLICY IF EXISTS "enterprise plan to add branches" ON public.branches;
CREATE POLICY "enterprise plan to add branches" ON public.branches
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.org_features_enabled());
DROP POLICY IF EXISTS "enterprise plan to change branches" ON public.branches;
CREATE POLICY "enterprise plan to change branches" ON public.branches
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.org_features_enabled())
  WITH CHECK (public.org_features_enabled());

DROP POLICY IF EXISTS "enterprise plan to add app users" ON public.app_users;
CREATE POLICY "enterprise plan to add app users" ON public.app_users
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.org_features_enabled());
-- UPDATE: Professional may change team membership only (enforce_app_users_plan).
DROP POLICY IF EXISTS "enterprise plan to change app users" ON public.app_users;
DROP POLICY IF EXISTS "paid plan to change app users" ON public.app_users;
CREATE POLICY "paid plan to change app users" ON public.app_users
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.collab_features_enabled())
  WITH CHECK (public.collab_features_enabled());

-- ---- audit triggers -----------------------------------------------------------
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.organization_teams;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.organization_teams
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('professional');
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.resource_collaborators;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.resource_collaborators
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('professional');
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.departments;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.departments
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('enterprise');
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.config_roles;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.config_roles
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('enterprise');
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.branches;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.branches
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('enterprise');
DROP TRIGGER IF EXISTS audit_org_feature_bypass ON public.app_users;
CREATE TRIGGER audit_org_feature_bypass AFTER INSERT OR UPDATE ON public.app_users
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.audit_org_feature_bypass('enterprise');

-- ---- server-side audit ----------------------------------------------------------
-- Server functions log their bypass use through this. It writes nothing unless the
-- caller is actually using the bypass for `p_required`, takes no ids from the
-- caller, and accepts only a short action name (never customer data).
DROP FUNCTION IF EXISTS public.log_org_feature_bypass(text);
CREATE OR REPLACE FUNCTION public.log_org_feature_bypass(p_action text, p_required text DEFAULT 'enterprise')
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_org uuid;
BEGIN
  IF p_action IS NULL OR p_action !~ '^[A-Za-z][A-Za-z0-9_.:-]{0,63}$' THEN
    RAISE EXCEPTION 'invalid action name' USING ERRCODE = '22023';
  END IF;
  IF p_required IS NULL OR p_required NOT IN ('professional', 'enterprise') THEN
    RAISE EXCEPTION 'invalid plan level' USING ERRCODE = '22023';
  END IF;
  IF auth.uid() IS NULL OR NOT public.org_admin_bypass_active(p_required) THEN
    RETURN;
  END IF;
  SELECT p.organization_id INTO v_org
    FROM public.profiles p
   WHERE p.user_id = auth.uid() OR p.id = auth.uid()
   ORDER BY (p.user_id = auth.uid()) DESC
   LIMIT 1;
  PERFORM avaloka_meta.write_bypass_log(v_org, 'server', p_action, 'CALL');
END
$$;
REVOKE ALL ON FUNCTION public.log_org_feature_bypass(text, text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.log_org_feature_bypass(text, text) TO authenticated;
