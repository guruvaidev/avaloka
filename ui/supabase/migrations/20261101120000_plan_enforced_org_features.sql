-- NEW-BG-01, server side: organisation and team features need an Enterprise plan.
--
-- Owner decision 2026-10-31: Free and Professional users see no organisation or
-- team features. The UI hides them (usePlanFeatures), but a signed-in browser can
-- write to Supabase directly, and the permissive policies on organization_teams
-- and resource_collaborators carry no plan check. This migration adds one.
--
-- How
--   * public.current_plan_tier() resolves the caller's plan the same way the UI's
--     canonical resolver does (ui/src/lib/plan.functions.ts getMyActivePlan): the
--     latest subscription of the caller's organisation, else the latest one owned
--     by the caller's profile; active when trialing/active/trial or still inside
--     its trial window; enterprise/professional by plan_type or plan name. Anything
--     else -- no profile, no subscription, an inactive or unrecognised plan -- is
--     'free'. It fails closed.
--   * public.org_features_enabled() is true for an Enterprise tier. It also has a
--     single, labelled "support-desk admin bypass" clause (user_roles 'admin', the
--     SQL side of ADMIN_EMAILS, whom the UI's resolver treats as Enterprise); that
--     clause is under review with the owner and can be deleted on its own.
--   * RESTRICTIVE policies for INSERT and UPDATE on both tables. A restrictive
--     policy is ANDed with every permissive one, so the check holds whichever of
--     the existing permissive INSERT/UPDATE policies would have let the row in
--     (there are several on each table, accumulated over earlier migrations).
--     The existing policies are not touched.
--
-- Not changed: SELECT and DELETE. Rows a Free or Professional organisation already
-- has stay readable and removable. No row is modified. Email results and public
-- links (resource_share_links, the send-*-email functions) are not org features
-- and are not affected. service_role (server functions using the admin client)
-- bypasses RLS; those functions check org_features_enabled() themselves.
--
-- Idempotent (CREATE OR REPLACE, DROP POLICY IF EXISTS). Not in the frozen
-- db-migrations.baseline: the chart's migrate hook (db-migrate.sh) runs it once,
-- in one transaction with ON_ERROR_STOP, and records it.

CREATE OR REPLACE FUNCTION public.current_plan_tier()
RETURNS text
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_uid        uuid := auth.uid();
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

REVOKE ALL ON FUNCTION public.current_plan_tier() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.current_plan_tier() TO authenticated, service_role;

CREATE OR REPLACE FUNCTION public.org_features_enabled()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT coalesce(
    public.current_plan_tier() = 'enterprise'
    -- SUPPORT-DESK ADMIN BYPASS (under review with the owner). This one OR clause
    -- is the whole bypass; delete it to remove it.
    OR EXISTS (SELECT 1 FROM public.user_roles r
                WHERE r.user_id = auth.uid() AND r.role = 'admin'::public.app_role),
    false)
$$;

REVOKE ALL ON FUNCTION public.org_features_enabled() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.org_features_enabled() TO authenticated, service_role;

-- Teams: creating or changing a team needs an Enterprise plan.
DROP POLICY IF EXISTS "enterprise plan to add teams" ON public.organization_teams;
CREATE POLICY "enterprise plan to add teams" ON public.organization_teams
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.org_features_enabled());

DROP POLICY IF EXISTS "enterprise plan to change teams" ON public.organization_teams;
CREATE POLICY "enterprise plan to change teams" ON public.organization_teams
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.org_features_enabled())
  WITH CHECK (public.org_features_enabled());

-- Collaborators: inviting someone, or changing their access, needs an Enterprise plan.
DROP POLICY IF EXISTS "enterprise plan to invite collaborators" ON public.resource_collaborators;
CREATE POLICY "enterprise plan to invite collaborators" ON public.resource_collaborators
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.org_features_enabled());

DROP POLICY IF EXISTS "enterprise plan to change collaborators" ON public.resource_collaborators;
CREATE POLICY "enterprise plan to change collaborators" ON public.resource_collaborators
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.org_features_enabled())
  WITH CHECK (public.org_features_enabled());
