-- D-2: the app 'admin' role reaches no other tenant's customer data, and no user
-- reads another tenant's profile rows.
--
-- Owner rulings (2026-11-01)
--   * The 'admin' role in public.user_roles (granted from ADMIN_EMAILS, and the
--     role is_support_admin() checks) keeps support's own data: support tickets
--     and their messages. Customer resources in other organisations are not
--     readable or editable through it. Platform-staff actions will go through
--     separate audited RPCs.
--   * Profile reads are scoped to: self, own organisation, and report
--     collaborators. Display fields only (name, avatar) where the UI shows other
--     people; no email across tenants.
--
-- What changes
--   C1  reports            "collaborators_can_read_shared_reports" loses
--                          `OR has_role(auth.uid(), 'admin')` (every report).
--   C2  get_my_report_access()  loses its admin branch, which returned 'edit' on
--                          every report and, through it, on every report's
--                          comments. Search path is now empty.
--   C3  resource_collaborators "report_owner_admin_read_collaborators" loses the
--                          admin term (who is shared on any report).
--   C4  get_report_teams() loses the admin term, and returns a member's email
--                          only when that member's roster row is in the
--                          caller's own organisation. Elsewhere the email is
--                          NULL and the name falls back to 'Unknown', not the
--                          email.
--   C5  prevent_profile_org_change()  no longer lets an admin move its own
--                          profile into any organisation (after which
--                          current_org_id() and is_org_admin() made every
--                          own-org admin policy apply there). Only a caller with
--                          no end user (service role, migrations) may change
--                          profiles.organization_id.
--   C6  user_roles         "admins manage roles" is dropped: an admin can no
--                          longer grant or revoke roles from the browser. Roles
--                          come from configuration (the migrate hook's
--                          ADMIN_EMAILS bootstrap, service role). "users read own
--                          roles" stays.
--   C13 current_org_id()   no longer matches an app_users row by the caller's
--                          profiles.company_email, a field the caller writes
--                          itself: any user without an organisation could name a
--                          member's address there and take that organisation as
--                          its own. Matching is by profile id or auth user id
--                          only. Otherwise unchanged.
--   P   profiles           "Authenticated users can read profiles" USING (true)
--                          is dropped: every signed-in user could read every
--                          user's profile row (phone, company and recovery email,
--                          billing ids). Row reads stay: own profile; org admins
--                          of their own organisation ("Admins read org
--                          profiles"); creators of reports one can access in
--                          one's own organisation. Other people are shown
--                          through public.get_display_profiles(ids), which
--                          returns id, user_id, names and avatar only, for
--                          people related to the caller: same organisation, or
--                          on a report, analysis or project the caller takes
--                          part in.
--
-- Kept on purpose (own organisation only once C5 and C13 hold, or support's own
-- data): is_org_admin() and the "org admins ..." policies; "Owner or admin can
-- update their organization"; "Admins read org profiles"; "Creators or admins
-- delete reports"; is_support_admin() and the support_tickets /
-- support_ticket_messages policies; internal_roles (dormant); the plan helpers.
--
-- Forward only, not in db-migrations.baseline. The chart's migrate hook runs it
-- once in one transaction with ON_ERROR_STOP. Idempotent (DROP ... IF EXISTS,
-- CREATE OR REPLACE).

-- ---- C1 reports -------------------------------------------------------------
DROP POLICY IF EXISTS "collaborators_can_read_shared_reports" ON public.reports;
CREATE POLICY "collaborators_can_read_shared_reports"
ON public.reports
FOR SELECT
TO authenticated
USING (
  created_by = auth.uid()
  OR public.get_my_report_access(id) IN ('view', 'comment', 'edit')
);

-- ---- C2 get_my_report_access ----------------------------------------------------
-- As 20261013130000, without the admin override.
CREATE OR REPLACE FUNCTION public.get_my_report_access(_report_id uuid)
RETURNS text
LANGUAGE plpgsql
VOLATILE                      -- the backfill below is an UPDATE
SECURITY DEFINER
SET search_path = ''
AS $function$
DECLARE
  _auth uuid := auth.uid();
  _auth_email text;
  _profile uuid;
  _created_by uuid;
  _analysis_id uuid;
  _project_id uuid;
  _level text;
BEGIN
  IF _auth IS NULL THEN
    RETURN 'none';
  END IF;

  SELECT r.created_by, r.analysis_id, r.project_id
    INTO _created_by, _analysis_id, _project_id
    FROM public.reports r WHERE r.id = _report_id;

  IF NOT FOUND THEN
    RETURN 'none';
  END IF;

  SELECT u.email INTO _auth_email FROM auth.users u WHERE u.id = _auth LIMIT 1;

  SELECT p.id INTO _profile FROM public.profiles p WHERE p.user_id = _auth LIMIT 1;
  IF _profile IS NULL THEN
    SELECT p.id INTO _profile FROM public.profiles p WHERE p.id = _auth LIMIT 1;
  END IF;

  IF _created_by = _auth OR (_profile IS NOT NULL AND _created_by = _profile) THEN
    RETURN 'edit';
  END IF;

  -- Backfill a missing auth_user_id link (unchanged).
  IF _auth_email IS NOT NULL THEN
    UPDATE public.app_users
       SET auth_user_id = _auth, updated_at = now()
     WHERE auth_user_id IS NULL
       AND lower(email) = lower(_auth_email);
  END IF;

  SELECT rc.access_level INTO _level
    FROM public.resource_collaborators rc
    JOIN public.app_users au ON au.id = rc.app_user_id
   WHERE (
       au.auth_user_id = _auth
       OR (_auth_email IS NOT NULL AND lower(au.email) = lower(_auth_email))
       OR (_profile IS NOT NULL AND au.profile_id = _profile)
     )
     AND (
       (rc.resource_type = 'report' AND rc.resource_id = _report_id)
       OR (rc.resource_type = 'analysis' AND rc.resource_id = _analysis_id)
       OR (_project_id IS NOT NULL AND rc.resource_type = 'project' AND rc.resource_id = _project_id)
     )
   ORDER BY CASE lower(rc.access_level)
     WHEN 'edit' THEN 1
     WHEN 'comment' THEN 2
     WHEN 'view' THEN 3
     ELSE 4 END
   LIMIT 1;

  RETURN coalesce(lower(_level), 'none');
END;
$function$;
REVOKE ALL ON FUNCTION public.get_my_report_access(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.get_my_report_access(uuid) TO authenticated, service_role;

-- ---- C3 resource_collaborators ---------------------------------------------------
DROP POLICY IF EXISTS "report_owner_admin_read_collaborators" ON public.resource_collaborators;
CREATE POLICY "report_owner_admin_read_collaborators"
ON public.resource_collaborators
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.reports r
     WHERE r.id = resource_collaborators.resource_id
       AND resource_collaborators.resource_type = 'report'
       AND r.created_by = auth.uid()
  )
);

-- ---- C4 get_report_teams ----------------------------------------------------------
-- Same signature and rows. Names come from to_jsonb(profile) so the function does
-- not depend on which optional name columns a deployment's profiles table has.
CREATE OR REPLACE FUNCTION public.get_report_teams(_report_ids uuid[])
RETURNS TABLE (
  report_id uuid,
  member_id uuid,
  name text,
  avatar_url text,
  email text
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  _org uuid := public.current_org_id();
BEGIN
  IF auth.uid() IS NULL THEN
    RETURN;
  END IF;
  RETURN QUERY
  WITH accessible AS (
    SELECT r.id AS report_id, r.analysis_id, r.project_id, r.created_by
      FROM public.reports r
     WHERE r.id = ANY(_report_ids)
       AND (
         r.created_by = auth.uid()
         OR public.is_resource_collaborator('report', r.id)
         OR public.is_resource_collaborator('analysis', r.analysis_id)
         OR (r.project_id IS NOT NULL AND public.is_resource_collaborator('project', r.project_id))
       )
  ),
  creators AS (
    SELECT a.report_id,
           coalesce(p.id, a.created_by) AS member_id,
           p.j AS j,
           p.avatar_url,
           CASE WHEN au_lookup.organization_id = _org THEN au_lookup.email END AS email
      FROM accessible a
      LEFT JOIN LATERAL (
        SELECT pp.id, pp.avatar_url, to_jsonb(pp) AS j FROM public.profiles pp
         WHERE pp.id = a.created_by OR pp.user_id = a.created_by
         LIMIT 1
      ) p ON true
      LEFT JOIN LATERAL (
        SELECT x.email, x.organization_id FROM public.app_users x
         WHERE x.auth_user_id = a.created_by OR x.id = a.created_by
         ORDER BY (x.organization_id = _org) DESC NULLS LAST
         LIMIT 1
      ) au_lookup ON true
  ),
  collabs AS (
    SELECT a.report_id,
           au.id AS member_id,
           p.j AS j,
           p.avatar_url,
           CASE WHEN au.organization_id = _org THEN au.email END AS email
      FROM accessible a
      JOIN public.resource_collaborators rc
        ON (rc.resource_type = 'report'   AND rc.resource_id = a.report_id)
        OR (rc.resource_type = 'analysis' AND rc.resource_id = a.analysis_id)
        OR (a.project_id IS NOT NULL AND rc.resource_type = 'project' AND rc.resource_id = a.project_id)
      JOIN public.app_users au ON au.id = rc.app_user_id
      LEFT JOIN LATERAL (
        SELECT pp.avatar_url, to_jsonb(pp) AS j FROM public.profiles pp
         WHERE (au.auth_user_id IS NOT NULL AND (pp.id = au.auth_user_id OR pp.user_id = au.auth_user_id))
            OR (au.profile_id IS NOT NULL AND pp.id = au.profile_id)
         LIMIT 1
      ) p ON true
  )
  SELECT DISTINCT ON (t.report_id, t.member_id)
         t.report_id, t.member_id,
         coalesce(nullif(t.j ->> 'full_name', ''),
                  nullif(btrim(concat_ws(' ', t.j ->> 'first_name', t.j ->> 'last_name')), ''),
                  t.email, 'Unknown') AS name,
         t.avatar_url, t.email
    FROM (SELECT * FROM creators UNION ALL SELECT * FROM collabs) t
   ORDER BY t.report_id, t.member_id, 3;
END;
$$;
REVOKE ALL ON FUNCTION public.get_report_teams(uuid[]) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.get_report_teams(uuid[]) TO authenticated;

-- ---- C5 profile organisation ------------------------------------------------------
CREATE OR REPLACE FUNCTION public.prevent_profile_org_change()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $function$
DECLARE
  _role text := coalesce(
    nullif(current_setting('request.jwt.claim.role', true), ''),
    nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'role',
    '');
BEGIN
  IF NEW.organization_id IS DISTINCT FROM OLD.organization_id THEN
    -- Server-side code (service role) and migrations have no end user. Every
    -- signed-in caller, admin or not, is refused.
    IF _role = 'service_role' OR auth.uid() IS NULL THEN
      RETURN NEW;
    END IF;
    RAISE EXCEPTION 'organization_id cannot be modified' USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$function$;

-- ---- C6 user_roles -------------------------------------------------------------------
DROP POLICY IF EXISTS "admins manage roles" ON public.user_roles;

-- ---- C13 current_org_id ----------------------------------------------------------------
-- As 20260821145608, without the company_email match.
CREATE OR REPLACE FUNCTION public.current_org_id()
RETURNS uuid
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  _pid uuid;
  _org uuid;
BEGIN
  _pid := public.current_profile_id();
  IF _pid IS NULL THEN
    RETURN NULL;
  END IF;

  SELECT p.organization_id INTO _org FROM public.profiles p WHERE p.id = _pid LIMIT 1;
  IF _org IS NOT NULL THEN
    RETURN _org;
  END IF;

  -- Fallback 1: an organisation the user owns.
  SELECT o.id INTO _org FROM public.organizations o WHERE o.owner_profile_id = _pid LIMIT 1;
  IF _org IS NOT NULL THEN
    RETURN _org;
  END IF;

  -- Fallback 2: a roster row linked to this profile or auth user.
  SELECT au.organization_id INTO _org
    FROM public.app_users au
   WHERE au.organization_id IS NOT NULL
     AND (au.profile_id = _pid OR au.auth_user_id = auth.uid())
   LIMIT 1;

  RETURN _org;
END;
$$;
REVOKE ALL ON FUNCTION public.current_org_id() FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.current_org_id() TO authenticated, service_role;

-- ---- P profiles -------------------------------------------------------------------------
DROP POLICY IF EXISTS "Authenticated users can read profiles" ON public.profiles;

-- Display fields of the given people, for those related to the caller. Accepts
-- profile ids or auth user ids. Never returns email, phone or billing columns.
CREATE OR REPLACE FUNCTION public.get_display_profiles(p_ids uuid[])
RETURNS TABLE (
  id uuid,
  user_id uuid,
  full_name text,
  first_name text,
  last_name text,
  avatar_url text
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  WITH me AS (
    SELECT auth.uid() AS uid, public.current_profile_id() AS pid, public.current_org_id() AS org
  ),
  ids AS (
    SELECT DISTINCT x FROM unnest((coalesce(p_ids, '{}'::uuid[]))[1:500]) AS x WHERE x IS NOT NULL
  ),
  -- Resources the caller takes part in.
  mine AS (
    SELECT rc.resource_type, rc.resource_id
      FROM public.resource_collaborators rc
      JOIN public.app_users au ON au.id = rc.app_user_id
      CROSS JOIN me
     WHERE au.auth_user_id = me.uid OR (me.pid IS NOT NULL AND au.profile_id = me.pid)
    UNION SELECT 'report', r.id FROM public.reports r CROSS JOIN me
     WHERE r.created_by = me.uid OR (me.pid IS NOT NULL AND r.created_by = me.pid)
    UNION SELECT 'analysis', a.id FROM public.analyses a CROSS JOIN me
     WHERE me.pid IS NOT NULL AND a.owner_id = me.pid
    UNION SELECT 'project', pr.id FROM public.projects pr CROSS JOIN me
     WHERE me.pid IS NOT NULL AND pr.owner_id = me.pid
  ),
  -- A report brings in the people on its analysis and project, as get_report_teams does.
  scope AS (
    SELECT resource_type, resource_id FROM mine
    UNION SELECT 'analysis', r.analysis_id FROM mine m JOIN public.reports r
      ON m.resource_type = 'report' AND r.id = m.resource_id WHERE r.analysis_id IS NOT NULL
    UNION SELECT 'project', r.project_id FROM mine m JOIN public.reports r
      ON m.resource_type = 'report' AND r.id = m.resource_id WHERE r.project_id IS NOT NULL
  ),
  people AS (
    SELECT au.auth_user_id AS ref FROM scope s
      JOIN public.resource_collaborators rc ON rc.resource_type = s.resource_type AND rc.resource_id = s.resource_id
      JOIN public.app_users au ON au.id = rc.app_user_id
    UNION SELECT au.profile_id FROM scope s
      JOIN public.resource_collaborators rc ON rc.resource_type = s.resource_type AND rc.resource_id = s.resource_id
      JOIN public.app_users au ON au.id = rc.app_user_id
    UNION SELECT r.created_by FROM scope s JOIN public.reports r ON s.resource_type = 'report' AND r.id = s.resource_id
    UNION SELECT r.created_by FROM scope s JOIN public.reports r
      ON (s.resource_type = 'analysis' AND r.analysis_id = s.resource_id)
      OR (s.resource_type = 'project' AND r.project_id = s.resource_id)
    UNION SELECT a.owner_id FROM scope s JOIN public.analyses a ON s.resource_type = 'analysis' AND a.id = s.resource_id
    UNION SELECT pr.owner_id FROM scope s JOIN public.projects pr ON s.resource_type = 'project' AND pr.id = s.resource_id
  )
  SELECT p.id, p.user_id,
         nullif(to_jsonb(p) ->> 'full_name', ''),
         to_jsonb(p) ->> 'first_name',
         to_jsonb(p) ->> 'last_name',
         p.avatar_url
    FROM public.profiles p CROSS JOIN me
   WHERE me.uid IS NOT NULL
     AND (p.id IN (SELECT x FROM ids) OR p.user_id IN (SELECT x FROM ids))
     AND (
       p.user_id = me.uid OR p.id = me.pid
       OR (me.org IS NOT NULL AND p.organization_id = me.org)
       OR p.id IN (SELECT ref FROM people WHERE ref IS NOT NULL)
       OR p.user_id IN (SELECT ref FROM people WHERE ref IS NOT NULL)
     )
$$;
REVOKE ALL ON FUNCTION public.get_display_profiles(uuid[]) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.get_display_profiles(uuid[]) TO authenticated, service_role;
