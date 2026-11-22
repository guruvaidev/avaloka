-- Professional profile/team invitations: Free recipients remain viewers.
-- Legacy Enterprise organization policies and permissions are preserved.
CREATE OR REPLACE FUNCTION public.professional_project_grant_level(p_project_id uuid)
RETURNS integer LANGUAGE sql STABLE SECURITY DEFINER SET search_path = '' AS $$
  SELECT coalesce(max(CASE WHEN public.current_plan_tier() = 'free' THEN 1
    ELSE CASE s.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END END), 0)::integer
  FROM public.project_shares s
  WHERE s.project_id = p_project_id AND (
    s.recipient_profile_id = public.current_profile_id()
    OR EXISTS (SELECT 1 FROM public.team_memberships tm WHERE tm.team_id = s.team_id AND tm.user_id = public.current_profile_id())
    OR EXISTS (SELECT 1 FROM public.teams t WHERE t.id = s.team_id AND public.current_profile_id() IN (t.owner_profile_id, t.lead_profile_id))
  )
$$;
REVOKE ALL ON FUNCTION public.professional_project_grant_level(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.professional_project_grant_level(uuid) TO authenticated, service_role;


CREATE OR REPLACE FUNCTION public.project_share_access_level(p_project_id uuid)
RETURNS integer
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  WITH caller AS (
    SELECT public.current_profile_id() AS profile_id
  ), requested AS (
    SELECT id, id AS root_id, owner_id
      FROM public.projects
     WHERE id = p_project_id
  ), accessible_teams AS (
    SELECT tm.team_id
      FROM public.team_memberships tm, caller c
     WHERE tm.user_id = c.profile_id
    UNION
    SELECT t.id
      FROM public.teams t, caller c
     WHERE t.owner_profile_id = c.profile_id
        OR t.lead_profile_id = c.profile_id
  ), grants AS (
    SELECT 3 AS level
      FROM requested r, caller c
     WHERE c.profile_id IS NOT NULL
       AND r.owner_id = c.profile_id
    UNION ALL
    SELECT CASE WHEN public.current_plan_tier() = 'free' THEN 1
      ELSE CASE s.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END END
      FROM public.project_shares s, requested r, caller c
     WHERE s.project_id IN (r.id, r.root_id)
       AND (s.recipient_profile_id = c.profile_id
         OR s.team_id IN (SELECT team_id FROM accessible_teams))
    UNION ALL
    SELECT CASE rc.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END
      FROM public.resource_collaborators rc
      JOIN public.app_users au ON au.id = rc.app_user_id
      JOIN requested r ON rc.resource_id IN (r.id, r.root_id)
     WHERE rc.resource_type = 'project'
       AND au.auth_user_id = auth.uid()
  )
  SELECT coalesce(max(level), 0)::integer FROM grants
$$;

CREATE OR REPLACE FUNCTION public.analysis_share_access_level(p_analysis_id uuid)
RETURNS integer
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  WITH caller AS (
    SELECT public.current_profile_id() AS profile_id
  ), requested AS (
    SELECT id, coalesce(parent_analysis_id, id) AS root_id, owner_id, project_id
      FROM public.analyses
     WHERE id = p_analysis_id
  ), accessible_teams AS (
    SELECT tm.team_id
      FROM public.team_memberships tm, caller c
     WHERE tm.user_id = c.profile_id
    UNION
    SELECT t.id
      FROM public.teams t, caller c
     WHERE t.owner_profile_id = c.profile_id
        OR t.lead_profile_id = c.profile_id
  ), grants AS (
    SELECT 3 AS level
      FROM requested r, caller c
     WHERE c.profile_id IS NOT NULL
       AND r.owner_id = c.profile_id
    UNION ALL
    SELECT CASE WHEN public.current_plan_tier() = 'free' THEN 1
      ELSE CASE s.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END END
      FROM public.analysis_shares s, requested r, caller c
     WHERE s.analysis_id IN (r.id, r.root_id)
       AND (s.recipient_profile_id = c.profile_id
         OR s.team_id IN (SELECT team_id FROM accessible_teams))
    UNION ALL
    SELECT CASE rc.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END
      FROM public.resource_collaborators rc
      JOIN public.app_users au ON au.id = rc.app_user_id
      JOIN requested r ON rc.resource_id IN (r.id, r.root_id)
     WHERE rc.resource_type = 'analysis'
       AND au.auth_user_id = auth.uid()
    UNION ALL
    SELECT public.project_share_access_level(r.project_id) FROM requested r WHERE r.project_id IS NOT NULL
  )
  SELECT CASE
    WHEN public.current_plan_tier() <> 'enterprise'
      AND public.professional_project_grant_level((SELECT project_id FROM requested)) > 0
    THEN least(coalesce(max(level), 0)::integer,
      public.project_share_access_level((SELECT project_id FROM requested)))
    ELSE coalesce(max(level), 0)::integer END FROM grants
$$;




-- Restore Enterprise's previous comment paths, including organization access
-- that is intentionally absent from the Professional sharing resolver.
DROP POLICY IF EXISTS "comment access required" ON public.analysis_comments;
CREATE POLICY "comment access required" ON public.analysis_comments
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.current_plan_tier() = 'enterprise' OR public.analysis_share_access_level(analysis_id) >= 2);
DROP POLICY IF EXISTS "comment access required for updates" ON public.analysis_comments;
CREATE POLICY "comment access required for updates" ON public.analysis_comments
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.current_plan_tier() = 'enterprise' OR public.analysis_share_access_level(analysis_id) >= 2)
  WITH CHECK (public.current_plan_tier() = 'enterprise' OR public.analysis_share_access_level(analysis_id) >= 2);

-- A View/Comment invite must not create an analysis using the existing
-- owner_id insert policy. Only new profile/team invitees are restricted here.
DROP POLICY IF EXISTS "professional project invite insert access" ON public.analyses;
CREATE POLICY "professional project invite insert access" ON public.analyses
  AS RESTRICTIVE FOR INSERT TO authenticated WITH CHECK (
    public.current_plan_tier() = 'enterprise'
    OR public.professional_project_grant_level(project_id) = 0
    OR public.project_share_access_level(project_id) >= 3
  );
DROP POLICY IF EXISTS "professional project invite update access" ON public.analyses;
CREATE POLICY "professional project invite update access" ON public.analyses
  AS RESTRICTIVE FOR UPDATE TO authenticated
  USING (public.current_plan_tier() = 'enterprise'
    OR public.professional_project_grant_level(project_id) = 0
    OR public.analysis_share_access_level(id) >= 3)
  WITH CHECK (public.current_plan_tier() = 'enterprise'
    OR public.professional_project_grant_level(project_id) = 0
    OR public.analysis_share_access_level(id) >= 3);

DROP POLICY IF EXISTS "professional project invite message access" ON public.analysis_messages;
CREATE POLICY "professional project invite message access" ON public.analysis_messages
  AS RESTRICTIVE FOR INSERT TO authenticated WITH CHECK (
    public.current_plan_tier() = 'enterprise' OR NOT EXISTS (
      SELECT 1 FROM public.analyses a WHERE a.id = analysis_id
        AND public.professional_project_grant_level(a.project_id) > 0
        AND public.analysis_share_access_level(a.id) < 3
    )
  );
NOTIFY pgrst, 'reload schema';
