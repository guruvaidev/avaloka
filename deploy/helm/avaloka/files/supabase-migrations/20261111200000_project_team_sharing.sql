-- Profile/team sharing for projects, inherited by their analyses. This stays separate from
-- resource_collaborators, whose recipients are organization app_users.
CREATE TABLE IF NOT EXISTS public.project_shares (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  recipient_profile_id uuid REFERENCES public.profiles(id) ON DELETE CASCADE,
  team_id uuid REFERENCES public.teams(id) ON DELETE CASCADE,
  access_level text NOT NULL DEFAULT 'view'
    CHECK (access_level IN ('view', 'comment', 'edit')),
  can_share boolean NOT NULL DEFAULT false,
  shared_by uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT project_shares_exactly_one_recipient
    CHECK ((recipient_profile_id IS NOT NULL) <> (team_id IS NOT NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_project_shares_profile
  ON public.project_shares(project_id, recipient_profile_id)
  WHERE recipient_profile_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_project_shares_team
  ON public.project_shares(project_id, team_id)
  WHERE team_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_project_shares_profile
  ON public.project_shares(recipient_profile_id)
  WHERE recipient_profile_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_project_shares_team
  ON public.project_shares(team_id)
  WHERE team_id IS NOT NULL;

ALTER TABLE public.project_shares ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.project_shares FROM PUBLIC, anon, authenticated;
GRANT ALL ON public.project_shares TO service_role;

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
    SELECT CASE s.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END
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
    SELECT CASE s.access_level WHEN 'edit' THEN 3 WHEN 'comment' THEN 2 ELSE 1 END
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
  SELECT coalesce(max(level), 0)::integer FROM grants
$$;


REVOKE ALL ON FUNCTION public.project_share_access_level(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.project_share_access_level(uuid) TO authenticated, service_role;

CREATE POLICY "team shares read projects" ON public.projects
  FOR SELECT TO authenticated USING (public.project_share_access_level(id) >= 1);
CREATE POLICY "team shares read project analyses" ON public.analyses
  FOR SELECT TO authenticated USING (project_id IS NOT NULL AND public.project_share_access_level(project_id) >= 1);
CREATE POLICY "team editors update project analyses" ON public.analyses
  FOR UPDATE TO authenticated
  USING (project_id IS NOT NULL AND public.project_share_access_level(project_id) >= 3)
  WITH CHECK (project_id IS NOT NULL AND public.project_share_access_level(project_id) >= 3
    AND owner_id = public.analysis_owner_profile(id));

-- Existing message and comment policies already use analysis_share_access_level.
-- Restrict all comment insert paths so a View grant cannot write comments.
CREATE POLICY "comment access required" ON public.analysis_comments
  AS RESTRICTIVE FOR INSERT TO authenticated
  WITH CHECK (public.analysis_share_access_level(analysis_id) >= 2);

CREATE TRIGGER set_project_shares_updated_at BEFORE UPDATE ON public.project_shares
  FOR EACH ROW EXECUTE FUNCTION public.set_analysis_shares_updated_at();
