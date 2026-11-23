-- Display names and avatars for owners and authors on accessible analyses,
-- including analyses inherited through a shared project. Preserve the existing
-- Enterprise organization and legacy collaborator paths and display-only fields.
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
    -- Include profile/team grants and inherited project access without exposing
    -- private profile fields or granting access to the resources themselves.
    UNION SELECT 'analysis', a.id FROM public.analyses a CROSS JOIN me
     WHERE me.pid IS NOT NULL AND public.analysis_share_access_level(a.id) >= 1
    UNION SELECT 'project', pr.id FROM public.projects pr CROSS JOIN me
     WHERE me.pid IS NOT NULL AND public.project_share_access_level(pr.id) >= 1
  ),
  -- A report brings in the people on its analysis and project, as get_report_teams does.
  scope AS (
    SELECT resource_type, resource_id FROM mine
    UNION SELECT 'analysis', r.analysis_id FROM mine m JOIN public.reports r
      ON m.resource_type = 'report' AND r.id = m.resource_id WHERE r.analysis_id IS NOT NULL
    UNION SELECT 'project', r.project_id FROM mine m JOIN public.reports r
      ON m.resource_type = 'report' AND r.id = m.resource_id WHERE r.project_id IS NOT NULL
    UNION SELECT 'analysis', a.id FROM mine m JOIN public.analyses a
      ON m.resource_type = 'project' AND a.project_id = m.resource_id
    UNION SELECT 'analysis', a.id FROM mine m JOIN public.analyses a
      ON m.resource_type = 'analysis' AND a.parent_analysis_id = m.resource_id
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
    UNION SELECT msg.author_id FROM scope s JOIN public.analysis_messages msg
      ON s.resource_type = 'analysis' AND msg.analysis_id = s.resource_id
    UNION SELECT c.author_id FROM scope s JOIN public.analysis_comments c
      ON s.resource_type = 'analysis' AND c.analysis_id = s.resource_id
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

NOTIFY pgrst, 'reload schema';
