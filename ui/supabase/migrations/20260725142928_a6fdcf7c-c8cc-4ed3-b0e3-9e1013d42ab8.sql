
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
SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  WITH accessible AS (
    SELECT r.id AS report_id, r.analysis_id, r.project_id, r.created_by
    FROM public.reports r
    WHERE r.id = ANY(_report_ids)
      AND (
        r.created_by = auth.uid()
        OR public.has_role(auth.uid(), 'admin')
        OR public.is_resource_collaborator('report', r.id)
        OR public.is_resource_collaborator('analysis', r.analysis_id)
        OR (r.project_id IS NOT NULL AND public.is_resource_collaborator('project', r.project_id))
      )
  ),
  creators AS (
    SELECT a.report_id,
           COALESCE(p.id, a.created_by) AS member_id,
           COALESCE(NULLIF(p.full_name, ''), NULLIF(TRIM(CONCAT_WS(' ', p.first_name, p.last_name)), ''), au_lookup.email, 'Unknown') AS name,
           p.avatar_url,
           au_lookup.email
    FROM accessible a
    LEFT JOIN public.profiles p
      ON p.id = a.created_by OR p.user_id = a.created_by
    LEFT JOIN LATERAL (
      SELECT email FROM public.app_users
      WHERE auth_user_id = a.created_by OR id = a.created_by
      LIMIT 1
    ) au_lookup ON true
  ),
  collabs AS (
    SELECT a.report_id,
           au.id AS member_id,
           COALESCE(NULLIF(p.full_name, ''), NULLIF(TRIM(CONCAT_WS(' ', p.first_name, p.last_name)), ''), au.email, 'Unknown') AS name,
           p.avatar_url,
           au.email
    FROM accessible a
    JOIN public.resource_collaborators rc
      ON (rc.resource_type = 'report'   AND rc.resource_id = a.report_id)
      OR (rc.resource_type = 'analysis' AND rc.resource_id = a.analysis_id)
      OR (a.project_id IS NOT NULL AND rc.resource_type = 'project' AND rc.resource_id = a.project_id)
    JOIN public.app_users au ON au.id = rc.app_user_id
    LEFT JOIN public.profiles p
      ON (au.auth_user_id IS NOT NULL AND (p.id = au.auth_user_id OR p.user_id = au.auth_user_id))
      OR (au.profile_id IS NOT NULL AND p.id = au.profile_id)
  )
  SELECT DISTINCT ON (t.report_id, t.member_id)
         t.report_id, t.member_id, t.name, t.avatar_url, t.email
  FROM (
    SELECT * FROM creators
    UNION ALL
    SELECT * FROM collabs
  ) t
  ORDER BY t.report_id, t.member_id, t.name;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_report_teams(uuid[]) TO authenticated;
