
CREATE OR REPLACE FUNCTION public.get_my_report_access(_report_id uuid)
RETURNS text
LANGUAGE plpgsql
STABLE SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  _auth uuid := auth.uid();
  _profile uuid;
  _created_by uuid;
  _analysis_id uuid;
  _project_id uuid;
  _level text;
BEGIN
  IF _auth IS NULL THEN
    RETURN 'none';
  END IF;

  SELECT created_by, analysis_id, project_id
    INTO _created_by, _analysis_id, _project_id
  FROM public.reports WHERE id = _report_id;

  IF NOT FOUND THEN
    RETURN 'none';
  END IF;

  SELECT id INTO _profile FROM public.profiles WHERE user_id = _auth LIMIT 1;
  IF _profile IS NULL THEN
    SELECT id INTO _profile FROM public.profiles WHERE id = _auth LIMIT 1;
  END IF;

  IF _created_by = _auth OR (_profile IS NOT NULL AND _created_by = _profile) THEN
    RETURN 'edit';
  END IF;

  IF public.has_role(_auth, 'admin') THEN
    RETURN 'edit';
  END IF;

  SELECT rc.access_level INTO _level
  FROM public.resource_collaborators rc
  JOIN public.app_users au ON au.id = rc.app_user_id
  WHERE au.auth_user_id = _auth
    AND (
      (rc.resource_type = 'report'   AND rc.resource_id = _report_id)
      OR (rc.resource_type = 'analysis' AND rc.resource_id = _analysis_id)
      OR (_project_id IS NOT NULL AND rc.resource_type = 'project' AND rc.resource_id = _project_id)
    )
  ORDER BY CASE lower(rc.access_level)
    WHEN 'edit' THEN 1
    WHEN 'comment' THEN 2
    WHEN 'view' THEN 3
    ELSE 4 END
  LIMIT 1;

  RETURN COALESCE(lower(_level), 'view');
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_my_report_access(uuid) TO authenticated;
