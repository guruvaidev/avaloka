CREATE OR REPLACE FUNCTION public.get_my_report_access(_report_id uuid)
 RETURNS text
 LANGUAGE plpgsql
 STABLE SECURITY DEFINER
 SET search_path TO 'public'
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

  SELECT created_by, analysis_id, project_id
    INTO _created_by, _analysis_id, _project_id
  FROM public.reports WHERE id = _report_id;

  IF NOT FOUND THEN
    RETURN 'none';
  END IF;

  SELECT email INTO _auth_email FROM auth.users WHERE id = _auth LIMIT 1;

  SELECT id INTO _profile FROM public.profiles WHERE user_id = _auth LIMIT 1;
  IF _profile IS NULL THEN
    SELECT id INTO _profile FROM public.profiles WHERE id = _auth LIMIT 1;
  END IF;

  IF _created_by = _auth OR (_profile IS NOT NULL AND _created_by = _profile) THEN
    RETURN 'edit';
  END IF;

  IF public.has_role(_auth, 'admin'::public.app_role) THEN
    RETURN 'edit';
  END IF;

  -- Opportunistically backfill missing auth_user_id link so future checks are fast
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

  RETURN COALESCE(lower(_level), 'none');
END;
$function$;

REVOKE EXECUTE ON FUNCTION public.get_my_report_access(uuid) FROM anon, PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_my_report_access(uuid) TO authenticated, service_role;