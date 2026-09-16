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

  IF public.has_role(_auth, 'admin'::public.app_role) THEN
    RETURN 'edit';
  END IF;

  SELECT rc.access_level INTO _level
  FROM public.resource_collaborators rc
  JOIN public.app_users au ON au.id = rc.app_user_id
  WHERE au.auth_user_id = _auth
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
$$;

-- Allow owners/admins of a report to read all collaborator records for that report
CREATE POLICY "report_owner_admin_read_collaborators"
ON public.resource_collaborators
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.reports r
    WHERE r.id = resource_collaborators.resource_id
      AND resource_collaborators.resource_type = 'report'
      AND (
        r.created_by = auth.uid()
        OR public.has_role(auth.uid(), 'admin'::public.app_role)
      )
  )
);

-- Allow any authenticated user to read their own collaboration rows, even across orgs
CREATE POLICY "collaborator_can_read_own_row"
ON public.resource_collaborators
FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.app_users au
    WHERE au.id = resource_collaborators.app_user_id
      AND au.auth_user_id = auth.uid()
  )
);

-- Allow collaborators to read reports they have access to via share/collaboration
CREATE POLICY "collaborators_can_read_shared_reports"
ON public.reports
FOR SELECT
TO authenticated
USING (
  created_by = auth.uid()
  OR public.has_role(auth.uid(), 'admin'::public.app_role)
  OR public.get_my_report_access(id) IN ('view', 'comment', 'edit')
);

-- Allow shared users with comment or edit access to create comments
CREATE POLICY "report_collaborators_can_comment"
ON public.report_comments
FOR INSERT
TO authenticated
WITH CHECK (
  public.get_my_report_access(report_id) IN ('comment', 'edit')
);

-- Allow shared users with comment or edit access to read comments
CREATE POLICY "report_collaborators_can_read_comments"
ON public.report_comments
FOR SELECT
TO authenticated
USING (
  public.get_my_report_access(report_id) IN ('view', 'comment', 'edit')
);

-- Allow report owners/admins to update/delete any comment on their report
CREATE POLICY "report_owner_admin_can_manage_comments"
ON public.report_comments
FOR ALL
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM public.reports r
    WHERE r.id = report_comments.report_id
      AND (r.created_by = auth.uid() OR public.has_role(auth.uid(), 'admin'::public.app_role))
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.reports r
    WHERE r.id = report_comments.report_id
      AND (r.created_by = auth.uid() OR public.has_role(auth.uid(), 'admin'::public.app_role))
  )
);

-- Allow users to read app_users records linked to themselves (needed for self-lookup)
CREATE POLICY "users_can_read_own_app_user"
ON public.app_users
FOR SELECT
TO authenticated
USING (
  auth_user_id = auth.uid() OR id = auth.uid()
);
