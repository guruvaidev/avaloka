CREATE OR REPLACE FUNCTION public.current_org_id()
RETURNS uuid
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  _pid uuid;
  _org uuid;
BEGIN
  _pid := public.current_profile_id();
  IF _pid IS NULL THEN
    RETURN NULL;
  END IF;

  SELECT organization_id INTO _org FROM public.profiles WHERE id = _pid LIMIT 1;
  IF _org IS NOT NULL THEN
    RETURN _org;
  END IF;

  -- Fallback 1: organization the user owns
  SELECT id INTO _org FROM public.organizations WHERE owner_profile_id = _pid LIMIT 1;
  IF _org IS NOT NULL THEN
    RETURN _org;
  END IF;

  -- Fallback 2: membership row in app_users (profile link, auth link or email)
  SELECT au.organization_id INTO _org
  FROM public.app_users au
  LEFT JOIN public.profiles pr ON pr.id = _pid
  WHERE au.organization_id IS NOT NULL
    AND (
      au.profile_id = _pid
      OR au.auth_user_id = auth.uid()
      OR lower(au.email) = lower(coalesce(pr.company_email, ''))
    )
  LIMIT 1;

  RETURN _org;
END;
$$;

REVOKE ALL ON FUNCTION public.current_org_id() FROM anon;