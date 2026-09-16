CREATE OR REPLACE FUNCTION public.ensure_org_owner_app_user(_org_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  _owner_profile uuid;
  _auth_id uuid;
  _email text;
  _full_perms jsonb := jsonb_build_object(
    'User Management', jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Analysis',        jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Projects',        jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Dashboard',       jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Data Configuration', jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true)
  );
  _existing uuid;
BEGIN
  SELECT o.owner_profile_id INTO _owner_profile
  FROM public.organizations o WHERE o.id = _org_id;

  IF _owner_profile IS NULL THEN
    RETURN;
  END IF;

  SELECT COALESCE(p.user_id, p.id) INTO _auth_id
  FROM public.profiles p WHERE p.id = _owner_profile;

  SELECT u.email INTO _email FROM auth.users u WHERE u.id = _auth_id;
  IF _email IS NULL THEN
    SELECT p.company_email INTO _email FROM public.profiles p WHERE p.id = _owner_profile;
  END IF;
  IF _email IS NULL THEN
    RETURN;
  END IF;

  SELECT au.id INTO _existing
  FROM public.app_users au
  WHERE au.organization_id = _org_id
    AND (au.profile_id = _owner_profile
         OR (_auth_id IS NOT NULL AND au.auth_user_id = _auth_id)
         OR lower(au.email) = lower(_email))
  LIMIT 1;

  IF _existing IS NOT NULL THEN
    UPDATE public.app_users
    SET role = 'Owner',
        permissions = _full_perms,
        status = 'Active',
        profile_id = COALESCE(profile_id, _owner_profile),
        auth_user_id = COALESCE(auth_user_id, _auth_id),
        accepted_at = COALESCE(accepted_at, now()),
        updated_at = now()
    WHERE id = _existing;
  ELSE
    INSERT INTO public.app_users
      (email, user_id_code, role, department, branch, status, permissions,
       organization_id, profile_id, auth_user_id, accepted_at)
    VALUES
      (_email,
       'OWN-' || upper(substr(replace(_org_id::text, '-', ''), 1, 8)),
       'Owner', 'General', 'HQ', 'Active', _full_perms,
       _org_id, _owner_profile, _auth_id, now());
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.trg_ensure_org_owner_app_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  PERFORM public.ensure_org_owner_app_user(NEW.id);
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS organizations_ensure_owner_app_user ON public.organizations;
CREATE TRIGGER organizations_ensure_owner_app_user
AFTER INSERT OR UPDATE OF owner_profile_id ON public.organizations
FOR EACH ROW
WHEN (NEW.owner_profile_id IS NOT NULL)
EXECUTE FUNCTION public.trg_ensure_org_owner_app_user();

DO $$
DECLARE r record;
BEGIN
  FOR r IN SELECT id FROM public.organizations WHERE owner_profile_id IS NOT NULL LOOP
    PERFORM public.ensure_org_owner_app_user(r.id);
  END LOOP;
END $$;

REVOKE ALL ON FUNCTION public.ensure_org_owner_app_user(uuid) FROM anon;
REVOKE ALL ON FUNCTION public.trg_ensure_org_owner_app_user() FROM anon;