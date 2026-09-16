CREATE OR REPLACE FUNCTION public.current_profile_id()
RETURNS uuid
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  _profile_id uuid;
  _has_user_id boolean;
BEGIN
  SELECT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'profiles'
      AND column_name = 'user_id'
  ) INTO _has_user_id;

  IF _has_user_id THEN
    EXECUTE 'SELECT id FROM public.profiles WHERE user_id = $1 LIMIT 1'
      INTO _profile_id
      USING auth.uid();

    IF _profile_id IS NOT NULL THEN
      RETURN _profile_id;
    END IF;
  END IF;

  SELECT id
  INTO _profile_id
  FROM public.profiles
  WHERE id = auth.uid()
  LIMIT 1;

  RETURN _profile_id;
END;
$$;

GRANT EXECUTE ON FUNCTION public.current_profile_id() TO authenticated;

CREATE OR REPLACE FUNCTION public.current_org_id()
RETURNS uuid
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT organization_id
  FROM public.profiles
  WHERE id = public.current_profile_id()
  LIMIT 1
$$;

GRANT EXECUTE ON FUNCTION public.current_org_id() TO authenticated;

DROP POLICY IF EXISTS "Members can view their organization" ON public.organizations;
DROP POLICY IF EXISTS "Owner can insert their organization" ON public.organizations;
DROP POLICY IF EXISTS "Owner can update their organization" ON public.organizations;
DROP POLICY IF EXISTS "Owner or admin can update their organization" ON public.organizations;

CREATE POLICY "Members can view their organization"
ON public.organizations
FOR SELECT TO authenticated
USING (
  id = public.current_org_id()
  OR owner_profile_id = public.current_profile_id()
);

CREATE POLICY "Owner can insert their organization"
ON public.organizations
FOR INSERT TO authenticated
WITH CHECK (
  owner_profile_id = public.current_profile_id()
);

CREATE POLICY "Owner or admin can update their organization"
ON public.organizations
FOR UPDATE TO authenticated
USING (
  owner_profile_id = public.current_profile_id()
  OR (id = public.current_org_id() AND public.has_role(auth.uid(), 'admin'))
)
WITH CHECK (
  owner_profile_id = public.current_profile_id()
  OR (id = public.current_org_id() AND public.has_role(auth.uid(), 'admin'))
);