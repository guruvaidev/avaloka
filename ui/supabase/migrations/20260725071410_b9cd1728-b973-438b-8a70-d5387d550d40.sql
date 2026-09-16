-- 1) Backfill app_users.organization_id
UPDATE public.app_users au
SET organization_id = p.organization_id, updated_at = now()
FROM public.profiles p
WHERE au.organization_id IS NULL
  AND p.organization_id IS NOT NULL
  AND (p.id = au.auth_user_id OR p.id = au.profile_id);

UPDATE public.app_users au
SET organization_id = p.organization_id, updated_at = now()
FROM public.profiles p
WHERE au.organization_id IS NULL
  AND au.invited_by IS NOT NULL
  AND p.id = au.invited_by
  AND p.organization_id IS NOT NULL;

-- 2) Backfill app_users.profile_id where possible (via auth_user_id)
UPDATE public.app_users au
SET profile_id = au.auth_user_id, updated_at = now()
WHERE au.profile_id IS NULL
  AND au.auth_user_id IS NOT NULL
  AND EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = au.auth_user_id);

-- 3) Backfill organization_teams.organization_id from lead or any member
UPDATE public.organization_teams t
SET organization_id = lead.organization_id, updated_at = now()
FROM public.app_users lead
WHERE t.organization_id IS NULL
  AND t.lead_user_id = lead.id
  AND lead.organization_id IS NOT NULL;

UPDATE public.organization_teams t
SET organization_id = m.organization_id, updated_at = now()
FROM public.app_users m
WHERE t.organization_id IS NULL
  AND m.team_id = t.id
  AND m.organization_id IS NOT NULL;

-- 4) Trigger to auto-fill organization_id on future app_users inserts
CREATE OR REPLACE FUNCTION public.app_users_fill_org()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF NEW.organization_id IS NULL THEN
    SELECT p.organization_id INTO NEW.organization_id
    FROM public.profiles p
    WHERE p.id = COALESCE(NEW.auth_user_id, NEW.profile_id, NEW.invited_by)
      AND p.organization_id IS NOT NULL
    LIMIT 1;
  END IF;

  IF NEW.organization_id IS NULL AND NEW.invited_by IS NOT NULL THEN
    SELECT p.organization_id INTO NEW.organization_id
    FROM public.profiles p
    WHERE p.id = NEW.invited_by
    LIMIT 1;
  END IF;

  IF NEW.organization_id IS NULL THEN
    SELECT p.organization_id INTO NEW.organization_id
    FROM public.profiles p
    WHERE p.id = public.current_profile_id()
    LIMIT 1;
  END IF;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_app_users_fill_org ON public.app_users;
CREATE TRIGGER trg_app_users_fill_org
BEFORE INSERT ON public.app_users
FOR EACH ROW EXECUTE FUNCTION public.app_users_fill_org();