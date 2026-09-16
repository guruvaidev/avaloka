
-- 1) One-time backfill: link app_users to auth.users by lowercase email
UPDATE public.app_users AS au
SET auth_user_id = u.id, updated_at = now()
FROM auth.users u
WHERE au.auth_user_id IS NULL
  AND au.email IS NOT NULL
  AND lower(au.email) = lower(u.email);

-- 2) Trigger function: on insert/email-change of app_users, resolve auth_user_id
CREATE OR REPLACE FUNCTION public.link_app_user_from_email()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF NEW.auth_user_id IS NULL AND NEW.email IS NOT NULL THEN
    SELECT u.id INTO NEW.auth_user_id
    FROM auth.users u
    WHERE lower(u.email) = lower(NEW.email)
    LIMIT 1;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_link_app_user_from_email_ins ON public.app_users;
CREATE TRIGGER trg_link_app_user_from_email_ins
BEFORE INSERT ON public.app_users
FOR EACH ROW EXECUTE FUNCTION public.link_app_user_from_email();

DROP TRIGGER IF EXISTS trg_link_app_user_from_email_upd ON public.app_users;
CREATE TRIGGER trg_link_app_user_from_email_upd
BEFORE UPDATE OF email ON public.app_users
FOR EACH ROW EXECUTE FUNCTION public.link_app_user_from_email();
