REVOKE ALL ON FUNCTION public.ensure_org_owner_app_user(uuid) FROM anon, authenticated, PUBLIC;
REVOKE ALL ON FUNCTION public.trg_ensure_org_owner_app_user() FROM anon, authenticated, PUBLIC;