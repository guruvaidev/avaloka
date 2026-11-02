-- public.is_support_admin() decided who may read and update EVERY support
-- ticket (and every ticket message) by comparing the caller's JWT email with a
-- literal address (20260818080148). The support_tickets and
-- support_ticket_messages RLS policies call it, so whoever held an account with
-- that address could read all of them.
--
-- A support admin is now an account holding the 'admin' role in
-- public.user_roles. The chart's migrate hook grants that role from admin.emails
-- (ADMIN_EMAILS) at deploy time; elsewhere, grant it with an insert into
-- public.user_roles. No address appears here. Existing user_roles grants are
-- untouched; an account that relied only on the literal address loses this
-- access until it is granted the role.
--
-- search_path is empty and every name is schema-qualified, so the SECURITY
-- DEFINER body cannot be redirected through a caller-controlled search_path.
CREATE OR REPLACE FUNCTION public.is_support_admin()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.user_roles r
    WHERE r.user_id = auth.uid()
      AND r.role = 'admin'::public.app_role
  )
$$;

REVOKE EXECUTE ON FUNCTION public.is_support_admin() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.is_support_admin() FROM anon;
GRANT EXECUTE ON FUNCTION public.is_support_admin() TO authenticated;
GRANT EXECUTE ON FUNCTION public.is_support_admin() TO service_role;
