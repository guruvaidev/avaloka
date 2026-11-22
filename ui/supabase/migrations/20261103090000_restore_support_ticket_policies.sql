-- Restores public.is_support_admin() and the four support-ticket RLS policies on
-- installs where they were never created.
--
-- 20260818080148 defines is_support_admin() with auth.jwt() and creates the four
-- policies on top of it. The DB image has no auth.jwt(); gotrue's own migrations
-- add it. On a fresh install the chart's migrate hook could run before gotrue had
-- started, so that file's function and policy statements failed, and the hook
-- (which applies baseline files once, errors reported but not fatal) recorded it
-- anyway and never ran it again. The hook now waits for auth.jwt(); this repairs
-- installs made before it did. Without the policies support_ticket_messages has
-- RLS on and no policy, so nobody but service_role could read or write a ticket
-- message, and support admins could not see other users' tickets: it failed
-- closed.
--
-- On an install where they already exist this changes nothing: the function is
-- replaced by its identical current definition (20261101120000_support_admin_from_roles,
-- never the literal-address form), its grants are re-stated, and each policy is
-- created only if no policy of that name exists on its table.

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

ALTER TABLE public.support_ticket_messages ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public'
                 AND tablename = 'support_ticket_messages' AND policyname = 'Read own ticket messages') THEN
    CREATE POLICY "Read own ticket messages"
      ON public.support_ticket_messages FOR SELECT TO authenticated
      USING (
        public.is_support_admin()
        OR EXISTS (
          SELECT 1 FROM public.support_tickets t
          WHERE t.id = support_ticket_messages.ticket_id AND t.user_id = auth.uid()
        )
      );
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public'
                 AND tablename = 'support_ticket_messages' AND policyname = 'Write own ticket messages') THEN
    CREATE POLICY "Write own ticket messages"
      ON public.support_ticket_messages FOR INSERT TO authenticated
      WITH CHECK (
        author_id = auth.uid()
        AND (
          public.is_support_admin()
          OR EXISTS (
            SELECT 1 FROM public.support_tickets t
            WHERE t.id = support_ticket_messages.ticket_id AND t.user_id = auth.uid()
          )
        )
      );
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public'
                 AND tablename = 'support_tickets' AND policyname = 'Support admin can view all tickets') THEN
    CREATE POLICY "Support admin can view all tickets"
      ON public.support_tickets FOR SELECT TO authenticated
      USING (public.is_support_admin());
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public'
                 AND tablename = 'support_tickets' AND policyname = 'Support admin can update tickets') THEN
    CREATE POLICY "Support admin can update tickets"
      ON public.support_tickets FOR UPDATE TO authenticated
      USING (public.is_support_admin())
      WITH CHECK (public.is_support_admin());
  END IF;
END
$$;
