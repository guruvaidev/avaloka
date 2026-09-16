CREATE OR REPLACE FUNCTION public.is_support_admin()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT COALESCE(lower(auth.jwt() ->> 'email') = 'support@avaloka.ai', false)
$$;

REVOKE EXECUTE ON FUNCTION public.is_support_admin() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.is_support_admin() FROM anon;
GRANT EXECUTE ON FUNCTION public.is_support_admin() TO authenticated;
GRANT EXECUTE ON FUNCTION public.is_support_admin() TO service_role;

CREATE TABLE IF NOT EXISTS public.support_ticket_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ticket_id uuid NOT NULL REFERENCES public.support_tickets(id) ON DELETE CASCADE,
  author_id uuid NOT NULL,
  author_role text NOT NULL DEFAULT 'user',
  body text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS support_ticket_messages_ticket_idx
  ON public.support_ticket_messages(ticket_id, created_at);

GRANT SELECT, INSERT ON public.support_ticket_messages TO authenticated;
GRANT ALL ON public.support_ticket_messages TO service_role;

ALTER TABLE public.support_ticket_messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Read own ticket messages" ON public.support_ticket_messages;
CREATE POLICY "Read own ticket messages"
  ON public.support_ticket_messages FOR SELECT TO authenticated
  USING (
    public.is_support_admin()
    OR EXISTS (
      SELECT 1 FROM public.support_tickets t
      WHERE t.id = support_ticket_messages.ticket_id AND t.user_id = auth.uid()
    )
  );

DROP POLICY IF EXISTS "Write own ticket messages" ON public.support_ticket_messages;
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

DROP POLICY IF EXISTS "Support admin can view all tickets" ON public.support_tickets;
CREATE POLICY "Support admin can view all tickets"
  ON public.support_tickets FOR SELECT TO authenticated
  USING (public.is_support_admin());

DROP POLICY IF EXISTS "Support admin can update tickets" ON public.support_tickets;
CREATE POLICY "Support admin can update tickets"
  ON public.support_tickets FOR UPDATE TO authenticated
  USING (public.is_support_admin())
  WITH CHECK (public.is_support_admin());

GRANT UPDATE ON public.support_tickets TO authenticated;