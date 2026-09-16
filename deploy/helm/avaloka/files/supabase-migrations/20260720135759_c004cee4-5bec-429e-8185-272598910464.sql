
CREATE TABLE public.notifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  recipient_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  actor_id uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  type text NOT NULL,
  title text NOT NULL,
  body text,
  resource_type text,
  resource_id uuid,
  link text,
  read_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.notifications TO authenticated;
GRANT ALL ON public.notifications TO service_role;

ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY "recipients read own notifications"
  ON public.notifications FOR SELECT
  TO authenticated
  USING (recipient_id = public.current_profile_id());

CREATE POLICY "recipients update own notifications"
  ON public.notifications FOR UPDATE
  TO authenticated
  USING (recipient_id = public.current_profile_id())
  WITH CHECK (recipient_id = public.current_profile_id());

CREATE POLICY "recipients delete own notifications"
  ON public.notifications FOR DELETE
  TO authenticated
  USING (recipient_id = public.current_profile_id());

CREATE POLICY "authenticated can insert notifications"
  ON public.notifications FOR INSERT
  TO authenticated
  WITH CHECK (
    actor_id = public.current_profile_id()
    OR recipient_id = public.current_profile_id()
  );

CREATE INDEX notifications_recipient_created_idx
  ON public.notifications (recipient_id, created_at DESC);

CREATE INDEX notifications_recipient_unread_idx
  ON public.notifications (recipient_id)
  WHERE read_at IS NULL;

ALTER PUBLICATION supabase_realtime ADD TABLE public.notifications;
