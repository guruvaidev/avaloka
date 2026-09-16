
CREATE TYPE public.analysis_message_role AS ENUM ('user', 'assistant');

CREATE TABLE public.analysis_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id uuid NOT NULL REFERENCES public.analyses(id) ON DELETE CASCADE,
  role public.analysis_message_role NOT NULL,
  content text NOT NULL,
  output jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_analysis_messages_analysis ON public.analysis_messages(analysis_id, created_at);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.analysis_messages TO authenticated;
GRANT ALL ON public.analysis_messages TO service_role;

ALTER TABLE public.analysis_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "owners read own analysis messages"
  ON public.analysis_messages FOR SELECT
  TO authenticated
  USING (EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = analysis_id AND a.owner_id = auth.uid()));

CREATE POLICY "owners insert own analysis messages"
  ON public.analysis_messages FOR INSERT
  TO authenticated
  WITH CHECK (EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = analysis_id AND a.owner_id = auth.uid()));

CREATE POLICY "owners update own analysis messages"
  ON public.analysis_messages FOR UPDATE
  TO authenticated
  USING (EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = analysis_id AND a.owner_id = auth.uid()))
  WITH CHECK (EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = analysis_id AND a.owner_id = auth.uid()));

CREATE POLICY "owners delete own analysis messages"
  ON public.analysis_messages FOR DELETE
  TO authenticated
  USING (EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = analysis_id AND a.owner_id = auth.uid()));
