ALTER TABLE public.analysis_messages
  ADD COLUMN IF NOT EXISTS author_id uuid REFERENCES public.profiles(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_analysis_messages_author ON public.analysis_messages(author_id);

UPDATE public.analysis_messages m
SET author_id = a.owner_id
FROM public.analyses a
WHERE m.analysis_id = a.id
  AND m.role = 'user'
  AND m.author_id IS NULL;