ALTER TABLE public.report_comments
  ADD COLUMN IF NOT EXISTS attachments jsonb DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_report_comments_parent_id ON public.report_comments(parent_id);