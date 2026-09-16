ALTER TABLE public.reports
  ADD COLUMN IF NOT EXISTS bookmarked boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS archived boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS pinned boolean NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS reports_org_flags_idx
  ON public.reports (organization_id, archived, bookmarked);