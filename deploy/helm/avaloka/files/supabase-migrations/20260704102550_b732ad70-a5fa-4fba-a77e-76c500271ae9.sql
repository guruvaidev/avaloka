
ALTER TABLE public.departments
  DROP COLUMN IF EXISTS head_name,
  DROP COLUMN IF EXISTS head_email,
  DROP COLUMN IF EXISTS head_avatar,
  ADD COLUMN IF NOT EXISTS head_profile_id uuid REFERENCES public.profiles(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_departments_head_profile_id ON public.departments(head_profile_id);
