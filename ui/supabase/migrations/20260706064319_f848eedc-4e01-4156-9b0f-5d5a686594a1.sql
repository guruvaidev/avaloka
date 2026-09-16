
-- Restructure app_users: profile-based user data lives in profiles
ALTER TABLE public.app_users
  ADD COLUMN IF NOT EXISTS profile_id uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS invited_by uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS auth_user_id uuid,
  ADD COLUMN IF NOT EXISTS invited_at timestamptz,
  ADD COLUMN IF NOT EXISTS accepted_at timestamptz;

-- Backfill profile_id from auth_user_id where possible (profiles.id = auth uid)
UPDATE public.app_users au
SET profile_id = p.id
FROM public.profiles p
WHERE au.profile_id IS NULL
  AND au.auth_user_id IS NOT NULL
  AND p.id = au.auth_user_id;

-- Drop duplicate columns now stored in profiles
ALTER TABLE public.app_users
  DROP COLUMN IF EXISTS name,
  DROP COLUMN IF EXISTS avatar;

-- Allow Invited status
ALTER TABLE public.app_users DROP CONSTRAINT IF EXISTS app_users_status_check;
ALTER TABLE public.app_users
  ADD CONSTRAINT app_users_status_check
  CHECK (status = ANY (ARRAY['Active'::text, 'Inactive'::text, 'Invited'::text]));

CREATE INDEX IF NOT EXISTS app_users_profile_id_idx ON public.app_users(profile_id);
CREATE INDEX IF NOT EXISTS app_users_auth_user_id_idx ON public.app_users(auth_user_id);
CREATE INDEX IF NOT EXISTS app_users_email_org_idx ON public.app_users(organization_id, lower(email));
