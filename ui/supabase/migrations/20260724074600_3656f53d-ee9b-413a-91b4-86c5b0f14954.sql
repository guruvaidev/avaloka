
-- Normalize teams schema
ALTER TABLE public.teams RENAME TO organization_teams;

ALTER TABLE public.organization_teams
  DROP COLUMN IF EXISTS lead_name,
  DROP COLUMN IF EXISTS lead_email,
  DROP COLUMN IF EXISTS lead_avatar,
  DROP COLUMN IF EXISTS user_count,
  DROP COLUMN IF EXISTS user_avatars,
  DROP COLUMN IF EXISTS member_emails,
  ADD COLUMN IF NOT EXISTS lead_user_id uuid REFERENCES public.app_users(id) ON DELETE SET NULL;

ALTER TABLE public.app_users
  ADD COLUMN IF NOT EXISTS team_id uuid REFERENCES public.organization_teams(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_app_users_team_id ON public.app_users(team_id);
CREATE INDEX IF NOT EXISTS idx_organization_teams_lead_user_id ON public.organization_teams(lead_user_id);

-- Recreate policy on renamed table (policy carries over on RENAME, but ensure GRANTs are present)
GRANT SELECT, INSERT, UPDATE, DELETE ON public.organization_teams TO authenticated;
GRANT ALL ON public.organization_teams TO service_role;
