-- Make the standalone teams table owned by a profile so Professional users can
-- create teams without an organization or customer account.
-- Existing teams retain their account owner's profile as their new owner.

-- Replace policies and helper bodies that currently read customer_account_id.
DROP POLICY IF EXISTS "Account owners and enterprise admins can create teams" ON public.teams;
DROP POLICY IF EXISTS "Account owners and enterprise admins can delete teams" ON public.teams;
DROP POLICY IF EXISTS "Account owners and enterprise admins can update teams" ON public.teams;
DROP POLICY IF EXISTS "Team members can view their team" ON public.teams;
DROP POLICY IF EXISTS "Team admins and account owners can create invites" ON public.team_invitations;
DROP POLICY IF EXISTS "Team admins and account owners can invite members" ON public.team_memberships;

ALTER TABLE public.teams ADD COLUMN IF NOT EXISTS owner_profile_id uuid;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'teams'
      AND column_name = 'customer_account_id'
  ) THEN
    UPDATE public.teams t
       SET owner_profile_id = ca.owner_user_id
      FROM public.customer_accounts ca
     WHERE t.customer_account_id = ca.id
       AND t.owner_profile_id IS NULL;
  END IF;
END
$$;

-- Fail rather than silently orphan any pre-existing teams that cannot be mapped.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM public.teams WHERE owner_profile_id IS NULL) THEN
    RAISE EXCEPTION 'Cannot migrate teams: an existing team has no customer account owner profile';
  END IF;
END
$$;

ALTER TABLE public.teams ALTER COLUMN owner_profile_id SET NOT NULL;
ALTER TABLE public.teams ADD COLUMN IF NOT EXISTS lead_profile_id uuid;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'teams_owner_profile_id_fkey') THEN
    ALTER TABLE public.teams ADD CONSTRAINT teams_owner_profile_id_fkey
      FOREIGN KEY (owner_profile_id) REFERENCES public.profiles(id) ON DELETE CASCADE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'teams_lead_profile_id_fkey') THEN
    ALTER TABLE public.teams ADD CONSTRAINT teams_lead_profile_id_fkey
      FOREIGN KEY (lead_profile_id) REFERENCES public.profiles(id) ON DELETE SET NULL;
  END IF;
END
$$;

ALTER TABLE public.teams DROP COLUMN IF EXISTS customer_account_id;
CREATE INDEX IF NOT EXISTS idx_teams_owner_profile_id ON public.teams(owner_profile_id);
CREATE INDEX IF NOT EXISTS idx_teams_lead_profile_id ON public.teams(lead_profile_id);

CREATE POLICY "Team owners can create teams" ON public.teams
  FOR INSERT TO authenticated
  WITH CHECK (owner_profile_id = public.current_profile_id());
CREATE POLICY "Team members and owners can view teams" ON public.teams
  FOR SELECT TO authenticated
  USING (owner_profile_id = public.current_profile_id()
      OR public.is_team_member(public.current_profile_id(), id));
CREATE POLICY "Team owners can update teams" ON public.teams
  FOR UPDATE TO authenticated
  USING (owner_profile_id = public.current_profile_id())
  WITH CHECK (owner_profile_id = public.current_profile_id());
CREATE POLICY "Team owners can delete teams" ON public.teams
  FOR DELETE TO authenticated
  USING (owner_profile_id = public.current_profile_id());

CREATE POLICY "Team admins and owners can invite members" ON public.team_memberships
  FOR INSERT TO authenticated
  WITH CHECK (
    public.is_team_admin(public.current_profile_id(), team_id)
    OR EXISTS (SELECT 1 FROM public.teams t
                WHERE t.id = team_id AND t.owner_profile_id = public.current_profile_id())
  );
CREATE POLICY "Team admins and owners can create invites" ON public.team_invitations
  FOR INSERT TO authenticated
  WITH CHECK (
    public.is_team_admin(public.current_profile_id(), team_id)
    OR EXISTS (SELECT 1 FROM public.teams t
                WHERE t.id = team_id AND t.owner_profile_id = public.current_profile_id())
  );

CREATE OR REPLACE FUNCTION public.get_team_member_public_info(p_team_id uuid)
RETURNS TABLE(membership_id uuid, team_id uuid, profile_id uuid, auth_user_id uuid,
              full_name text, avatar_url text, role public.team_role,
              joined_at timestamptz, email text)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT tm.id, tm.team_id, p.id, p.user_id, p.full_name, p.avatar_url,
         tm.role, tm.joined_at, public.get_user_email(p.user_id)
    FROM public.team_memberships tm
    JOIN public.profiles p ON p.id = tm.user_id
   WHERE tm.team_id = p_team_id
     AND (public.is_team_member(public.current_profile_id(), p_team_id)
       OR public.is_team_admin(public.current_profile_id(), p_team_id)
       OR EXISTS (SELECT 1 FROM public.teams t WHERE t.id = p_team_id
                   AND t.owner_profile_id = public.current_profile_id()));
$$;

CREATE OR REPLACE FUNCTION public.get_user_customer_account(p_user_id uuid)
RETURNS uuid LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT ca.id
    FROM public.customer_accounts ca
    JOIN public.teams t ON t.owner_profile_id = ca.owner_user_id
    JOIN public.team_memberships tm ON tm.team_id = t.id
   WHERE tm.user_id = p_user_id
   LIMIT 1;
$$;
