
-- 1. profiles.organization_id (one org per user; auto-generated if missing)
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS organization_id UUID;
UPDATE public.profiles SET organization_id = gen_random_uuid() WHERE organization_id IS NULL;
ALTER TABLE public.profiles ALTER COLUMN organization_id SET NOT NULL;
ALTER TABLE public.profiles ALTER COLUMN organization_id SET DEFAULT gen_random_uuid();
CREATE INDEX IF NOT EXISTS profiles_organization_id_idx ON public.profiles(organization_id);

-- Ensure every auth user has a profile row (so current_org_id works for existing users)
INSERT INTO public.profiles (id)
SELECT u.id FROM auth.users u
LEFT JOIN public.profiles p ON p.id = u.id
WHERE p.id IS NULL;

-- 2. Helper: current org for the signed-in user
CREATE OR REPLACE FUNCTION public.current_org_id()
RETURNS UUID
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT organization_id FROM public.profiles WHERE id = auth.uid()
$$;

GRANT EXECUTE ON FUNCTION public.current_org_id() TO authenticated;

-- 3. Add organization_id to each config table
ALTER TABLE public.departments    ADD COLUMN IF NOT EXISTS organization_id UUID;
ALTER TABLE public.config_roles   ADD COLUMN IF NOT EXISTS organization_id UUID;
ALTER TABLE public.branches       ADD COLUMN IF NOT EXISTS organization_id UUID;
ALTER TABLE public.app_users      ADD COLUMN IF NOT EXISTS organization_id UUID;
ALTER TABLE public.teams          ADD COLUMN IF NOT EXISTS organization_id UUID;
ALTER TABLE public.models         ADD COLUMN IF NOT EXISTS organization_id UUID;

CREATE INDEX IF NOT EXISTS departments_org_idx  ON public.departments(organization_id);
CREATE INDEX IF NOT EXISTS config_roles_org_idx ON public.config_roles(organization_id);
CREATE INDEX IF NOT EXISTS branches_org_idx     ON public.branches(organization_id);
CREATE INDEX IF NOT EXISTS app_users_org_idx    ON public.app_users(organization_id);
CREATE INDEX IF NOT EXISTS teams_org_idx        ON public.teams(organization_id);
CREATE INDEX IF NOT EXISTS models_org_idx       ON public.models(organization_id);

-- 4. Replace admin-only policies with org-scoped policies for authenticated users
DROP POLICY IF EXISTS "admin all departments"  ON public.departments;
DROP POLICY IF EXISTS "admin all config_roles" ON public.config_roles;
DROP POLICY IF EXISTS "admin all branches"     ON public.branches;
DROP POLICY IF EXISTS "admin all app_users"    ON public.app_users;
DROP POLICY IF EXISTS "admin all teams"        ON public.teams;
DROP POLICY IF EXISTS "admin all models"       ON public.models;

CREATE POLICY "org members manage departments" ON public.departments
  FOR ALL TO authenticated
  USING (organization_id = public.current_org_id())
  WITH CHECK (organization_id = public.current_org_id());

CREATE POLICY "org members manage config_roles" ON public.config_roles
  FOR ALL TO authenticated
  USING (organization_id = public.current_org_id())
  WITH CHECK (organization_id = public.current_org_id());

CREATE POLICY "org members manage branches" ON public.branches
  FOR ALL TO authenticated
  USING (organization_id = public.current_org_id())
  WITH CHECK (organization_id = public.current_org_id());

CREATE POLICY "org members manage app_users" ON public.app_users
  FOR ALL TO authenticated
  USING (organization_id = public.current_org_id())
  WITH CHECK (organization_id = public.current_org_id());

CREATE POLICY "org members manage teams" ON public.teams
  FOR ALL TO authenticated
  USING (organization_id = public.current_org_id())
  WITH CHECK (organization_id = public.current_org_id());

CREATE POLICY "org members manage models" ON public.models
  FOR ALL TO authenticated
  USING (organization_id = public.current_org_id())
  WITH CHECK (organization_id = public.current_org_id());
