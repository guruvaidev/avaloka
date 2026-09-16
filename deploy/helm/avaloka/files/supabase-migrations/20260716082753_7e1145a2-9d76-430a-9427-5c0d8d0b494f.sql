
-- projects
DROP POLICY IF EXISTS "owners insert own projects" ON public.projects;
DROP POLICY IF EXISTS "owners read own projects" ON public.projects;
DROP POLICY IF EXISTS "owners update own projects" ON public.projects;
DROP POLICY IF EXISTS "owners delete own projects" ON public.projects;

CREATE POLICY "owners insert own projects" ON public.projects
  FOR INSERT TO authenticated
  WITH CHECK (owner_id = public.current_profile_id());
CREATE POLICY "owners read own projects" ON public.projects
  FOR SELECT TO authenticated
  USING (owner_id = public.current_profile_id());
CREATE POLICY "owners update own projects" ON public.projects
  FOR UPDATE TO authenticated
  USING (owner_id = public.current_profile_id())
  WITH CHECK (owner_id = public.current_profile_id());
CREATE POLICY "owners delete own projects" ON public.projects
  FOR DELETE TO authenticated
  USING (owner_id = public.current_profile_id());

-- analyses
DROP POLICY IF EXISTS "owners insert own analyses" ON public.analyses;
DROP POLICY IF EXISTS "owners read own analyses" ON public.analyses;
DROP POLICY IF EXISTS "owners update own analyses" ON public.analyses;
DROP POLICY IF EXISTS "owners delete own analyses" ON public.analyses;

CREATE POLICY "owners insert own analyses" ON public.analyses
  FOR INSERT TO authenticated
  WITH CHECK (owner_id = public.current_profile_id());
CREATE POLICY "owners read own analyses" ON public.analyses
  FOR SELECT TO authenticated
  USING (owner_id = public.current_profile_id());
CREATE POLICY "owners update own analyses" ON public.analyses
  FOR UPDATE TO authenticated
  USING (owner_id = public.current_profile_id())
  WITH CHECK (owner_id = public.current_profile_id());
CREATE POLICY "owners delete own analyses" ON public.analyses
  FOR DELETE TO authenticated
  USING (owner_id = public.current_profile_id());
