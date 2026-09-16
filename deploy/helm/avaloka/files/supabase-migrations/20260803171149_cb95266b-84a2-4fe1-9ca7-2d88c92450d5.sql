
DROP POLICY IF EXISTS "org members manage branches" ON public.branches;
CREATE POLICY "org members read branches" ON public.branches FOR SELECT TO authenticated USING (organization_id = public.current_org_id());
CREATE POLICY "org admins write branches" ON public.branches FOR INSERT TO authenticated WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins update branches" ON public.branches FOR UPDATE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin()) WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins delete branches" ON public.branches FOR DELETE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin());

DROP POLICY IF EXISTS "org members manage config_roles" ON public.config_roles;
CREATE POLICY "org members read config_roles" ON public.config_roles FOR SELECT TO authenticated USING (organization_id = public.current_org_id());
CREATE POLICY "org admins write config_roles" ON public.config_roles FOR INSERT TO authenticated WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins update config_roles" ON public.config_roles FOR UPDATE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin()) WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins delete config_roles" ON public.config_roles FOR DELETE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin());

DROP POLICY IF EXISTS "org members manage departments" ON public.departments;
CREATE POLICY "org members read departments" ON public.departments FOR SELECT TO authenticated USING (organization_id = public.current_org_id());
CREATE POLICY "org admins write departments" ON public.departments FOR INSERT TO authenticated WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins update departments" ON public.departments FOR UPDATE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin()) WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins delete departments" ON public.departments FOR DELETE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin());

DROP POLICY IF EXISTS "org members manage models" ON public.models;
CREATE POLICY "org members read models" ON public.models FOR SELECT TO authenticated USING (organization_id = public.current_org_id());
CREATE POLICY "org admins write models" ON public.models FOR INSERT TO authenticated WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins update models" ON public.models FOR UPDATE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin()) WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins delete models" ON public.models FOR DELETE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin());

DROP POLICY IF EXISTS "org members manage teams" ON public.organization_teams;
CREATE POLICY "org members read teams" ON public.organization_teams FOR SELECT TO authenticated USING (organization_id = public.current_org_id());
CREATE POLICY "org admins write teams" ON public.organization_teams FOR INSERT TO authenticated WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins update teams" ON public.organization_teams FOR UPDATE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin()) WITH CHECK (organization_id = public.current_org_id() AND public.is_org_admin());
CREATE POLICY "org admins delete teams" ON public.organization_teams FOR DELETE TO authenticated USING (organization_id = public.current_org_id() AND public.is_org_admin());
