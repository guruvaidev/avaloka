
-- 1) Revoke EXECUTE from anon/public on SECURITY DEFINER functions
REVOKE EXECUTE ON FUNCTION public.current_org_id() FROM anon, public;
REVOKE EXECUTE ON FUNCTION public.is_resource_collaborator(text, uuid) FROM anon, public;
REVOKE EXECUTE ON FUNCTION public.has_role(uuid, public.app_role) FROM anon, public;
GRANT EXECUTE ON FUNCTION public.current_org_id() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.is_resource_collaborator(text, uuid) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.has_role(uuid, public.app_role) TO authenticated, service_role;

-- 2 & 4) Tighten resource_share_links policies
DROP POLICY IF EXISTS "Authenticated read share links" ON public.resource_share_links;
DROP POLICY IF EXISTS "Authenticated update share links" ON public.resource_share_links;
DROP POLICY IF EXISTS "Authenticated write share links" ON public.resource_share_links;

CREATE POLICY "Owners read share links for their resources"
ON public.resource_share_links
FOR SELECT
TO authenticated
USING (
  (resource_type = 'analysis' AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = resource_share_links.resource_id AND a.owner_id = auth.uid()
  ))
  OR (resource_type = 'project' AND EXISTS (
    SELECT 1 FROM public.projects p
    WHERE p.id = resource_share_links.resource_id AND p.owner_id = auth.uid()
  ))
);

CREATE POLICY "Owners insert share links for their resources"
ON public.resource_share_links
FOR INSERT
TO authenticated
WITH CHECK (
  (resource_type = 'analysis' AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = resource_share_links.resource_id AND a.owner_id = auth.uid()
  ))
  OR (resource_type = 'project' AND EXISTS (
    SELECT 1 FROM public.projects p
    WHERE p.id = resource_share_links.resource_id AND p.owner_id = auth.uid()
  ))
);

CREATE POLICY "Owners update share links for their resources"
ON public.resource_share_links
FOR UPDATE
TO authenticated
USING (
  (resource_type = 'analysis' AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = resource_share_links.resource_id AND a.owner_id = auth.uid()
  ))
  OR (resource_type = 'project' AND EXISTS (
    SELECT 1 FROM public.projects p
    WHERE p.id = resource_share_links.resource_id AND p.owner_id = auth.uid()
  ))
)
WITH CHECK (
  (resource_type = 'analysis' AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = resource_share_links.resource_id AND a.owner_id = auth.uid()
  ))
  OR (resource_type = 'project' AND EXISTS (
    SELECT 1 FROM public.projects p
    WHERE p.id = resource_share_links.resource_id AND p.owner_id = auth.uid()
  ))
);

CREATE POLICY "Owners delete share links for their resources"
ON public.resource_share_links
FOR DELETE
TO authenticated
USING (
  (resource_type = 'analysis' AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = resource_share_links.resource_id AND a.owner_id = auth.uid()
  ))
  OR (resource_type = 'project' AND EXISTS (
    SELECT 1 FROM public.projects p
    WHERE p.id = resource_share_links.resource_id AND p.owner_id = auth.uid()
  ))
);

-- 3) Tighten dashboard_tabs and dashboard_graphs
DROP POLICY IF EXISTS "owners manage own tabs" ON public.dashboard_tabs;
CREATE POLICY "owners manage own tabs"
ON public.dashboard_tabs
FOR ALL
TO authenticated
USING (
  auth.uid() = owner_id
  AND EXISTS (
    SELECT 1 FROM public.analysis_dashboards d
    WHERE d.id = dashboard_tabs.dashboard_id AND d.owner_id = auth.uid()
  )
)
WITH CHECK (
  auth.uid() = owner_id
  AND EXISTS (
    SELECT 1 FROM public.analysis_dashboards d
    JOIN public.analyses a ON a.id = d.analysis_id
    WHERE d.id = dashboard_tabs.dashboard_id
      AND d.owner_id = auth.uid()
      AND a.owner_id = auth.uid()
  )
);

DROP POLICY IF EXISTS "owners manage own graphs" ON public.dashboard_graphs;
CREATE POLICY "owners manage own graphs"
ON public.dashboard_graphs
FOR ALL
TO authenticated
USING (
  auth.uid() = owner_id
  AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = dashboard_graphs.analysis_id AND a.owner_id = auth.uid()
  )
)
WITH CHECK (
  auth.uid() = owner_id
  AND EXISTS (
    SELECT 1 FROM public.analyses a
    WHERE a.id = dashboard_graphs.analysis_id AND a.owner_id = auth.uid()
  )
  AND (
    dashboard_graphs.tab_id IS NULL
    OR EXISTS (
      SELECT 1 FROM public.dashboard_tabs t
      JOIN public.analysis_dashboards d ON d.id = t.dashboard_id
      WHERE t.id = dashboard_graphs.tab_id
        AND t.owner_id = auth.uid()
        AND d.owner_id = auth.uid()
        AND d.analysis_id = dashboard_graphs.analysis_id
    )
  )
);
