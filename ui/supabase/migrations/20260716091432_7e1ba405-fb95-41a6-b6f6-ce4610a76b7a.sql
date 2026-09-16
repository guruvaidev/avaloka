
DROP POLICY IF EXISTS "Owners read share links for their resources" ON public.resource_share_links;
DROP POLICY IF EXISTS "Owners insert share links for their resources" ON public.resource_share_links;
DROP POLICY IF EXISTS "Owners update share links for their resources" ON public.resource_share_links;
DROP POLICY IF EXISTS "Owners delete share links for their resources" ON public.resource_share_links;

CREATE POLICY "Owners read share links for their resources"
ON public.resource_share_links FOR SELECT
USING (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_share_links.resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_share_links.resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_share_links.resource_id AND r.created_by = auth.uid()))
);

CREATE POLICY "Owners insert share links for their resources"
ON public.resource_share_links FOR INSERT
WITH CHECK (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_share_links.resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_share_links.resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_share_links.resource_id AND r.created_by = auth.uid()))
);

CREATE POLICY "Owners update share links for their resources"
ON public.resource_share_links FOR UPDATE
USING (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_share_links.resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_share_links.resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_share_links.resource_id AND r.created_by = auth.uid()))
)
WITH CHECK (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_share_links.resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_share_links.resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_share_links.resource_id AND r.created_by = auth.uid()))
);

CREATE POLICY "Owners delete share links for their resources"
ON public.resource_share_links FOR DELETE
USING (
  (resource_type = 'analysis' AND EXISTS (SELECT 1 FROM public.analyses a WHERE a.id = resource_share_links.resource_id AND a.owner_id = public.current_profile_id()))
  OR (resource_type = 'project' AND EXISTS (SELECT 1 FROM public.projects p WHERE p.id = resource_share_links.resource_id AND p.owner_id = public.current_profile_id()))
  OR (resource_type = 'report' AND EXISTS (SELECT 1 FROM public.reports r WHERE r.id = resource_share_links.resource_id AND r.created_by = auth.uid()))
);
