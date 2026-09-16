
-- Drop the overly broad org-wide read policy on profiles
DROP POLICY IF EXISTS "Org members read profiles" ON public.profiles;

-- Admins can read all profiles in their org (for admin tooling)
CREATE POLICY "Admins read org profiles"
ON public.profiles
FOR SELECT
TO authenticated
USING (
  organization_id IS NOT NULL
  AND organization_id = public.current_org_id()
  AND public.has_role(auth.uid(), 'admin')
);

-- Safe directory view for org members: only non-sensitive display fields
CREATE OR REPLACE VIEW public.org_member_directory
WITH (security_invoker = true)
AS
SELECT
  p.id,
  p.user_id,
  p.organization_id,
  p.first_name,
  p.last_name,
  p.full_name,
  p.avatar_url,
  p.company_name,
  p.company_slug,
  p.tagline,
  p.country,
  p.timezone
FROM public.profiles p
WHERE p.organization_id IS NOT NULL
  AND p.organization_id = public.current_org_id();

GRANT SELECT ON public.org_member_directory TO authenticated;
