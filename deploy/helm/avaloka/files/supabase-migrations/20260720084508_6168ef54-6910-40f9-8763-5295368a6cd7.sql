
CREATE OR REPLACE FUNCTION public.seed_default_config_roles(_org_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  modules text[] := ARRAY['User Management','Analysis','Projects','Dashboard','Data Configuration'];
  actions text[] := ARRAY['view','create','edit','approve'];
  admin_perms text[] := '{}';
  analyst_perms text[] := '{}';
  m text;
  a text;
BEGIN
  FOREACH m IN ARRAY modules LOOP
    FOREACH a IN ARRAY actions LOOP
      admin_perms := admin_perms || (m || ':' || a);
    END LOOP;
    analyst_perms := analyst_perms || (m || ':view');
  END LOOP;

  INSERT INTO public.config_roles (role, department, permissions, status, organization_id)
  SELECT 'Admin', 'General', admin_perms, 'Active', _org_id
  WHERE NOT EXISTS (
    SELECT 1 FROM public.config_roles
    WHERE organization_id = _org_id AND lower(role) = 'admin'
  );

  INSERT INTO public.config_roles (role, department, permissions, status, organization_id)
  SELECT 'Analyst', 'General', analyst_perms, 'Active', _org_id
  WHERE NOT EXISTS (
    SELECT 1 FROM public.config_roles
    WHERE organization_id = _org_id AND lower(role) = 'analyst'
  );
END;
$$;

CREATE OR REPLACE FUNCTION public.trg_seed_default_config_roles()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  PERFORM public.seed_default_config_roles(NEW.id);
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS seed_default_config_roles_on_org ON public.organizations;
CREATE TRIGGER seed_default_config_roles_on_org
AFTER INSERT ON public.organizations
FOR EACH ROW EXECUTE FUNCTION public.trg_seed_default_config_roles();

-- Backfill any existing organizations with no roles configured
DO $$
DECLARE
  o record;
BEGIN
  FOR o IN
    SELECT id FROM public.organizations org
    WHERE NOT EXISTS (SELECT 1 FROM public.config_roles r WHERE r.organization_id = org.id)
  LOOP
    PERFORM public.seed_default_config_roles(o.id);
  END LOOP;
END $$;
