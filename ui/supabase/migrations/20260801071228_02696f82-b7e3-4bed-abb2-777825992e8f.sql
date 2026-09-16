CREATE OR REPLACE FUNCTION public.seed_default_config_roles(_org_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $function$
DECLARE
  modules text[] := ARRAY['User Management','Analysis','Projects','Dashboard','Data Configuration'];
  actions text[] := ARRAY['view','create','edit','approve'];
  admin_perms text[] := '{}';
  analyst_perms text[] := '{}';
  m text;
  a text;
BEGIN
  -- Default division
  INSERT INTO public.departments (organization_id, department, code, employees, status)
  SELECT _org_id, 'Management', 'MGMT', 0, 'Active'
  WHERE NOT EXISTS (
    SELECT 1 FROM public.departments
    WHERE organization_id = _org_id AND lower(department) = 'management'
  );

  FOREACH m IN ARRAY modules LOOP
    FOREACH a IN ARRAY actions LOOP
      admin_perms := admin_perms || (m || ':' || a);
    END LOOP;
    analyst_perms := analyst_perms || (m || ':view');
  END LOOP;

  INSERT INTO public.config_roles (role, department, permissions, status, organization_id)
  SELECT 'Admin', 'Management', admin_perms, 'Active', _org_id
  WHERE NOT EXISTS (
    SELECT 1 FROM public.config_roles
    WHERE organization_id = _org_id AND lower(role) = 'admin'
  );

  INSERT INTO public.config_roles (role, department, permissions, status, organization_id)
  SELECT 'Analyst', 'Management', analyst_perms, 'Active', _org_id
  WHERE NOT EXISTS (
    SELECT 1 FROM public.config_roles
    WHERE organization_id = _org_id AND lower(role) = 'analyst'
  );
END;
$function$;

DROP TRIGGER IF EXISTS trg_seed_default_config_roles_after_insert ON public.organizations;
CREATE TRIGGER trg_seed_default_config_roles_after_insert
AFTER INSERT ON public.organizations
FOR EACH ROW EXECUTE FUNCTION public.trg_seed_default_config_roles();