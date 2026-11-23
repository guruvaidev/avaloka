-- BG-05 / BG-06 (#646): departments and roles become linked
-- records. Unique names per organisation; app_users and config_roles get
-- department_id / role_id; a user's role, department and permissions follow the
-- linked role; renames propagate. Existing app_users text is backfilled.
--
-- Reconciled for develop-1.6 (bugfix/1.6-migration-fix-reconciled):
--   * renumbered from 20261101120000, which collided with #649's and #650's
--     migrations, to run after D-2; both copies; inner BEGIN/COMMIT removed;
--   * the unique indexes are created only when no duplicates exist; otherwise a
--     WARNING names the index and it is skipped: under the hook's ON_ERROR_STOP
--     a duplicate would have failed the whole upgrade;
--   * the backfill (new departments and roles, and the rewrite of users' role,
--     department and permissions from the linked role) is limited to Enterprise
--     organisations, by the subscription rules of current_plan_tier() (#650):
--     organisation structure is an Enterprise feature (owner rule). Linking
--     existing rows by id is done for every organisation;
--   * the app_users trigger normalises role, department and permissions only on a
--     real change (insert, a name change, or a role_id change), and is named to
--     fire before any enforce_* plan trigger, so a plan check sees the final row;
--   * search_path is empty on every function.

CREATE OR REPLACE FUNCTION public.role_permissions_jsonb(p text[]) RETURNS jsonb LANGUAGE sql IMMUTABLE SET search_path = '' AS $$
  SELECT coalesce(jsonb_object_agg(section, acts), '{}'::jsonb)
  FROM (SELECT split_part(x, ':', 1) AS section, jsonb_object_agg(initcap(split_part(x, ':', 2)), true) AS acts
        FROM unnest(coalesce(p, '{}'::text[])) AS x WHERE position(':' IN x) > 0 GROUP BY 1) s
$$;
-- Unique names per organisation, only where the data already allows it. A
-- duplicate must not abort an upgrade (ON_ERROR_STOP): warn, skip the index, and
-- leave the clean-up to the organisation. The backfill below never adds one.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM public.departments
              GROUP BY organization_id, lower(department) HAVING count(*) > 1) THEN
    RAISE WARNING 'departments_org_name_key not created: duplicate department names exist in % organisation(s)',
      (SELECT count(DISTINCT organization_id) FROM (SELECT organization_id FROM public.departments
         GROUP BY organization_id, lower(department) HAVING count(*) > 1) d);
  ELSE
    CREATE UNIQUE INDEX IF NOT EXISTS departments_org_name_key ON public.departments (organization_id, lower(department));
  END IF;
  IF EXISTS (SELECT 1 FROM public.config_roles
              GROUP BY organization_id, lower(role), lower(department) HAVING count(*) > 1) THEN
    RAISE WARNING 'config_roles_org_role_dept_key not created: duplicate role names exist in % organisation(s)',
      (SELECT count(DISTINCT organization_id) FROM (SELECT organization_id FROM public.config_roles
         GROUP BY organization_id, lower(role), lower(department) HAVING count(*) > 1) r);
  ELSE
    CREATE UNIQUE INDEX IF NOT EXISTS config_roles_org_role_dept_key ON public.config_roles (organization_id, lower(role), lower(department));
  END IF;
END $$;

-- Enterprise organisations, by the rules of public.current_plan_tier() (#650):
-- the organisation's latest subscription is active, trialing or inside its trial
-- window, and its plan is enterprise by type or name. Dropped at commit.
CREATE TEMP TABLE _bg0506_enterprise_orgs ON COMMIT DROP AS
SELECT s.organization_id
  FROM (SELECT DISTINCT ON (organization_id) organization_id, lower(status) AS status, trial_ends_at, plan_id
          FROM public.subscriptions WHERE organization_id IS NOT NULL
         ORDER BY organization_id, created_at DESC) s
  LEFT JOIN public.plans pl ON pl.id = s.plan_id
 WHERE (s.status IN ('trialing', 'active', 'trial')
        OR (s.trial_ends_at > now() AND s.status NOT IN ('canceled', 'cancelled', 'unpaid', 'incomplete_expired')))
   AND (lower(pl.plan_type::text) = 'enterprise' OR lower(coalesce(pl.name, '')) LIKE '%enterprise%');

INSERT INTO public.departments (organization_id, department, code)
SELECT DISTINCT ON (au.organization_id, lower(au.department)) au.organization_id, au.department,
       coalesce(nullif(upper(left(regexp_replace(au.department, '[^A-Za-z0-9]', '', 'g'), 6)), ''), 'DEPT')
FROM public.app_users au
WHERE nullif(btrim(au.department), '') IS NOT NULL
  AND au.organization_id IN (SELECT organization_id FROM _bg0506_enterprise_orgs)
  AND NOT EXISTS (SELECT 1 FROM public.departments d WHERE d.organization_id = au.organization_id AND lower(d.department) = lower(au.department))
ORDER BY au.organization_id, lower(au.department), au.created_at;
INSERT INTO public.config_roles (organization_id, role, department, permissions)
SELECT DISTINCT ON (au.organization_id, lower(au.role), lower(au.department)) au.organization_id, au.role, au.department,
       coalesce(array(SELECT e.key || ':' || lower(f.key)
         FROM jsonb_each(CASE WHEN jsonb_typeof(au.permissions) = 'object' THEN au.permissions ELSE '{}'::jsonb END) e,
              jsonb_each(CASE WHEN jsonb_typeof(e.value) = 'object' THEN e.value ELSE '{}'::jsonb END) f
         WHERE f.value = 'true'::jsonb ORDER BY 1), '{}'::text[])
FROM public.app_users au
WHERE nullif(btrim(au.role), '') IS NOT NULL AND nullif(btrim(au.department), '') IS NOT NULL
  AND au.organization_id IN (SELECT organization_id FROM _bg0506_enterprise_orgs)
  AND NOT EXISTS (SELECT 1 FROM public.config_roles cr WHERE cr.organization_id = au.organization_id
                  AND lower(cr.role) = lower(au.role) AND lower(cr.department) = lower(au.department))
ORDER BY au.organization_id, lower(au.role), lower(au.department), au.created_at;
ALTER TABLE public.config_roles ADD COLUMN IF NOT EXISTS department_id uuid REFERENCES public.departments(id) ON DELETE CASCADE;
ALTER TABLE public.app_users ADD COLUMN IF NOT EXISTS department_id uuid REFERENCES public.departments(id) ON DELETE SET NULL;
ALTER TABLE public.app_users ADD COLUMN IF NOT EXISTS role_id uuid REFERENCES public.config_roles(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS config_roles_department_id_idx ON public.config_roles (department_id);
CREATE INDEX IF NOT EXISTS app_users_department_id_idx ON public.app_users (department_id);
CREATE INDEX IF NOT EXISTS app_users_role_id_idx ON public.app_users (role_id);
UPDATE public.config_roles cr SET department_id = d.id FROM public.departments d
 WHERE d.organization_id = cr.organization_id AND lower(d.department) = lower(cr.department) AND cr.department_id IS DISTINCT FROM d.id;
UPDATE public.app_users au SET department_id = d.id FROM public.departments d
 WHERE d.organization_id = au.organization_id AND lower(d.department) = lower(au.department) AND au.department_id IS DISTINCT FROM d.id;
UPDATE public.app_users au SET role_id = cr.id FROM public.config_roles cr
 WHERE cr.organization_id = au.organization_id AND lower(cr.role) = lower(au.role)
   AND lower(cr.department) = lower(au.department) AND au.role_id IS DISTINCT FROM cr.id;
CREATE OR REPLACE FUNCTION public.config_roles_resolve_department() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
BEGIN
  IF TG_OP = 'INSERT' OR NEW.department IS DISTINCT FROM OLD.department OR NEW.organization_id IS DISTINCT FROM OLD.organization_id THEN
    SELECT d.id INTO NEW.department_id FROM public.departments d
     WHERE d.organization_id = NEW.organization_id AND lower(d.department) = lower(NEW.department);
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS trg_config_roles_resolve_department ON public.config_roles;
CREATE TRIGGER trg_config_roles_resolve_department BEFORE INSERT OR UPDATE ON public.config_roles
  FOR EACH ROW EXECUTE FUNCTION public.config_roles_resolve_department();
CREATE OR REPLACE FUNCTION public.app_users_sync_role_department() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
DECLARE cr public.config_roles%ROWTYPE; names_changed boolean;
BEGIN
  names_changed := TG_OP = 'INSERT' OR NEW.role IS DISTINCT FROM OLD.role
    OR NEW.department IS DISTINCT FROM OLD.department OR NEW.organization_id IS DISTINCT FROM OLD.organization_id;
  IF names_changed AND (TG_OP = 'INSERT' AND NEW.role_id IS NULL OR TG_OP = 'UPDATE' AND NEW.role_id IS NOT DISTINCT FROM OLD.role_id) THEN
    SELECT c.id INTO NEW.role_id FROM public.config_roles c
     WHERE c.organization_id = NEW.organization_id AND lower(c.role) = lower(NEW.role) AND lower(c.department) = lower(NEW.department);
  END IF;
  IF names_changed AND (TG_OP = 'INSERT' AND NEW.department_id IS NULL OR TG_OP = 'UPDATE' AND NEW.department_id IS NOT DISTINCT FROM OLD.department_id) THEN
    SELECT d.id INTO NEW.department_id FROM public.departments d
     WHERE d.organization_id = NEW.organization_id AND lower(d.department) = lower(NEW.department);
  END IF;
  -- Normalise only on a real change: an insert, a name change, or a new role_id.
  -- An update that touches neither (e.g. team membership) passes through as written.
  IF NEW.role_id IS NOT NULL
     AND (TG_OP = 'INSERT' OR names_changed OR NEW.role_id IS DISTINCT FROM OLD.role_id) THEN
    SELECT * INTO cr FROM public.config_roles c WHERE c.id = NEW.role_id;
    IF NOT FOUND OR cr.organization_id <> NEW.organization_id THEN
      RAISE EXCEPTION 'Role % does not belong to organization %', NEW.role_id, NEW.organization_id;
    END IF;
    NEW.role := cr.role; NEW.department := cr.department;
    NEW.department_id := coalesce(cr.department_id, NEW.department_id);
    NEW.permissions := public.role_permissions_jsonb(cr.permissions);
  END IF;
  RETURN NEW;
END $$;
-- BEFORE triggers fire in name order: "app_users_sync_..." runs before any
-- "enforce_*" plan trigger, which then checks the normalised row.
DROP TRIGGER IF EXISTS trg_app_users_sync_role_department ON public.app_users;
DROP TRIGGER IF EXISTS app_users_sync_role_department ON public.app_users;
CREATE TRIGGER app_users_sync_role_department BEFORE INSERT OR UPDATE ON public.app_users
  FOR EACH ROW EXECUTE FUNCTION public.app_users_sync_role_department();
CREATE OR REPLACE FUNCTION public.config_roles_propagate_to_users() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
BEGIN
  UPDATE public.app_users SET role = NEW.role, department = NEW.department,
         permissions = public.role_permissions_jsonb(NEW.permissions), updated_at = now()
   WHERE role_id = NEW.id;
  RETURN NULL;
END $$;
DROP TRIGGER IF EXISTS trg_config_roles_propagate_to_users ON public.config_roles;
CREATE TRIGGER trg_config_roles_propagate_to_users AFTER UPDATE OF role, department, permissions ON public.config_roles
  FOR EACH ROW EXECUTE FUNCTION public.config_roles_propagate_to_users();
CREATE OR REPLACE FUNCTION public.departments_propagate_rename() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
BEGIN
  IF NEW.department IS DISTINCT FROM OLD.department THEN
    UPDATE public.config_roles SET department = NEW.department, updated_at = now() WHERE department_id = NEW.id;
    UPDATE public.app_users SET department = NEW.department, updated_at = now() WHERE department_id = NEW.id AND role_id IS NULL;
  END IF;
  RETURN NULL;
END $$;
DROP TRIGGER IF EXISTS trg_departments_propagate_rename ON public.departments;
CREATE TRIGGER trg_departments_propagate_rename AFTER UPDATE OF department ON public.departments
  FOR EACH ROW EXECUTE FUNCTION public.departments_propagate_rename();
-- Rewrites each linked user's role, department and permissions from the role:
-- Enterprise organisations only (see the header).
UPDATE public.app_users au SET role = cr.role, department = cr.department,
       permissions = public.role_permissions_jsonb(cr.permissions)
  FROM public.config_roles cr
 WHERE cr.id = au.role_id
   AND au.organization_id IN (SELECT organization_id FROM _bg0506_enterprise_orgs);
DROP POLICY IF EXISTS "departments insert" ON public.departments;
DROP POLICY IF EXISTS "departments update" ON public.departments;
DROP POLICY IF EXISTS "departments delete" ON public.departments;
DROP POLICY IF EXISTS "departments select" ON public.departments;
