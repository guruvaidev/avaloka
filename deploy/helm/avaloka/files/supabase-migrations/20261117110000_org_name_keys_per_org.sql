-- Unique department and role names per organisation, for every organisation whose
-- rows allow it (BG-05/06's departments_org_name_key and config_roles_org_role_dept_key).
--
-- BG-05/06 (20261103150000) creates both indexes only when no organisation holds a
-- duplicate name, and otherwise warns and skips the index for everyone. Duplicates
-- that differ are kept for an admin to merge (20261103145000), so one such
-- organisation used to leave every organisation without the index. Here an index
-- that is missing is created for every organisation except those that still hold a
-- duplicate, with a warning that says how many were left out.
--
-- Safe whether or not the earlier files ran, and on any re-run: a full index is left
-- as it is; a per-organisation one is rebuilt from the rows as they are now, and
-- becomes the full index once no organisation holds a duplicate. Re-apply this file
-- after an admin merges duplicates (docs/deployment.md).

DO $$
DECLARE
  left_out uuid[];
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                  WHERE c.oid = to_regclass('public.departments_org_name_key') AND i.indpred IS NULL) THEN
    SELECT coalesce(array_agg(DISTINCT organization_id ORDER BY organization_id), '{}') INTO left_out
      FROM (SELECT organization_id FROM public.departments WHERE organization_id IS NOT NULL
             GROUP BY organization_id, lower(department) HAVING count(*) > 1) d;
    DROP INDEX IF EXISTS public.departments_org_name_key;
    IF cardinality(left_out) = 0 THEN
      CREATE UNIQUE INDEX departments_org_name_key ON public.departments (organization_id, lower(department));
    ELSE
      RAISE WARNING 'departments_org_name_key created without % organisation(s) that still have duplicate department names',
        cardinality(left_out);
      EXECUTE format('CREATE UNIQUE INDEX departments_org_name_key ON public.departments (organization_id, lower(department)) '
                     'WHERE organization_id <> ALL (%L::uuid[])', left_out);
    END IF;
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                  WHERE c.oid = to_regclass('public.config_roles_org_role_dept_key') AND i.indpred IS NULL) THEN
    SELECT coalesce(array_agg(DISTINCT organization_id ORDER BY organization_id), '{}') INTO left_out
      FROM (SELECT organization_id FROM public.config_roles WHERE organization_id IS NOT NULL
             GROUP BY organization_id, lower(role), lower(department) HAVING count(*) > 1) r;
    DROP INDEX IF EXISTS public.config_roles_org_role_dept_key;
    IF cardinality(left_out) = 0 THEN
      CREATE UNIQUE INDEX config_roles_org_role_dept_key ON public.config_roles (organization_id, lower(role), lower(department));
    ELSE
      RAISE WARNING 'config_roles_org_role_dept_key created without % organisation(s) that still have duplicate role names',
        cardinality(left_out);
      EXECUTE format('CREATE UNIQUE INDEX config_roles_org_role_dept_key ON public.config_roles (organization_id, lower(role), lower(department)) '
                     'WHERE organization_id <> ALL (%L::uuid[])', left_out);
    END IF;
  END IF;
END $$;
