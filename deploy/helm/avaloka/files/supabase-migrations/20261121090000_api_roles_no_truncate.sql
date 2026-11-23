-- The API roles (anon, authenticated) cannot TRUNCATE any public table.
--
-- Row-level security does not apply to TRUNCATE, so a role that holds the
-- privilege can empty a table whatever its policies say. Supabase's default
-- privileges grant it on every new public table, so each table created after
-- 20261103130000 (which revoked it from every table then present) got it back:
-- the ml_jobs, ml_models, ml_endpoints and ml_cross_validation tables of
-- 20261117130000 among them. Same statement as that migration, over every
-- public table; revoking a privilege that is not held changes nothing, so the
-- file is safe to run on any install. No BEGIN/COMMIT: the chart's migrate hook
-- runs each file in one transaction with its schema_migrations record.

DO $$
DECLARE r record;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
    EXECUTE format('REVOKE TRUNCATE ON TABLE public.%I FROM anon, authenticated', r.tablename);
  END LOOP;
END $$;
