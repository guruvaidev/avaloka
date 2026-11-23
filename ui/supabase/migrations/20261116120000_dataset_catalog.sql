-- The personal dataset catalog (docs/design/1.7-DATA-PLATFORM.md part C; app/catalog/store.py).
--
-- 1. Five tables, the same shape SQLAlchemy Core creates for a self-hosted
--    install or a SQLite catalog: catalog_dataset, source_schema_snapshots
--    (item 10a's table, created here because it was on no branch),
--    catalog_dataset_version, catalog_dataset_use, catalog_term.
-- 2. Row-level security with exactly ONE predicate on every table, for reads
--    and writes alike: owner_user_id = auth.uid(). There is no project or
--    organisation policy in this migration. Those arrive with the Enterprise
--    shared catalog (16d) and only when `shared_catalog` is granted.
-- 3. A trigger that refuses any visibility but 'personal' unless the
--    deployment grants `shared_catalog` (public.catalog_shared_enabled()).
--    A direct database write cannot widen visibility either.
-- 4. The service role bypasses RLS as it does elsewhere; the backend scopes
--    every statement by owner itself (CatalogStore binds to one user).
--
-- Safe to run again.

CREATE TABLE IF NOT EXISTS public.catalog_dataset (
  id               text PRIMARY KEY,
  owner_user_id    varchar(128) NOT NULL,
  organization_id  varchar(128),
  project_id       varchar(128),
  kind             varchar(32)  NOT NULL
                   CHECK (kind IN ('file','object','db_table','warehouse_table','lake_table','kafka_window','derived')),
  locator          text NOT NULL DEFAULT '{}',
  locator_key      varchar(64)  NOT NULL,
  display_name     text NOT NULL DEFAULT '',
  visibility       varchar(16)  NOT NULL DEFAULT 'personal'
                   CHECK (visibility IN ('personal','project','organization')),
  state            varchar(16)  NOT NULL DEFAULT 'draft'
                   CHECK (state IN ('draft','certified','deprecated')),
  tags             text NOT NULL DEFAULT '[]',
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),
  deleted_at       timestamptz
);
CREATE INDEX IF NOT EXISTS ix_catalog_dataset_owner   ON public.catalog_dataset (owner_user_id, kind);
CREATE INDEX IF NOT EXISTS ix_catalog_dataset_locator ON public.catalog_dataset (owner_user_id, locator_key);

CREATE TABLE IF NOT EXISTS public.source_schema_snapshots (
  id             text PRIMARY KEY,
  owner_user_id  varchar(128) NOT NULL,
  fingerprint    varchar(64)  NOT NULL,
  columns        text NOT NULL DEFAULT '[]',
  column_count   integer NOT NULL DEFAULT 0,
  captured_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_source_schema_snapshots_owner ON public.source_schema_snapshots (owner_user_id, fingerprint);

CREATE TABLE IF NOT EXISTS public.catalog_dataset_version (
  id              text PRIMARY KEY,
  dataset_id      text NOT NULL REFERENCES public.catalog_dataset (id) ON DELETE CASCADE,
  owner_user_id   varchar(128) NOT NULL,
  version_no      integer NOT NULL,
  source_version  text,
  snapshot_id     text NOT NULL REFERENCES public.source_schema_snapshots (id),
  row_count       bigint,
  row_count_kind  varchar(16) NOT NULL DEFAULT 'unknown'
                  CHECK (row_count_kind IN ('exact','estimate','unknown')),
  bytes           bigint,
  run_id          text,
  captured_at     timestamptz NOT NULL DEFAULT now(),
  captured_by     varchar(128) NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_catalog_dataset_version_dataset ON public.catalog_dataset_version (dataset_id, version_no);

CREATE TABLE IF NOT EXISTS public.catalog_dataset_use (
  analysis_id    text NOT NULL,
  dataset_id     text NOT NULL REFERENCES public.catalog_dataset (id) ON DELETE CASCADE,
  owner_user_id  varchar(128) NOT NULL,
  version_id     text NOT NULL,
  used_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (analysis_id, dataset_id)
);
CREATE INDEX IF NOT EXISTS ix_catalog_dataset_use_dataset ON public.catalog_dataset_use (owner_user_id, dataset_id);

CREATE TABLE IF NOT EXISTS public.catalog_term (
  dataset_id     text NOT NULL REFERENCES public.catalog_dataset (id) ON DELETE CASCADE,
  term           text NOT NULL,
  owner_user_id  varchar(128) NOT NULL,
  column_digest  text,
  added_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (dataset_id, term)
);

-- Whether this deployment grants the shared catalog. Off until the Enterprise
-- shared catalog migration (16d) redefines it; the backend's editions matrix
-- (app/core/editions.py: shared_catalog) is the contract this mirrors.
CREATE OR REPLACE FUNCTION public.catalog_shared_enabled()
RETURNS boolean
LANGUAGE sql
STABLE
SET search_path = ''
AS $$ SELECT false $$;

CREATE OR REPLACE FUNCTION public.catalog_dataset_personal_only()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
  IF NEW.visibility IS DISTINCT FROM 'personal' AND NOT public.catalog_shared_enabled() THEN
    RAISE EXCEPTION 'dataset visibility % is part of the Enterprise shared catalog; this deployment keeps catalogs personal',
      NEW.visibility USING ERRCODE = 'check_violation';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS catalog_dataset_personal_only ON public.catalog_dataset;
CREATE TRIGGER catalog_dataset_personal_only
  BEFORE INSERT OR UPDATE OF visibility ON public.catalog_dataset
  FOR EACH ROW EXECUTE FUNCTION public.catalog_dataset_personal_only();

-- Row-level security: the owner, and nobody else. One predicate per table.
ALTER TABLE public.catalog_dataset          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.source_schema_snapshots  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.catalog_dataset_version  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.catalog_dataset_use      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.catalog_term             ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS catalog_dataset_owner ON public.catalog_dataset;
CREATE POLICY catalog_dataset_owner ON public.catalog_dataset
  FOR ALL TO authenticated
  USING (owner_user_id = (select auth.uid())::text)
  WITH CHECK (owner_user_id = (select auth.uid())::text);

DROP POLICY IF EXISTS source_schema_snapshots_owner ON public.source_schema_snapshots;
CREATE POLICY source_schema_snapshots_owner ON public.source_schema_snapshots
  FOR ALL TO authenticated
  USING (owner_user_id = (select auth.uid())::text)
  WITH CHECK (owner_user_id = (select auth.uid())::text);

DROP POLICY IF EXISTS catalog_dataset_version_owner ON public.catalog_dataset_version;
CREATE POLICY catalog_dataset_version_owner ON public.catalog_dataset_version
  FOR ALL TO authenticated
  USING (owner_user_id = (select auth.uid())::text)
  WITH CHECK (owner_user_id = (select auth.uid())::text);

DROP POLICY IF EXISTS catalog_dataset_use_owner ON public.catalog_dataset_use;
CREATE POLICY catalog_dataset_use_owner ON public.catalog_dataset_use
  FOR ALL TO authenticated
  USING (owner_user_id = (select auth.uid())::text)
  WITH CHECK (owner_user_id = (select auth.uid())::text);

DROP POLICY IF EXISTS catalog_term_owner ON public.catalog_term;
CREATE POLICY catalog_term_owner ON public.catalog_term
  FOR ALL TO authenticated
  USING (owner_user_id = (select auth.uid())::text)
  WITH CHECK (owner_user_id = (select auth.uid())::text);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.catalog_dataset, public.source_schema_snapshots,
  public.catalog_dataset_version, public.catalog_dataset_use, public.catalog_term TO authenticated;
