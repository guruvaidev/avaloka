CREATE TYPE public.data_provider AS ENUM ('mysql','postgres','snowflake','mssql','clickhouse','mariadb');
CREATE TYPE public.data_source_status AS ENUM ('connected','disconnected','error');

CREATE TABLE public.data_sources (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider public.data_provider NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  domain TEXT,
  use_cases TEXT,
  status public.data_source_status NOT NULL DEFAULT 'disconnected',
  config JSONB NOT NULL DEFAULT '{}'::jsonb,
  last_tested_at TIMESTAMPTZ,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.data_sources TO authenticated;
GRANT ALL ON public.data_sources TO service_role;
ALTER TABLE public.data_sources ENABLE ROW LEVEL SECURITY;
CREATE POLICY "owners manage own data_sources" ON public.data_sources
  FOR ALL TO authenticated USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());
CREATE TRIGGER tr_data_sources_updated BEFORE UPDATE ON public.data_sources
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TABLE public.data_source_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  data_source_id UUID NOT NULL REFERENCES public.data_sources(id) ON DELETE CASCADE,
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  size_bytes BIGINT,
  selected BOOLEAN NOT NULL DEFAULT false,
  uploaded BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.data_source_files TO authenticated;
GRANT ALL ON public.data_source_files TO service_role;
ALTER TABLE public.data_source_files ENABLE ROW LEVEL SECURITY;
CREATE POLICY "owners manage own data_source_files" ON public.data_source_files
  FOR ALL TO authenticated USING (owner_id = auth.uid()) WITH CHECK (owner_id = auth.uid());
CREATE INDEX idx_data_source_files_ds ON public.data_source_files(data_source_id);