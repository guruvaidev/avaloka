
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- PROJECTS
CREATE TABLE public.projects (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  bookmarked BOOLEAN NOT NULL DEFAULT false,
  archived BOOLEAN NOT NULL DEFAULT false,
  pinned BOOLEAN NOT NULL DEFAULT false,
  deleted_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_projects_owner ON public.projects(owner_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_projects_name_trgm ON public.projects USING gin (name gin_trgm_ops);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.projects TO authenticated;
GRANT ALL ON public.projects TO service_role;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;

CREATE POLICY "owners read own projects" ON public.projects
  FOR SELECT TO authenticated USING (auth.uid() = owner_id);
CREATE POLICY "owners insert own projects" ON public.projects
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = owner_id);
CREATE POLICY "owners update own projects" ON public.projects
  FOR UPDATE TO authenticated USING (auth.uid() = owner_id) WITH CHECK (auth.uid() = owner_id);
CREATE POLICY "owners delete own projects" ON public.projects
  FOR DELETE TO authenticated USING (auth.uid() = owner_id);

-- ANALYSES
CREATE TABLE public.analyses (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  team_label TEXT NOT NULL DEFAULT 'Team',
  team_count INTEGER NOT NULL DEFAULT 0,
  created_by_name TEXT,
  created_by_role TEXT,
  created_by_initials TEXT,
  deleted_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_analyses_project ON public.analyses(project_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_analyses_owner ON public.analyses(owner_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.analyses TO authenticated;
GRANT ALL ON public.analyses TO service_role;
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "owners read own analyses" ON public.analyses
  FOR SELECT TO authenticated USING (auth.uid() = owner_id);
CREATE POLICY "owners insert own analyses" ON public.analyses
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = owner_id);
CREATE POLICY "owners update own analyses" ON public.analyses
  FOR UPDATE TO authenticated USING (auth.uid() = owner_id) WITH CHECK (auth.uid() = owner_id);
CREATE POLICY "owners delete own analyses" ON public.analyses
  FOR DELETE TO authenticated USING (auth.uid() = owner_id);

-- DASHBOARDS
CREATE TABLE public.analysis_dashboards (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  analysis_id UUID NOT NULL UNIQUE REFERENCES public.analyses(id) ON DELETE CASCADE,
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.analysis_dashboards TO authenticated;
GRANT ALL ON public.analysis_dashboards TO service_role;
ALTER TABLE public.analysis_dashboards ENABLE ROW LEVEL SECURITY;
CREATE POLICY "owners manage own dashboards" ON public.analysis_dashboards
  FOR ALL TO authenticated USING (auth.uid() = owner_id) WITH CHECK (auth.uid() = owner_id);

-- TABS
CREATE TABLE public.dashboard_tabs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  dashboard_id UUID NOT NULL REFERENCES public.analysis_dashboards(id) ON DELETE CASCADE,
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  tab_index INTEGER NOT NULL,
  title TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (dashboard_id, tab_index)
);
CREATE INDEX idx_dashboard_tabs_dash ON public.dashboard_tabs(dashboard_id);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.dashboard_tabs TO authenticated;
GRANT ALL ON public.dashboard_tabs TO service_role;
ALTER TABLE public.dashboard_tabs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "owners manage own tabs" ON public.dashboard_tabs
  FOR ALL TO authenticated USING (auth.uid() = owner_id) WITH CHECK (auth.uid() = owner_id);

-- GRAPHS
CREATE TABLE public.dashboard_graphs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  tab_id UUID NOT NULL REFERENCES public.dashboard_tabs(id) ON DELETE CASCADE,
  analysis_id UUID NOT NULL REFERENCES public.analyses(id) ON DELETE CASCADE,
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  graph_key TEXT NOT NULL,
  title TEXT NOT NULL,
  chart_type TEXT NOT NULL DEFAULT 'bar',
  config JSONB NOT NULL DEFAULT '{}'::jsonb,
  position INTEGER NOT NULL DEFAULT 0,
  placed BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_graphs_tab ON public.dashboard_graphs(tab_id);
CREATE INDEX idx_graphs_analysis ON public.dashboard_graphs(analysis_id);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.dashboard_graphs TO authenticated;
GRANT ALL ON public.dashboard_graphs TO service_role;
ALTER TABLE public.dashboard_graphs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "owners manage own graphs" ON public.dashboard_graphs
  FOR ALL TO authenticated USING (auth.uid() = owner_id) WITH CHECK (auth.uid() = owner_id);

-- updated_at trigger
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql SET search_path = public AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END; $$;

CREATE TRIGGER trg_projects_updated BEFORE UPDATE ON public.projects
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER trg_analyses_updated BEFORE UPDATE ON public.analyses
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER trg_dashboards_updated BEFORE UPDATE ON public.analysis_dashboards
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER trg_tabs_updated BEFORE UPDATE ON public.dashboard_tabs
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
CREATE TRIGGER trg_graphs_updated BEFORE UPDATE ON public.dashboard_graphs
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
