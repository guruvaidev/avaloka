
CREATE TABLE public.reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL,
  project_id uuid REFERENCES public.projects(id) ON DELETE CASCADE,
  analysis_id uuid NOT NULL REFERENCES public.analyses(id) ON DELETE CASCADE,
  title text NOT NULL DEFAULT 'Untitled report',
  dataset_label text,
  tab_id text,
  key_insights jsonb DEFAULT '[]'::jsonb,
  status text NOT NULL DEFAULT 'generated',
  created_by uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_reports_org ON public.reports(organization_id);
CREATE INDEX idx_reports_project ON public.reports(project_id);
CREATE INDEX idx_reports_analysis ON public.reports(analysis_id);
CREATE INDEX idx_reports_created_by ON public.reports(created_by);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.reports TO authenticated;
GRANT ALL ON public.reports TO service_role;

ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members view reports"
  ON public.reports FOR SELECT TO authenticated
  USING (organization_id = public.current_org_id());

CREATE POLICY "Org members insert reports"
  ON public.reports FOR INSERT TO authenticated
  WITH CHECK (organization_id = public.current_org_id() AND created_by = auth.uid());

CREATE POLICY "Creators update their reports"
  ON public.reports FOR UPDATE TO authenticated
  USING (created_by = auth.uid() AND organization_id = public.current_org_id())
  WITH CHECK (created_by = auth.uid() AND organization_id = public.current_org_id());

CREATE POLICY "Creators delete their reports"
  ON public.reports FOR DELETE TO authenticated
  USING (created_by = auth.uid() AND organization_id = public.current_org_id());

CREATE TRIGGER trg_reports_updated_at
  BEFORE UPDATE ON public.reports
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
