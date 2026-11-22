-- Check the source analysis before its project (and inherited permissions) changes.
CREATE OR REPLACE FUNCTION public.require_analysis_edit_for_move()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  IF NEW.project_id IS DISTINCT FROM OLD.project_id
     AND auth.uid() IS NOT NULL
     AND public.current_plan_tier() IS DISTINCT FROM 'enterprise'
     AND coalesce(public.analysis_share_access_level(OLD.id), 0) < 3 THEN
    RAISE EXCEPTION 'Edit access is required to move this analysis'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS require_analysis_edit_for_move ON public.analyses;
CREATE TRIGGER require_analysis_edit_for_move
  BEFORE UPDATE OF project_id ON public.analyses
  FOR EACH ROW EXECUTE FUNCTION public.require_analysis_edit_for_move();
