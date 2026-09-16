-- 1. report_comments: require real access to the report
DROP POLICY IF EXISTS "Report viewers can read comments" ON public.report_comments;
DROP POLICY IF EXISTS "Report viewers can add comments" ON public.report_comments;
DROP POLICY IF EXISTS "Authors can update their report comments" ON public.report_comments;

CREATE POLICY "Report viewers can read comments"
ON public.report_comments FOR SELECT TO authenticated
USING (public.get_my_report_access(report_id) <> 'none');

CREATE POLICY "Report viewers can add comments"
ON public.report_comments FOR INSERT TO authenticated
WITH CHECK (
  author_id = public.current_profile_id()
  AND public.get_my_report_access(report_id) IN ('comment', 'edit')
);

CREATE POLICY "Authors can update their report comments"
ON public.report_comments FOR UPDATE TO authenticated
USING (author_id = public.current_profile_id())
WITH CHECK (
  author_id = public.current_profile_id()
  AND public.get_my_report_access(report_id) <> 'none'
);

-- 2. profiles: drop org-wide read of sensitive columns
DROP POLICY IF EXISTS "Read profiles in same organization" ON public.profiles;

-- 3. get_report_teams: not callable by anonymous visitors
REVOKE EXECUTE ON FUNCTION public.get_report_teams(uuid[]) FROM anon, PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_report_teams(uuid[]) TO authenticated;