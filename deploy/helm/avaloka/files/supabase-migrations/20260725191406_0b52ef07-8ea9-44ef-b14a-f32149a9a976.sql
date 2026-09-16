REVOKE ALL ON FUNCTION public.get_my_report_access(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_my_report_access(uuid) FROM anon;
REVOKE ALL ON FUNCTION public.get_my_report_access(uuid) FROM authenticated;

-- Ensure only the function owner can execute it (internal policy use remains unaffected)
ALTER FUNCTION public.get_my_report_access(uuid) OWNER TO postgres;
