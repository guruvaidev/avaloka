GRANT SELECT, INSERT, UPDATE, DELETE ON public.profiles TO authenticated;
GRANT ALL ON public.profiles TO service_role;
GRANT INSERT, SELECT, UPDATE ON public.profiles TO supabase_auth_admin;