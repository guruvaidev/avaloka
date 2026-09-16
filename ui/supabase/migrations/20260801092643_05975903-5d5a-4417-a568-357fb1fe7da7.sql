DROP TRIGGER IF EXISTS trg_seed_default_config_roles ON public.organizations;
DROP TRIGGER IF EXISTS seed_default_config_roles_on_org ON public.organizations;
DROP TRIGGER IF EXISTS trg_seed_default_config_roles_after_insert ON public.organizations;
DROP FUNCTION IF EXISTS public.trg_seed_default_config_roles() CASCADE;
DROP FUNCTION IF EXISTS public.seed_default_config_roles(uuid) CASCADE;