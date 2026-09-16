CREATE TABLE IF NOT EXISTS public.plan_change_log (
  id uuid primary key default gen_random_uuid(),
  owner_profile_id uuid,
  organization_id uuid,
  from_plan text,
  to_plan text,
  from_plan_name text,
  to_plan_name text,
  status text,
  payment_provider text,
  subscription_id text,
  created_at timestamptz not null default now()
);

CREATE INDEX IF NOT EXISTS plan_change_log_owner_idx ON public.plan_change_log (owner_profile_id, created_at DESC);
CREATE INDEX IF NOT EXISTS plan_change_log_org_idx ON public.plan_change_log (organization_id, created_at DESC);

GRANT SELECT ON public.plan_change_log TO authenticated;
GRANT ALL ON public.plan_change_log TO service_role;
ALTER TABLE public.plan_change_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "plan_change_log_read_own" ON public.plan_change_log;
CREATE POLICY "plan_change_log_read_own" ON public.plan_change_log
  FOR SELECT TO authenticated
  USING (
    owner_profile_id = public.current_profile_id()
    OR (organization_id IS NOT NULL AND organization_id = public.current_org_id())
  );