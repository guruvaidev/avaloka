
CREATE TABLE public.payment_methods (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_profile_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  organization_id uuid REFERENCES public.organizations(id) ON DELETE SET NULL,
  provider text NOT NULL CHECK (provider IN ('stripe','paypal')),
  provider_customer_id text,
  provider_payment_method_id text,
  provider_subscription_id text,
  brand text,
  last4 text,
  exp_month integer,
  exp_year integer,
  cardholder text,
  status text NOT NULL DEFAULT 'inactive' CHECK (status IN ('active','inactive')),
  is_default boolean NOT NULL DEFAULT false,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.payment_methods TO authenticated;
GRANT ALL ON public.payment_methods TO service_role;

ALTER TABLE public.payment_methods ENABLE ROW LEVEL SECURITY;

CREATE POLICY "payment_methods_owner_select" ON public.payment_methods
  FOR SELECT TO authenticated
  USING (owner_profile_id = public.current_profile_id());

CREATE POLICY "payment_methods_owner_insert" ON public.payment_methods
  FOR INSERT TO authenticated
  WITH CHECK (owner_profile_id = public.current_profile_id());

CREATE POLICY "payment_methods_owner_update" ON public.payment_methods
  FOR UPDATE TO authenticated
  USING (owner_profile_id = public.current_profile_id())
  WITH CHECK (owner_profile_id = public.current_profile_id());

CREATE POLICY "payment_methods_owner_delete" ON public.payment_methods
  FOR DELETE TO authenticated
  USING (owner_profile_id = public.current_profile_id());

CREATE UNIQUE INDEX payment_methods_unique_provider_pm
  ON public.payment_methods (owner_profile_id, provider, provider_payment_method_id)
  WHERE provider_payment_method_id IS NOT NULL;

CREATE UNIQUE INDEX payment_methods_one_active_per_owner
  ON public.payment_methods (owner_profile_id)
  WHERE status = 'active';

CREATE TRIGGER payment_methods_set_updated_at
  BEFORE UPDATE ON public.payment_methods
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
