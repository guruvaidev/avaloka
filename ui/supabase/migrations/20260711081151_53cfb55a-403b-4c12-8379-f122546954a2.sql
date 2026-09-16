
-- 1. organizations: subscription lifecycle columns (single source of truth)
ALTER TABLE public.organizations
  ADD COLUMN IF NOT EXISTS plan text,
  ADD COLUMN IF NOT EXISTS payment_provider text,
  ADD COLUMN IF NOT EXISTS payment_status text,
  ADD COLUMN IF NOT EXISTS subscription_status text,
  ADD COLUMN IF NOT EXISTS subscription_active boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS trial_ends_at timestamptz,
  ADD COLUMN IF NOT EXISTS current_period_end timestamptz,
  ADD COLUMN IF NOT EXISTS stripe_customer_id text,
  ADD COLUMN IF NOT EXISTS stripe_subscription_id text,
  ADD COLUMN IF NOT EXISTS stripe_price_id text,
  ADD COLUMN IF NOT EXISTS paypal_subscription_id text,
  ADD COLUMN IF NOT EXISTS paypal_payer_id text,
  ADD COLUMN IF NOT EXISTS monthly_price_cents integer NOT NULL DEFAULT 19900,
  ADD COLUMN IF NOT EXISTS currency text NOT NULL DEFAULT 'USD',
  ADD COLUMN IF NOT EXISTS canceled_at timestamptz;

CREATE INDEX IF NOT EXISTS organizations_stripe_subscription_id_idx
  ON public.organizations(stripe_subscription_id) WHERE stripe_subscription_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS organizations_paypal_subscription_id_idx
  ON public.organizations(paypal_subscription_id) WHERE paypal_subscription_id IS NOT NULL;

-- 2. profiles: mirror flags used by the app to decide "show Billing vs Choose Plan"
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS full_name text,
  ADD COLUMN IF NOT EXISTS selected_plan text,
  ADD COLUMN IF NOT EXISTS subscription_active boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS subscription_status text,
  ADD COLUMN IF NOT EXISTS trial_ends_at timestamptz,
  ADD COLUMN IF NOT EXISTS next_billing_date timestamptz,
  ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS stripe_customer_id text,
  ADD COLUMN IF NOT EXISTS stripe_subscription_id text,
  ADD COLUMN IF NOT EXISTS payment_provider text,
  ADD COLUMN IF NOT EXISTS user_id uuid;

-- 3. customer_accounts (billing profile per owner)
CREATE TABLE IF NOT EXISTS public.customer_accounts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_user_id uuid NOT NULL UNIQUE REFERENCES public.profiles(id) ON DELETE CASCADE,
  company_name text,
  billing_address text,
  vat_number text,
  country text,
  billing_email text,
  plan_type text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.customer_accounts TO authenticated;
GRANT ALL ON public.customer_accounts TO service_role;
ALTER TABLE public.customer_accounts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "owners read own customer_account"
  ON public.customer_accounts FOR SELECT
  TO authenticated
  USING (owner_user_id = public.current_profile_id());
CREATE POLICY "owners write own customer_account"
  ON public.customer_accounts FOR ALL
  TO authenticated
  USING (owner_user_id = public.current_profile_id())
  WITH CHECK (owner_user_id = public.current_profile_id());

CREATE TRIGGER customer_accounts_updated_at
  BEFORE UPDATE ON public.customer_accounts
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- 4. enterprise_licenses
CREATE TABLE IF NOT EXISTS public.enterprise_licenses (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_account_id uuid NOT NULL UNIQUE REFERENCES public.customer_accounts(id) ON DELETE CASCADE,
  organization_id uuid REFERENCES public.organizations(id) ON DELETE SET NULL,
  serial_number text NOT NULL UNIQUE,
  status text NOT NULL DEFAULT 'active',
  issued_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz,
  created_by uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  payment_amount numeric NOT NULL DEFAULT 0,
  payment_status text NOT NULL DEFAULT 'unpaid',
  paid_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT enterprise_licenses_payment_status_check
    CHECK (payment_status IN ('unpaid','paid','pending','trial'))
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.enterprise_licenses TO authenticated;
GRANT ALL ON public.enterprise_licenses TO service_role;
ALTER TABLE public.enterprise_licenses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "owners read own license"
  ON public.enterprise_licenses FOR SELECT
  TO authenticated
  USING (
    customer_account_id IN (
      SELECT id FROM public.customer_accounts
      WHERE owner_user_id = public.current_profile_id()
    )
  );

CREATE TRIGGER enterprise_licenses_updated_at
  BEFORE UPDATE ON public.enterprise_licenses
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- 5. billing_invoices (history for Billing History table)
CREATE TABLE IF NOT EXISTS public.billing_invoices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
  provider text NOT NULL,
  provider_invoice_id text,
  invoice_number text,
  amount_cents integer NOT NULL DEFAULT 0,
  currency text NOT NULL DEFAULT 'USD',
  status text NOT NULL DEFAULT 'paid',
  paid_at timestamptz,
  period_start timestamptz,
  period_end timestamptz,
  hosted_url text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (provider, provider_invoice_id)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.billing_invoices TO authenticated;
GRANT ALL ON public.billing_invoices TO service_role;
ALTER TABLE public.billing_invoices ENABLE ROW LEVEL SECURITY;

CREATE POLICY "org members read invoices"
  ON public.billing_invoices FOR SELECT
  TO authenticated
  USING (organization_id = public.current_org_id());
