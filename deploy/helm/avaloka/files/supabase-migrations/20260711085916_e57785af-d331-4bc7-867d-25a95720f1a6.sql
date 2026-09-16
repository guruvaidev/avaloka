
-- Enterprise billing: make organizations the single source of truth
ALTER TABLE public.organizations
  ADD COLUMN IF NOT EXISTS current_period_start timestamptz,
  ADD COLUMN IF NOT EXISTS unit_amount_cents integer,
  ADD COLUMN IF NOT EXISTS billing_interval text,
  ADD COLUMN IF NOT EXISTS card_brand text,
  ADD COLUMN IF NOT EXISTS card_last4 text,
  ADD COLUMN IF NOT EXISTS card_exp_month integer,
  ADD COLUMN IF NOT EXISTS card_exp_year integer,
  ADD COLUMN IF NOT EXISTS stripe_default_payment_method_id text,
  ADD COLUMN IF NOT EXISTS billing_email text,
  ADD COLUMN IF NOT EXISTS billing_name text;

ALTER TABLE public.enterprise_licenses
  ADD COLUMN IF NOT EXISTS stripe_subscription_id text;
