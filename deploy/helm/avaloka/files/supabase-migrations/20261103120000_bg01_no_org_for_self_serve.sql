-- BG-01: Free / Professional users must have profiles.organization_id = NULL.
--
-- Before: handle_new_user() (20260907000000_fix_new_user_org_provisioning)
-- pre-allocated an organization_id and inserted an organizations row for
-- EVERY self-serve signup.
--
-- After:
--   1. handle_new_user(): self-serve signup -> organization_id NULL, no org row.
--      Invited users (organization_id in user metadata) keep the inviter's org,
--      and pending app_users invites are still linked to the new auth user.
--      Orgs are now created only on Enterprise upgrade (upgradeToEnterprise,
--      stripe-webhook / paypal-webhook handleCheckoutCompleted), which already
--      create the org when profile.organization_id is NULL.
--   2. reports: organization_id becomes nullable, and the INSERT / UPDATE
--      policies accept a NULL org for users who have none
--      (org users must still write into their own org).
--
-- Existing users' auto-created orgs are NOT touched here.
-- No BEGIN/COMMIT: apply with psql --single-transaction (dry run: -c BEGIN -f ... -c ROLLBACK).


-- 1) ──────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.handle_new_user()
  RETURNS trigger
  LANGUAGE plpgsql
  SECURITY DEFINER
  SET search_path TO 'public','auth'
AS $function$
DECLARE
  v_org_id    uuid;
  v_full_name text;
BEGIN
  -- Invited users carry the inviter's org in metadata; self-serve users get NULL.
  v_org_id    := NULLIF(NEW.raw_user_meta_data->>'organization_id','')::uuid;
  v_full_name := NEW.raw_user_meta_data->>'full_name';

  INSERT INTO public.profiles (user_id, organization_id, full_name)
  VALUES (NEW.id, v_org_id, v_full_name)
  ON CONFLICT (user_id) DO UPDATE
    SET organization_id = COALESCE(public.profiles.organization_id, EXCLUDED.organization_id),
        full_name       = COALESCE(public.profiles.full_name, EXCLUDED.full_name);

  -- Link any pre-created (invited) app_users row to this auth user.
  UPDATE public.app_users
     SET auth_user_id = NEW.id,
         accepted_at  = now(),
         status       = 'Active'
   WHERE lower(email) = lower(NEW.email)
     AND auth_user_id IS NULL;

  RETURN NEW;
END;
$function$;

-- 2) ──────────────────────────────────────────────────────────────────────
ALTER TABLE public.reports ALTER COLUMN organization_id DROP NOT NULL;

DROP POLICY IF EXISTS "Org members insert reports" ON public.reports;
CREATE POLICY "Org members insert reports" ON public.reports
  FOR INSERT TO authenticated
  WITH CHECK (
    created_by = auth.uid()
    AND organization_id IS NOT DISTINCT FROM public.current_org_id()
  );

DROP POLICY IF EXISTS "Creators update their reports" ON public.reports;
CREATE POLICY "Creators update their reports" ON public.reports
  FOR UPDATE TO authenticated
  USING (
    created_by = auth.uid()
    AND organization_id IS NOT DISTINCT FROM public.current_org_id()
  )
  WITH CHECK (
    created_by = auth.uid()
    AND organization_id IS NOT DISTINCT FROM public.current_org_id()
  );

