-- Fix: new-user subscription flow — self-serve signups never got an
-- organizations row, so the Stripe webhook could not link a subscription.
--
-- Two coordinated function changes + a one-time backfill:
--   1. handle_new_user(): materialise the organization using the id already
--      pre-allocated onto the profile — for self-serve signups only.
--   2. ensure_org_owner_app_user(): make the owner INSERT idempotent on the
--      GLOBAL unique app_users_auth_user_id_key (the existing check was
--      org-scoped and missed rows under a different/stale org).
--   3. Backfill organizations rows for users already stuck with a phantom
--      organization_id (profile points at an org row that never existed).
--
-- prevent_profile_org_change() is intentionally LEFT UNTOUCHED:
-- organization_id is set once at signup and never modified in the happy path.
--
-- Confirmed against the live schema:
--   profiles.organization_id  -> NO FK to organizations (insert order is moot)
--   organizations NOT NULL, no default -> name, owner_profile_id (both set)
--   app_users uniques -> (id), (auth_user_id) only  => ON CONFLICT (auth_user_id)

BEGIN;

-- 1) ──────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.handle_new_user()
  RETURNS trigger
  LANGUAGE plpgsql
  SECURITY DEFINER
  SET search_path TO 'public','auth'
AS $function$
DECLARE
  v_meta_org   uuid;
  v_org_id     uuid;
  v_full_name  text;
  v_profile_id uuid;
  v_self_serve boolean;
BEGIN
  v_meta_org   := NULLIF(NEW.raw_user_meta_data->>'organization_id','')::uuid;
  v_full_name  := NEW.raw_user_meta_data->>'full_name';
  v_self_serve := v_meta_org IS NULL;          -- no invite => user owns a new org

  INSERT INTO public.profiles (user_id, organization_id, full_name)
  VALUES (NEW.id, COALESCE(v_meta_org, gen_random_uuid()), v_full_name)
  ON CONFLICT (user_id) DO UPDATE
    SET organization_id = COALESCE(public.profiles.organization_id, EXCLUDED.organization_id),
        full_name       = COALESCE(public.profiles.full_name, EXCLUDED.full_name)
  RETURNING id, organization_id INTO v_profile_id, v_org_id;

  -- Self-serve signup: materialise the org the profile now points at.
  -- (Invited users already have an org — skip.)
  IF v_self_serve THEN
    INSERT INTO public.organizations (id, owner_profile_id, name)
    VALUES (
      v_org_id,
      v_profile_id,
      COALESCE(NULLIF(v_full_name,''), split_part(NEW.email,'@',1)) || '''s Organization'
    )
    ON CONFLICT (id) DO NOTHING;
    -- trg_ensure_org_owner_app_user fires on this INSERT and creates the
    -- owner's app_users row (now idempotent — see change 2).
  END IF;

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
CREATE OR REPLACE FUNCTION public.ensure_org_owner_app_user(_org_id uuid)
  RETURNS void
  LANGUAGE plpgsql
  SECURITY DEFINER
  SET search_path TO 'public'
AS $function$
DECLARE
  _owner_profile uuid;
  _auth_id uuid;
  _email text;
  _full_perms jsonb := jsonb_build_object(
    'User Management',    jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Analysis',           jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Projects',           jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Dashboard',          jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true),
    'Data Configuration', jsonb_build_object('View', true, 'Create', true, 'Edit', true, 'Approve', true)
  );
  _existing uuid;
BEGIN
  SELECT o.owner_profile_id INTO _owner_profile
  FROM public.organizations o WHERE o.id = _org_id;

  IF _owner_profile IS NULL THEN
    RETURN;
  END IF;

  SELECT COALESCE(p.user_id, p.id) INTO _auth_id
  FROM public.profiles p WHERE p.id = _owner_profile;

  SELECT u.email INTO _email FROM auth.users u WHERE u.id = _auth_id;
  IF _email IS NULL THEN
    SELECT p.company_email INTO _email FROM public.profiles p WHERE p.id = _owner_profile;
  END IF;
  IF _email IS NULL THEN
    RETURN;
  END IF;

  SELECT au.id INTO _existing
  FROM public.app_users au
  WHERE au.organization_id = _org_id
    AND (au.profile_id = _owner_profile
         OR (_auth_id IS NOT NULL AND au.auth_user_id = _auth_id)
         OR lower(au.email) = lower(_email))
  LIMIT 1;

  IF _existing IS NOT NULL THEN
    UPDATE public.app_users
    SET role = 'Owner',
        permissions = _full_perms,
        status = 'Active',
        profile_id = COALESCE(profile_id, _owner_profile),
        auth_user_id = COALESCE(auth_user_id, _auth_id),
        accepted_at = COALESCE(accepted_at, now()),
        updated_at = now()
    WHERE id = _existing;
  ELSE
    INSERT INTO public.app_users
      (email, user_id_code, role, department, branch, status, permissions,
       organization_id, profile_id, auth_user_id, accepted_at)
    VALUES
      (_email,
       'OWN-' || upper(substr(replace(_org_id::text, '-', ''), 1, 8)),
       'Owner', 'General', 'HQ', 'Active', _full_perms,
       _org_id, _owner_profile, _auth_id, now())
    -- The org-scoped SELECT above can miss a row that exists under a
    -- different/stale org; the GLOBAL unique is on auth_user_id — guard it.
    ON CONFLICT (auth_user_id) DO UPDATE
      SET role            = 'Owner',
          permissions     = _full_perms,
          status          = 'Active',
          organization_id = _org_id,
          profile_id      = COALESCE(public.app_users.profile_id, _owner_profile),
          accepted_at     = COALESCE(public.app_users.accepted_at, now()),
          updated_at      = now();
  END IF;
END;
$function$;

-- 3) ──────────────────────────────────────────────────────────────────────
-- One-time backfill: create the missing organizations rows for users already
-- stuck (profile.organization_id points at an org that was never inserted).
-- The now-idempotent trigger creates each owner's app_users row.
INSERT INTO public.organizations (id, owner_profile_id, name)
SELECT p.organization_id,
       p.id,
       COALESCE(NULLIF(p.full_name,''), split_part(u.email,'@',1)) || '''s Organization'
FROM public.profiles p
JOIN auth.users u ON u.id = p.user_id
LEFT JOIN public.organizations o ON o.id = p.organization_id
WHERE p.organization_id IS NOT NULL
  AND o.id IS NULL
ON CONFLICT (id) DO NOTHING;

COMMIT;
