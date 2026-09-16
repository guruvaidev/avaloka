ALTER TABLE public.profiles
  ALTER COLUMN organization_id DROP DEFAULT,
  ALTER COLUMN organization_id DROP NOT NULL;

CREATE OR REPLACE FUNCTION public.prevent_profile_org_change()
RETURNS trigger
LANGUAGE plpgsql
SET search_path TO 'public'
AS $function$
BEGIN
  IF NEW.organization_id IS DISTINCT FROM OLD.organization_id THEN
    IF COALESCE(current_setting('request.jwt.claim.role', true), '') = 'service_role' THEN
      RETURN NEW;
    END IF;

    IF NOT public.has_role(auth.uid(), 'admin') THEN
      RAISE EXCEPTION 'organization_id cannot be modified';
    END IF;
  END IF;

  RETURN NEW;
END;
$function$;