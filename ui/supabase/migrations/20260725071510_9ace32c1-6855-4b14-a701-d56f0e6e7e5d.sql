DO $$
DECLARE
  _org uuid;
BEGIN
  SELECT organization_id INTO _org
  FROM public.app_users
  WHERE organization_id IS NOT NULL
  GROUP BY organization_id
  HAVING count(*) > 0
  LIMIT 2;

  -- Only backfill if there is exactly one candidate org across linked app_users
  IF (SELECT count(DISTINCT organization_id) FROM public.app_users WHERE organization_id IS NOT NULL) = 1 THEN
    SELECT organization_id INTO _org FROM public.app_users WHERE organization_id IS NOT NULL LIMIT 1;
    UPDATE public.app_users SET organization_id = _org, updated_at = now() WHERE organization_id IS NULL;
    UPDATE public.organization_teams SET organization_id = _org, updated_at = now() WHERE organization_id IS NULL;
  END IF;
END $$;