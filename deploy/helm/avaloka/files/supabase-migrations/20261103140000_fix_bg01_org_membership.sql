-- BG-01 (#646): current_org_id() honours profiles.organization_id
-- only when the caller owns that organisation or holds an Active roster row in
-- it; otherwise it falls back to an owned organisation, then to an Active roster
-- row linked to the caller.
--
-- Reconciled for develop-1.6 (bugfix/1.6-migration-fix-reconciled):
--   * renumbered from 20261101110000 (after D-2, 20261103120000); both copies;
--     inner BEGIN/COMMIT removed (see the BG-03 file);
--   * combined with D-2 as defence in depth. D-2 stops any signed-in caller,
--     admins included, from changing profiles.organization_id (C5) and dropped
--     the company_email match (C13). This function now also ignores a profile
--     organisation the caller is not a member of, so a profile moved by any
--     other path does not hand over that organisation either;
--   * 'Invited' no longer counts as membership, only 'Active': accepting an
--     invite (acceptAppUserInvite) flips the row to Active and links the account,
--     so an invitee has no organisation context before it accepts;
--   * search_path is empty (every name was already schema-qualified).

CREATE OR REPLACE FUNCTION public.current_org_id()
  RETURNS uuid LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path = ''
AS $function$
DECLARE _pid uuid; _org uuid;
BEGIN
  _pid := public.current_profile_id();
  IF _pid IS NULL THEN RETURN NULL; END IF;
  SELECT p.organization_id INTO _org FROM public.profiles p
  WHERE p.id = _pid AND p.organization_id IS NOT NULL
    AND ( EXISTS (SELECT 1 FROM public.organizations o
                  WHERE o.id = p.organization_id AND o.owner_profile_id = _pid)
       OR EXISTS (SELECT 1 FROM public.app_users au
                  WHERE au.organization_id = p.organization_id AND au.status = 'Active'
                    AND (au.profile_id = _pid OR au.auth_user_id = auth.uid())) )
  LIMIT 1;
  IF _org IS NOT NULL THEN RETURN _org; END IF;
  SELECT o.id INTO _org FROM public.organizations o
  WHERE o.owner_profile_id = _pid ORDER BY o.created_at LIMIT 1;
  IF _org IS NOT NULL THEN RETURN _org; END IF;
  SELECT au.organization_id INTO _org FROM public.app_users au
  WHERE au.status = 'Active' AND au.organization_id IS NOT NULL
    AND (au.profile_id = _pid OR au.auth_user_id = auth.uid())
  ORDER BY au.accepted_at DESC NULLS LAST LIMIT 1;
  RETURN _org;
END;
$function$;
