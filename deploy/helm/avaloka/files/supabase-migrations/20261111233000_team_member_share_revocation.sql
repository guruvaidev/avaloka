-- Direct Professional sharing is available only between current teammates.
-- Team-wide grants already follow membership dynamically. Revoke direct grants
-- when that relationship ends; resources and Enterprise app-user grants remain.
CREATE OR REPLACE FUNCTION public.profiles_share_professional_team(p_left uuid, p_right uuid)
RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = ''
AS $$
  SELECT p_left IS NOT NULL AND p_right IS NOT NULL AND EXISTS (
    SELECT 1 FROM public.teams t
    WHERE (
      p_left IN (t.owner_profile_id, t.lead_profile_id)
      OR EXISTS (SELECT 1 FROM public.team_memberships m WHERE m.team_id=t.id AND m.user_id=p_left)
    ) AND (
      p_right IN (t.owner_profile_id, t.lead_profile_id)
      OR EXISTS (SELECT 1 FROM public.team_memberships m WHERE m.team_id=t.id AND m.user_id=p_right)
    )
  )
$$;
REVOKE ALL ON FUNCTION public.profiles_share_professional_team(uuid,uuid) FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.revoke_departed_member_direct_shares()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  IF TG_OP = 'UPDATE' AND NEW.team_id = OLD.team_id AND NEW.user_id = OLD.user_id THEN
    RETURN NEW;
  END IF;
  DELETE FROM public.analysis_shares s
    WHERE s.recipient_profile_id=OLD.user_id AND s.shared_by IS NOT NULL
      AND NOT public.profiles_share_professional_team(s.shared_by,s.recipient_profile_id);
  DELETE FROM public.project_shares s
    WHERE s.recipient_profile_id=OLD.user_id AND s.shared_by IS NOT NULL
      AND NOT public.profiles_share_professional_team(s.shared_by,s.recipient_profile_id);
  IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
  RETURN NEW;
END;
$$;
REVOKE ALL ON FUNCTION public.revoke_departed_member_direct_shares() FROM PUBLIC, anon, authenticated;
DROP TRIGGER IF EXISTS revoke_departed_member_direct_shares ON public.team_memberships;
CREATE TRIGGER revoke_departed_member_direct_shares
  AFTER DELETE OR UPDATE OF team_id,user_id ON public.team_memberships
  FOR EACH ROW EXECUTE FUNCTION public.revoke_departed_member_direct_shares();

-- Prevent a grant created after a membership check from restoring revoked access.
CREATE OR REPLACE FUNCTION public.require_direct_share_teammates()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  IF NEW.recipient_profile_id IS NOT NULL AND NEW.shared_by IS NOT NULL
     AND NOT public.profiles_share_professional_team(NEW.shared_by,NEW.recipient_profile_id) THEN
    RAISE EXCEPTION 'Direct sharing requires a current shared team.' USING ERRCODE='42501';
  END IF;
  RETURN NEW;
END;
$$;
REVOKE ALL ON FUNCTION public.require_direct_share_teammates() FROM PUBLIC, anon, authenticated;
DROP TRIGGER IF EXISTS require_direct_share_teammates ON public.analysis_shares;
CREATE TRIGGER require_direct_share_teammates
  BEFORE INSERT OR UPDATE OF recipient_profile_id,shared_by ON public.analysis_shares
  FOR EACH ROW EXECUTE FUNCTION public.require_direct_share_teammates();
DROP TRIGGER IF EXISTS require_direct_share_teammates ON public.project_shares;
CREATE TRIGGER require_direct_share_teammates
  BEFORE INSERT OR UPDATE OF recipient_profile_id,shared_by ON public.project_shares
  FOR EACH ROW EXECUTE FUNCTION public.require_direct_share_teammates();

-- Repair direct grants retained after earlier membership removals.
DELETE FROM public.analysis_shares s
  WHERE s.recipient_profile_id IS NOT NULL AND s.shared_by IS NOT NULL
    AND NOT public.profiles_share_professional_team(s.shared_by,s.recipient_profile_id);
DELETE FROM public.project_shares s
  WHERE s.recipient_profile_id IS NOT NULL AND s.shared_by IS NOT NULL
    AND NOT public.profiles_share_professional_team(s.shared_by,s.recipient_profile_id);
NOTIFY pgrst, 'reload schema';
