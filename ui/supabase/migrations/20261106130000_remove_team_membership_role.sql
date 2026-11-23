-- Team membership is a relationship only. Team owners and leads carry admin
-- authority on the team row, so memberships do not need a role column.

-- Preserve the existing team admin checks using team owner/lead profiles.
CREATE OR REPLACE FUNCTION public.is_team_admin(p_user_id uuid, p_team_id uuid)
RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.teams t
     WHERE t.id = p_team_id
       AND p_user_id IN (t.owner_profile_id, t.lead_profile_id)
  );
$$;

-- The public member listing no longer exposes a membership role.
DROP FUNCTION IF EXISTS public.get_team_member_public_info(uuid);
CREATE FUNCTION public.get_team_member_public_info(p_team_id uuid)
RETURNS TABLE(membership_id uuid, team_id uuid, profile_id uuid, auth_user_id uuid,
              full_name text, avatar_url text, joined_at timestamptz, email text)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT tm.id, tm.team_id, p.id, p.user_id, p.full_name, p.avatar_url,
         tm.joined_at, public.get_user_email(p.user_id)
    FROM public.team_memberships tm
    JOIN public.profiles p ON p.id = tm.user_id
   WHERE tm.team_id = p_team_id
     AND (public.is_team_member(public.current_profile_id(), p_team_id)
       OR public.is_team_admin(public.current_profile_id(), p_team_id)
       OR EXISTS (SELECT 1 FROM public.teams t WHERE t.id = p_team_id
                   AND t.owner_profile_id = public.current_profile_id()));
$$;
GRANT EXECUTE ON FUNCTION public.get_team_member_public_info(uuid)
  TO anon, authenticated, service_role;

-- Invitation acceptance creates a plain membership row.
CREATE OR REPLACE FUNCTION public.handle_team_invitation_status_change()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public
AS $$
DECLARE
  v_team_name text;
  v_invited_auth_user uuid;
  v_inviter_auth_user uuid;
  v_invited_full_name text;
  v_exists boolean;
BEGIN
  IF OLD.status = 'pending' AND NEW.status IN ('accepted', 'declined') THEN
    SELECT name INTO v_team_name FROM public.teams WHERE id = NEW.team_id;
    SELECT user_id, full_name INTO v_invited_auth_user, v_invited_full_name
      FROM public.profiles WHERE id = NEW.invited_profile_id;
    SELECT user_id INTO v_inviter_auth_user
      FROM public.profiles WHERE id = NEW.invited_by;

    IF NEW.status = 'accepted' THEN
      SELECT EXISTS (
        SELECT 1 FROM public.team_memberships
         WHERE team_id = NEW.team_id AND user_id = NEW.invited_profile_id
      ) INTO v_exists;

      IF NOT v_exists THEN
        INSERT INTO public.team_memberships(team_id, user_id, invited_by)
        VALUES (NEW.team_id, NEW.invited_profile_id, NEW.invited_by);
      END IF;

      INSERT INTO public.notifications (user_id, type, title, message, data, read)
      VALUES (v_inviter_auth_user, 'team_invitation_accepted', 'Invitation accepted',
        coalesce(v_invited_full_name, 'A user') || ' accepted your invitation to join "'
          || coalesce(v_team_name, 'Unnamed Team') || '"',
        jsonb_build_object('invitation_id', NEW.id, 'team_id', NEW.team_id,
                           'team_name', v_team_name), false);
    ELSE
      INSERT INTO public.notifications (user_id, type, title, message, data, read)
      VALUES (v_inviter_auth_user, 'team_invitation_declined', 'Invitation declined',
        coalesce(v_invited_full_name, 'A user') || ' declined your invitation to join "'
          || coalesce(v_team_name, 'Unnamed Team') || '"',
        jsonb_build_object('invitation_id', NEW.id, 'team_id', NEW.team_id,
                           'team_name', v_team_name), false);
    END IF;
  END IF;
  RETURN NEW;
END;
$$;

ALTER TABLE public.team_memberships DROP COLUMN IF EXISTS role;
