-- Keep team invitation and membership notifications compatible with the
-- profile-based notifications table, and allow only Free/Professional invitees.

ALTER TABLE public.team_invitations ALTER COLUMN invited_profile_id DROP NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_team_invite_email_unique_pending
  ON public.team_invitations(team_id, lower(invited_email))
  WHERE status = 'pending';

CREATE OR REPLACE FUNCTION public.notify_team_invitation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_team_name text;
  v_inviter_name text;
BEGIN
  IF NEW.invited_profile_id IS NULL THEN
    RETURN NEW;
  END IF;
  SELECT name INTO v_team_name FROM public.teams WHERE id = NEW.team_id;
  SELECT full_name INTO v_inviter_name FROM public.profiles WHERE id = NEW.invited_by;

  INSERT INTO public.notifications (
    recipient_id, type, title, body, actor_id,
    resource_type, resource_id, metadata
  ) VALUES (
    NEW.invited_profile_id,
    'team_invitation',
    'Team Invitation',
    coalesce(v_inviter_name, 'A team owner') || ' invited you to join team "' || coalesce(v_team_name, 'Unnamed Team') || '".',
    NEW.invited_by,
    'team_invitation',
    NEW.id,
    jsonb_build_object(
      'invitation_id', NEW.id,
      'team_id', NEW.team_id,
      'team_name', v_team_name,
      'invited_by', NEW.invited_by,
      'inviter_name', v_inviter_name
    )
  );
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.handle_team_invitation_status_change()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_team_name text;
  v_inviter_name text;
  v_invitee_name text;
  v_exists boolean;
BEGIN
  IF OLD.status = 'pending' AND NEW.status IN ('accepted', 'declined') THEN
    SELECT name INTO v_team_name FROM public.teams WHERE id = NEW.team_id;
    SELECT full_name INTO v_inviter_name FROM public.profiles WHERE id = NEW.invited_by;
    SELECT full_name INTO v_invitee_name FROM public.profiles WHERE id = NEW.invited_profile_id;

    IF NEW.status = 'accepted' THEN
      SELECT EXISTS (
        SELECT 1 FROM public.team_memberships
         WHERE team_id = NEW.team_id AND user_id = NEW.invited_profile_id
      ) INTO v_exists;
      IF NOT v_exists THEN
        INSERT INTO public.team_memberships(team_id, user_id, invited_by)
        VALUES (NEW.team_id, NEW.invited_profile_id, NEW.invited_by);
      END IF;
    END IF;

    INSERT INTO public.notifications (
      recipient_id, type, title, body, actor_id,
      resource_type, resource_id, metadata
    ) VALUES (
      NEW.invited_by,
      CASE WHEN NEW.status = 'accepted' THEN 'team_invitation_accepted' ELSE 'team_invitation_declined' END,
      CASE WHEN NEW.status = 'accepted' THEN 'Invitation accepted' ELSE 'Invitation declined' END,
      coalesce(v_invitee_name, NEW.invited_email) ||
        CASE WHEN NEW.status = 'accepted' THEN ' accepted your invitation to team "' ELSE ' declined your invitation to team "' END ||
        coalesce(v_team_name, 'Unnamed Team') || '".',
      NEW.invited_profile_id,
      'team_invitation',
      NEW.id,
      jsonb_build_object('invitation_id', NEW.id, 'team_id', NEW.team_id, 'team_name', v_team_name)
    );
  END IF;
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.handle_team_member_added()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_team_name text;
  v_inviter_name text;
BEGIN
  IF NEW.invited_by IS NOT NULL AND NEW.invited_by <> NEW.user_id THEN
    SELECT name INTO v_team_name FROM public.teams WHERE id = NEW.team_id;
    SELECT full_name INTO v_inviter_name FROM public.profiles WHERE id = NEW.invited_by;
    INSERT INTO public.notifications (
      recipient_id, type, title, body, actor_id,
      resource_type, resource_id, metadata
    ) VALUES (
      NEW.user_id,
      'team_invitation_accepted',
      'Added to Team',
      coalesce(v_inviter_name, 'A team owner') || ' added you to team "' || coalesce(v_team_name, 'Unnamed Team') || '".',
      NEW.invited_by,
      'team',
      NEW.team_id,
      jsonb_build_object('team_id', NEW.team_id, 'team_name', v_team_name, 'membership_id', NEW.id)
    );
  END IF;
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION avaloka_meta.enforce_team_invitee_plan()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_auth_user_id uuid;
  v_tier text;
BEGIN
  IF TG_OP = 'UPDATE' AND NEW.status <> 'accepted' THEN
    RETURN NEW;
  END IF;
  IF TG_OP = 'INSERT' AND NEW.invited_profile_id IS NULL AND NEW.status = 'pending' THEN
    RETURN NEW;
  END IF;

  SELECT user_id INTO v_auth_user_id
    FROM public.profiles
   WHERE id = NEW.invited_profile_id;
  IF v_auth_user_id IS NULL THEN
    RAISE EXCEPTION 'The invited user must have an Avaloka account.' USING ERRCODE = '42501';
  END IF;

  v_tier := avaloka_meta.plan_tier_of(v_auth_user_id);
  IF v_tier NOT IN ('free', 'professional') THEN
    RAISE EXCEPTION 'Team invitations are available to Free and Professional plan users.' USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$$;
REVOKE ALL ON FUNCTION avaloka_meta.enforce_team_invitee_plan() FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS enforce_team_invitee_plan ON public.team_invitations;
CREATE TRIGGER enforce_team_invitee_plan
  BEFORE INSERT OR UPDATE OF invited_profile_id, status ON public.team_invitations
  FOR EACH ROW EXECUTE FUNCTION avaloka_meta.enforce_team_invitee_plan();

CREATE OR REPLACE FUNCTION public.attach_pending_team_invites_to_profile()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_email text;
BEGIN
  SELECT lower(email) INTO v_email
    FROM auth.users
   WHERE id = coalesce(NEW.user_id, NEW.id);
  IF v_email IS NOT NULL THEN
    UPDATE public.team_invitations
       SET invited_profile_id = NEW.id
     WHERE invited_profile_id IS NULL
       AND status = 'pending'
       AND lower(invited_email) = v_email;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS attach_pending_team_invites_to_profile ON public.profiles;
CREATE TRIGGER attach_pending_team_invites_to_profile
  AFTER INSERT OR UPDATE OF user_id ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.attach_pending_team_invites_to_profile();
