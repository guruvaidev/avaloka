-- New accounts have no profile at invitation creation time. Notify when the
-- pending invitation is later attached, as well as when it is first inserted.
DROP TRIGGER IF EXISTS notify_team_invitation_after_profile_attached ON public.team_invitations;
CREATE TRIGGER notify_team_invitation_after_profile_attached
  AFTER UPDATE OF invited_profile_id ON public.team_invitations
  FOR EACH ROW
  WHEN (OLD.invited_profile_id IS NULL AND NEW.invited_profile_id IS NOT NULL AND NEW.status = 'pending')
  EXECUTE FUNCTION public.notify_team_invitation();

-- Explicitly install the lifecycle triggers; acceptance and notifications must
-- not depend on which baseline schema a deployment was bootstrapped with.
CREATE OR REPLACE TRIGGER handle_team_invitation_status_change_after_update
  AFTER UPDATE ON public.team_invitations
  FOR EACH ROW EXECUTE FUNCTION public.handle_team_invitation_status_change();
CREATE OR REPLACE TRIGGER notify_team_invitation_after_insert
  AFTER INSERT ON public.team_invitations
  FOR EACH ROW EXECUTE FUNCTION public.notify_team_invitation();
CREATE OR REPLACE TRIGGER on_team_member_added
  AFTER INSERT ON public.team_memberships
  FOR EACH ROW EXECUTE FUNCTION public.handle_team_member_added();

-- Link invitations missed by earlier signups. Only a verified email may claim
-- an invitation. The normal login reconciliation completes acceptance.
UPDATE public.team_invitations i
   SET invited_profile_id = p.id
  FROM public.profiles p
  JOIN auth.users u ON u.id = coalesce(p.user_id, p.id)
 WHERE i.invited_profile_id IS NULL
   AND i.status = 'pending'
   AND lower(i.invited_email) = lower(u.email)
   AND u.email_confirmed_at IS NOT NULL;

-- Repair accepted invites on installations where the acceptance trigger was
-- missing. The member-added trigger also sends the recipient's notification.
INSERT INTO public.team_memberships (team_id, user_id, invited_by)
SELECT team_id, invited_profile_id, invited_by
  FROM public.team_invitations
 WHERE status = 'accepted' AND invited_profile_id IS NOT NULL
ON CONFLICT (team_id, user_id) DO NOTHING;

-- Repair missing invitation notifications without duplicating existing ones.
INSERT INTO public.notifications (
  recipient_id, type, title, body, actor_id, resource_type, resource_id, metadata
)
SELECT i.invited_profile_id, 'team_invitation', 'Team Invitation',
       coalesce(p.full_name, 'A team owner') || ' invited you to join team "' || t.name || '".',
       i.invited_by, 'team_invitation', i.id,
       jsonb_build_object('invitation_id', i.id, 'team_id', i.team_id, 'team_name', t.name)
  FROM public.team_invitations i
  JOIN public.teams t ON t.id = i.team_id
  LEFT JOIN public.profiles p ON p.id = i.invited_by
 WHERE i.status = 'pending' AND i.invited_profile_id IS NOT NULL
   AND NOT EXISTS (
     SELECT 1 FROM public.notifications n
      WHERE n.recipient_id = i.invited_profile_id
        AND n.type = 'team_invitation' AND n.resource_id = i.id
   );
