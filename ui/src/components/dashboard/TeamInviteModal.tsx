import { useEffect, useState, type FormEvent } from "react";
import { useServerFn } from "@tanstack/react-start";
import { Mail, Send, Users, X, Clock3, LoaderCircle, Crown, UserMinus } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Dialog, DialogOverlay, DialogPortal } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  assignProfessionalTeamLead,
  getProfessionalTeamDetails,
  inviteProfessionalTeamMember,
  removeProfessionalTeamMember,
} from "@/lib/configurations.functions";

type TeamRef = { id: string; name: string };

interface Props {
  team: TeamRef | null;
  canInvite: boolean;
  onOpenChange: (open: boolean) => void;
}

export function TeamInviteModal({ team, canInvite, onOpenChange }: Props) {
  const inviteFn = useServerFn(inviteProfessionalTeamMember);
  const detailsFn = useServerFn(getProfessionalTeamDetails);
  const assignLeadFn = useServerFn(assignProfessionalTeamLead);
  const removeMemberFn = useServerFn(removeProfessionalTeamMember);
  const queryClient = useQueryClient();
  const [email, setEmail] = useState("");
  const [saving, setSaving] = useState(false);
  const [resendingInviteId, setResendingInviteId] = useState<string | null>(null);
  const [resendError, setResendError] = useState<string | null>(null);
  const [resendMessage, setResendMessage] = useState<string | null>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);
  const [details, setDetails] = useState<Awaited<ReturnType<typeof detailsFn>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sentMessage, setSentMessage] = useState<string | null>(null);
  const [memberActionError, setMemberActionError] = useState<string | null>(null);
  const [actingOnMember, setActingOnMember] = useState<string | null>(null);

  const open = Boolean(team);

  const refreshDetails = async (teamId: string) => {
    setLoadingDetails(true);
    setDetailsError(null);
    try {
      setDetails(await detailsFn({ data: { teamId } }));
    } catch (cause) {
      setDetailsError(cause instanceof Error ? cause.message : "Could not load team members.");
    } finally {
      setLoadingDetails(false);
    }
  };

  const runMemberAction = async (profileId: string, action: "lead" | "remove") => {
    if (!team) return;
    setActingOnMember(profileId);
    setMemberActionError(null);
    try {
      if (action === "lead") {
        await assignLeadFn({ data: { teamId: team.id, profileId } });
      } else {
        await removeMemberFn({ data: { teamId: team.id, profileId } });
      }
      await queryClient.invalidateQueries({ queryKey: ["config", "teams"] });
      await refreshDetails(team.id);
    } catch (cause) {
      setMemberActionError(cause instanceof Error ? cause.message : "Could not update the team member.");
    } finally {
      setActingOnMember(null);
    }
  };

  useEffect(() => {
    setDetails(null);
    if (team) void refreshDetails(team.id);
  }, [team?.id]);

  const resendInvitation = async (invite: { id: string; email: string }) => {
    if (!team || resendingInviteId) return;
    setResendingInviteId(invite.id);
    setResendError(null);
    setResendMessage(null);
    try {
      await inviteFn({
        data: {
          teamId: team.id,
          email: invite.email,
          redirectTo: window.location.origin,
        },
      });
      setResendMessage(`Invitation resent to ${invite.email}.`);
      await refreshDetails(team.id);
      await queryClient.invalidateQueries({ queryKey: ["config", "teams"] });
    } catch (cause) {
      setResendError(cause instanceof Error ? cause.message : "Could not resend the invitation.");
    } finally {
      setResendingInviteId(null);
    }
  };

  const close = () => {
    setResendError(null);
    setResendMessage(null);
    setEmail("");
    setError(null);
    setSentMessage(null);
    setDetails(null);
    onOpenChange(false);
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!team) return;
    setSaving(true);
    setError(null);
    setSentMessage(null);
    try {
      const result = await inviteFn({
        data: {
          teamId: team.id,
          email: email.trim(),
          redirectTo: typeof window !== "undefined" ? window.location.origin : undefined,
        },
      });
      setEmail("");
      setSentMessage(result.signupRequired
        ? "Signup link sent. The user can accept the team invitation after creating their Avaloka account."
        : "Invitation sent. The user will join the team after accepting.");
      await refreshDetails(team.id);
      await queryClient.invalidateQueries({ queryKey: ["config", "teams"] });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not send the invitation.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !next && close()}>
      {open && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[520px] max-w-[94vw] max-h-[88vh] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-2xl border border-border bg-white shadow-xl dark:bg-[#0f1216] dark:border-white/10">
            <div className="flex items-center justify-between border-b border-border px-5 py-4 dark:border-white/10">
              <div className="flex items-center gap-3">
                <span className="flex size-9 items-center justify-center rounded-lg border border-border dark:border-white/10">
                  <Users className="size-4" />
                </span>
                <div>
                  <DialogPrimitive.Title className="text-base font-semibold text-foreground">Team members</DialogPrimitive.Title>
                  <DialogPrimitive.Description className="mt-0.5 text-xs text-muted-foreground">{team?.name}</DialogPrimitive.Description>
                </div>
              </div>
              <DialogPrimitive.Close onClick={close} className="text-muted-foreground hover:text-foreground" aria-label="Close">
                <X className="size-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="space-y-4 px-5 py-5">
              <section aria-label="Team members" className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-foreground">Members</h3>
                  {!loadingDetails && details && <span className="text-xs text-muted-foreground">{details.members.length}</span>}
                </div>
                {loadingDetails ? (
                  <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground"><LoaderCircle className="size-4 animate-spin" /> Loading members…</div>
                ) : detailsError ? (
                  <p role="alert" className="text-sm text-destructive">{detailsError}</p>
                ) : details?.members.length ? (
                  <ul className="max-h-44 divide-y divide-border overflow-y-auto rounded-lg border border-border px-3 dark:border-white/10">
                    {details.members.map((member) => (
                      <li key={member.id} className="flex items-center gap-3 py-2.5">
                        <span className="flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full bg-muted text-xs font-semibold text-muted-foreground">
                          {member.avatar ? <img src={member.avatar} alt="" className="size-full object-cover" /> : member.name.slice(0, 1).toUpperCase()}
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-medium text-foreground">{member.id === details.viewerProfileId ? "You" : member.name}</span>
                          {member.email && <span className="block truncate text-xs text-muted-foreground">{member.email}</span>}
                        </span>
                        {member.role && <span className="shrink-0 rounded-full bg-muted px-2 py-1 text-[10px] font-medium text-muted-foreground">{member.role}</span>}
                        {details.canInvite && (member.isMember || member.isOwner) && !member.isLead && (
                          <Button
                            type="button"
                            variant="outline"
                            className="h-8 gap-1 px-2 text-xs"
                            disabled={Boolean(actingOnMember)}
                            onClick={() => void runMemberAction(member.id, "lead")}
                            title="Assign as team lead"
                          >
                            {actingOnMember === member.id ? <LoaderCircle className="size-3 animate-spin" /> : <Crown className="size-3" />}
                            Make lead
                          </Button>
                        )}
                        {details.canInvite && member.isMember && !member.isOwner && (
                          <Button
                            type="button"
                            variant="outline"
                            className="h-8 gap-1 px-2 text-xs text-destructive hover:text-destructive"
                            disabled={Boolean(actingOnMember) || member.isLead}
                            onClick={() => void runMemberAction(member.id, "remove")}
                            title={member.isLead ? "Assign another lead before removing this member" : "Remove from team"}
                          >
                            <UserMinus className="size-3" />
                            Remove
                          </Button>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-sm text-muted-foreground">No members yet.</p>
                )}
                {memberActionError && <p role="alert" className="text-sm text-destructive">{memberActionError}</p>}
              </section>

              {details?.pendingInvitations.length ? (
                <section className="space-y-2">
                  <h3 className="text-sm font-semibold text-foreground">Pending invitations</h3>
                  <ul className="space-y-2">
                    {details.pendingInvitations.map((invite) => (
                      <li key={invite.id} className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock3 className="size-4 shrink-0" />
                        <span className="min-w-0 flex-1 truncate">{invite.email}</span>
                        {canInvite && details.canInvite && (
                          <Button
                            type="button"
                            variant="outline"
                            className="h-8 shrink-0 gap-1 px-2 text-xs"
                            aria-label={`Resend invitation to ${invite.email}`}
                            disabled={Boolean(resendingInviteId) || saving}
                            onClick={() => void resendInvitation(invite)}
                          >
                            {resendingInviteId === invite.id && <LoaderCircle className="size-3 animate-spin" />}
                            {resendingInviteId === invite.id ? "Sending…" : "Resend"}
                          </Button>
                        )}
                      </li>
                    ))}
                  </ul>
                  {resendError && <p role="alert" className="text-sm text-destructive">{resendError}</p>}
                  {resendMessage && <p role="status" className="text-sm text-emerald-600">{resendMessage}</p>}
                </section>
              ) : null}

              {canInvite && details?.canInvite && <form onSubmit={submit}>
              <div className="space-y-4 border-t border-border pt-4 dark:border-white/10">
                <h3 className="text-sm font-semibold text-foreground">Invite a user</h3>
                <div className="space-y-1.5">
                  <label htmlFor="team-invite-email" className="text-sm font-medium text-foreground">User email</label>
                  <div className="relative">
                    <Mail className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                    <Input
                      id="team-invite-email"
                      type="email"
                      required
                      autoComplete="email"
                      placeholder="name@example.com"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      className="h-10 pl-9"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">Users on any plan can join. New users will receive an Avaloka signup link.</p>
                </div>
                {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
                {sentMessage && <p role="status" className="text-sm text-emerald-600">{sentMessage}</p>}
              </div>
              <div className="flex justify-end gap-2 pt-4">
                <Button type="button" variant="outline" onClick={close} disabled={saving}>Close</Button>
                <Button type="submit" disabled={saving || !email.trim()} className="bg-[#1565EF] text-white hover:bg-[#1257cf]">
                  <Send className="size-4" />
                  {saving ? "Sending…" : "Send invite"}
                </Button>
              </div>
            </form>}
              {canInvite && details && !details.canInvite && <p className="border-t border-border pt-4 text-xs text-muted-foreground dark:border-white/10">Only the team owner or lead can invite users.</p>}
            </div>
            {!canInvite && <div className="flex justify-end border-t border-border px-5 py-4 dark:border-white/10"><Button type="button" variant="outline" onClick={close}>Close</Button></div>}
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
