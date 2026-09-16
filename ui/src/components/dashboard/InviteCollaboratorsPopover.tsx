import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useServerFn } from "@tanstack/react-start";
import { Button as AriaButton } from "react-aria-components";
import { Copy01, UserPlus01, X, ChevronDown, HelpCircle, SearchLg, Users01 } from "@untitledui/icons";
import { Button } from "@/components/base/buttons/button";
import { Avatar } from "@/components/base/avatar/avatar";
import { Dropdown } from "@/components/base/dropdown/dropdown";
import { cx } from "@/lib/utils/cx";
import { usePlanFeatures } from "@/lib/use-plan-features";
import { listAppUsers } from "@/lib/configurations.functions";
import { supabase } from "@/integrations/supabase/client";
import {
  copyTextToClipboard,
  ensureResourceShareLink,
  filterActiveAppUsers,
  inviteAppUsersToResource,
  loadResourceCollaboratorRows,
  mergeCollaboratorMembers,
  removeResourceCollaborator,
  shareLinkDisplayUrl,
  shareLinkFullUrl,
  updateResourceCollaboratorAccess,
  updateResourceShareLinkPublic,
  type AppUserRow,
  type CollaboratorMember,
  type DepartmentMemberPreview,
  type ResourceType,
} from "@/lib/resource-collaborators";
import { toast } from "sonner";


type TeamRow = {
  id: string;
  name: string;
  member_emails: string[];
  user_count: number;
};

const ACCESS_OPTIONS = ["View", "Edit", "Comment", "Remove access"];
const SHARE_VISIBILITY_OPTIONS = ["Public", "Private"];
const MEMBER_ACCESS_OPTIONS = ["View", "Edit", "Comment"];
const DROPDOWN_POPOVER_CLASS = "invite-collaborators-dropdown";


const dropdownTriggerClass = (extra?: string) => cx("cursor-pointer outline-offset-2 outline-focus-ring", extra);

function AccessDropdown({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <Dropdown.Root>
      <AriaButton
        className={({ isPressed, isFocusVisible }) =>
          dropdownTriggerClass(
            cx(
              "flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-3 py-2.5 text-sm font-semibold text-secondary hover:bg-primary_hover",
              (isPressed || isFocusVisible) && "outline-2",
            ),
          )
        }
      >
        {value}
        <ChevronDown className="size-4 text-fg-quaternary" />
      </AriaButton>
      <Dropdown.Popover placement="bottom right" offset={6} className={cx("w-40", DROPDOWN_POPOVER_CLASS)}>
        <Dropdown.Menu>
          {options.map((o) => (
            <Dropdown.Item key={o} onAction={() => onChange(o)}>
              {o}
            </Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown.Root>
  );
}

function MemberAccessDropdown({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <Dropdown.Root>
      <AriaButton
        className={({ isPressed, isFocusVisible }) =>
          dropdownTriggerClass(
            cx(
              "flex items-center gap-1 rounded-md px-2 py-1 text-sm font-semibold text-secondary hover:bg-primary_hover",
              (isPressed || isFocusVisible) && "outline-2",
            ),
          )
        }
      >
        {value}
        <ChevronDown className="size-4 text-fg-quaternary" />
      </AriaButton>
      <Dropdown.Popover placement="bottom right" offset={6} className={cx("w-40", DROPDOWN_POPOVER_CLASS)}>
        <Dropdown.Menu>
          {ACCESS_OPTIONS.map((o) => (
            <Dropdown.Item key={o} onAction={() => onChange(o)}>
              {o}
            </Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown.Root>
  );
}

interface InviteCollaboratorsPopoverProps {
  open: boolean;
  onClose: () => void;
  className?: string;
  projectId?: string | null;
  analysisId?: string | null;
  reportId?: string | null;
  onCollaborationChange?: () => void;
}

export function InviteCollaboratorsPopover({
  open,
  onClose,
  className,
  projectId,
  analysisId,
  reportId,
  onCollaborationChange,
}: InviteCollaboratorsPopoverProps) {
  const listAppUsersFn = useServerFn(listAppUsers);
  const { canInviteCollaborators } = usePlanFeatures();

  const resourceType: ResourceType | null = reportId ? "report" : projectId ? "project" : analysisId ? "analysis" : null;
  const resourceId = reportId ?? projectId ?? analysisId ?? null;
  const isPersisted = Boolean(resourceType && resourceId);

  const [shareLink, setShareLink] = useState("app.example.com/shared/...");
  const [shareToken, setShareToken] = useState<string | null>(null);
  const [shareVisibility, setShareVisibility] = useState<"Public" | "Private">("Private");

  const [memberAccessLevel, setMemberAccessLevel] = useState("View");
  const [appUsers, setAppUsers] = useState<AppUserRow[]>([]);
  const [userSearch, setUserSearch] = useState("");
  const [selectedUser, setSelectedUser] = useState<DepartmentMemberPreview | null>(null);
  const [userPickerOpen, setUserPickerOpen] = useState(false);
  const [members, setMembers] = useState<CollaboratorMember[]>([]);
  const [memberAccess, setMemberAccess] = useState<Record<string, string>>({});
  const [sending, setSending] = useState(false);
  const [emailRecipientsText, setEmailRecipientsText] = useState("");
  const [emailMessage, setEmailMessage] = useState("");
  const [emailSending, setEmailSending] = useState(false);
  const [pdfPassword, setPdfPassword] = useState("");
  const [showPdfPassword, setShowPdfPassword] = useState(false);
  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const emailRecipients = emailRecipientsText
    .split(/[,;\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const emailsAllValid = emailRecipients.length > 0 && emailRecipients.every((e) => EMAIL_RE.test(e));
  const passwordValid = pdfPassword.length >= 6 && pdfPassword.length <= 128;
  const handleSendEmailResults = async () => {
    if (!analysisId) {
      toast.error("Run an analysis first to email results.");
      return;
    }
    if (!emailsAllValid || !passwordValid) return;
    setEmailSending(true);
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      const accessToken = sessionData.session?.access_token;
      if (!accessToken) throw new Error("Please sign in again before sending results.");

      const response = await fetch("/api/public/send-analysis-email", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({
          analysis_id: analysisId,
          recipients: emailRecipients,
          message: emailMessage.trim() || null,
          pdf_password: pdfPassword,
        }),
      });
      const data = (await response.json().catch(() => null)) as { ok?: boolean; error?: string } | null;
      if (!response.ok || !data?.ok) throw new Error(data?.error || "Failed to send email");
      if (data && typeof data === "object" && "error" in data && (data as { error?: string }).error) {
        throw new Error(String((data as { error?: string }).error));
      }
      toast.success(`Protected PDF sent to ${emailRecipients.join(", ")}. Share the password separately.`);
      setEmailRecipientsText("");
      setEmailMessage("");
      setPdfPassword("");
      setShowPdfPassword(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to send email");
    } finally {
      setEmailSending(false);
    }
  };
  const [configLoading, setConfigLoading] = useState(false);
  const userPickerRef = useRef<HTMLDivElement>(null);

  // Share via link (new REST-based flow) — only for analyses
  const [linkAccess, setLinkAccess] = useState<"View" | "Edit">("View");
  const [generatedLink, setGeneratedLink] = useState<{ url: string; token: string } | null>(null);
  const [linkBusy, setLinkBusy] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);

  const handleGenerateShareLink = async () => {
    if (!analysisId) return;
    setLinkBusy(true);
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      const accessToken = sessionData.session?.access_token;
      if (!accessToken) throw new Error("Please sign in again.");
      const res = await fetch(`/api/analysis/${analysisId}/share-link`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
        body: JSON.stringify({ access_level: linkAccess.toLowerCase() }),
      });
      const data = (await res.json().catch(() => null)) as { url?: string; token?: string; error?: string } | null;
      if (!res.ok || !data?.url || !data?.token) throw new Error(data?.error || "Failed to generate link");
      setGeneratedLink({ url: data.url, token: data.token });
      toast.success("Share link ready");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to generate link");
    } finally {
      setLinkBusy(false);
    }
  };

  const handleCopyGeneratedLink = async () => {
    if (!generatedLink) return;
    const ok = await copyTextToClipboard(generatedLink.url);
    if (ok) {
      setLinkCopied(true);
      window.setTimeout(() => setLinkCopied(false), 2000);
    } else {
      toast.error("Could not copy link");
    }
  };

  const handleRevokeShareLink = async () => {
    if (!analysisId || !generatedLink) return;
    setLinkBusy(true);
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      const accessToken = sessionData.session?.access_token;
      if (!accessToken) throw new Error("Please sign in again.");
      const res = await fetch(`/api/analysis/${analysisId}/share-link/${generatedLink.token}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      const data = (await res.json().catch(() => null)) as { ok?: boolean; error?: string } | null;
      if (!res.ok || !data?.ok) throw new Error(data?.error || "Failed to revoke link");
      setGeneratedLink(null);
      setLinkCopied(false);
      toast.success("Link revoked");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to revoke link");
    } finally {
      setLinkBusy(false);
    }
  };

  const [teams, setTeams] = useState<TeamRow[]>([]);
  const [teamSearch, setTeamSearch] = useState("");
  const [selectedTeam, setSelectedTeam] = useState<TeamRow | null>(null);
  const [teamPickerOpen, setTeamPickerOpen] = useState(false);
  const [teamAccessLevel, setTeamAccessLevel] = useState("View");
  const teamPickerRef = useRef<HTMLDivElement>(null);

  const existingMemberIds = useMemo(() => new Set(members.map((m) => m.appUserId)), [members]);

  const filteredUsers = useMemo(
    () => filterActiveAppUsers(appUsers, userSearch, existingMemberIds),
    [appUsers, userSearch, existingMemberIds],
  );

  const filteredTeams = useMemo(() => {
    const q = teamSearch.trim().toLowerCase();
    return teams
      .filter((t) => (q ? t.name.toLowerCase().includes(q) : true))
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [teams, teamSearch]);

  const loadAppUsers = useCallback(async () => {
    setConfigLoading(true);
    try {
      const [users, teamsRes, membersRes] = await Promise.all([
        listAppUsersFn(),
        supabase.from("organization_teams").select("id,name").eq("status", "Active"),
        supabase.from("app_users").select("id,email,team_id").not("team_id", "is", null),
      ]);
      setAppUsers((users as AppUserRow[]) ?? []);

      // Derive team member_emails from app_users.team_id.
      const emailsByTeam = new Map<string, string[]>();
      for (const u of (membersRes.data ?? []) as Array<{ team_id: string | null; email: string | null }>) {
        if (!u.team_id || !u.email) continue;
        const list = emailsByTeam.get(u.team_id) ?? [];
        list.push(u.email);
        emailsByTeam.set(u.team_id, list);
      }
      const teams: TeamRow[] = ((teamsRes.data ?? []) as Array<{ id: string; name: string }>).map((t) => {
        const emails = emailsByTeam.get(t.id) ?? [];
        return { id: t.id, name: t.name, member_emails: emails, user_count: emails.length };
      });
      setTeams(teams);
    } catch (err) {
      console.error("[InviteCollaboratorsPopover] users load failed", err);
      setAppUsers([]);
      setTeams([]);
    } finally {
      setConfigLoading(false);
    }
  }, [listAppUsersFn]);

  const reload = useCallback(async () => {
    if (!isPersisted || !resourceType || !resourceId) return;
    const [link, rows, users] = await Promise.all([
      ensureResourceShareLink(resourceType, resourceId),
      loadResourceCollaboratorRows(resourceType, resourceId),
      listAppUsersFn(),
    ]);
    const normalizedUsers = (users as AppUserRow[]) ?? [];
    setAppUsers(normalizedUsers);
    const cols = mergeCollaboratorMembers(rows, normalizedUsers);
    setShareToken(link.token);
    setShareLink(shareLinkDisplayUrl(link.token, resourceType));
    setShareVisibility(link.isPublic ? "Public" : "Private");

    setMembers(cols);
    setMemberAccess(Object.fromEntries(cols.map((m) => [m.id, m.access])));
  }, [isPersisted, resourceType, resourceId, listAppUsersFn]);

  const notifyCollaborationChange = useCallback(() => {
    onCollaborationChange?.();
  }, [onCollaborationChange]);

  useEffect(() => {
    if (!open) return;
    void loadAppUsers();
  }, [open, loadAppUsers]);

  useEffect(() => {
    if (!open) {
      setUserSearch("");
      setSelectedUser(null);
      setUserPickerOpen(false);
      setTeamSearch("");
      setSelectedTeam(null);
      setTeamPickerOpen(false);
      return;
    }
    if (!isPersisted) return;
    reload()
      .then(() => notifyCollaborationChange())
      .catch((err) => {
        console.error("[InviteCollaboratorsPopover] load failed", err);
        toast.error("Failed to load share settings");
      });
  }, [open, isPersisted, reload, notifyCollaborationChange]);

  useEffect(() => {
    if (!open || !userPickerOpen) return;
    const handler = (e: MouseEvent) => {
      if (userPickerRef.current?.contains(e.target as Node)) return;
      setUserPickerOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, userPickerOpen]);

  useEffect(() => {
    if (!open || !teamPickerOpen) return;
    const handler = (e: MouseEvent) => {
      if (teamPickerRef.current?.contains(e.target as Node)) return;
      setTeamPickerOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, teamPickerOpen]);

  const handleShareVisibilityChange = async (value: string) => {
    const next: "Public" | "Private" = value === "Public" ? "Public" : "Private";
    setShareVisibility(next);
    if (!isPersisted || !resourceType || !resourceId) return;
    try {
      await updateResourceShareLinkPublic(resourceType, resourceId, next === "Public");
      notifyCollaborationChange();
      toast.success(next === "Public" ? "Anyone with the link can view" : "Link now requires sign-in");
    } catch (err) {
      console.error("[InviteCollaboratorsPopover] share visibility update failed", err);
      toast.error("Failed to update link visibility");
    }
  };

  const handleCopyLink = async () => {
    const url = shareToken && resourceType ? shareLinkFullUrl(shareToken, resourceType) : shareLink;
    const copied = await copyTextToClipboard(url);
    if (copied) toast.success("Link copied");
    else toast.error("Could not copy link");
  };


  const handleMemberAccessChange = async (memberId: string, value: string) => {
    setMemberAccess((prev) => ({ ...prev, [memberId]: value }));
    if (!isPersisted) return;
    try {
      if (value === "Remove access") {
        await removeResourceCollaborator(memberId);
        setMembers((prev) => prev.filter((m) => m.id !== memberId));
      } else {
        await updateResourceCollaboratorAccess(memberId, value);
        setMembers((prev) => prev.map((m) => (m.id === memberId ? { ...m, access: value } : m)));
      }
      notifyCollaborationChange();
    } catch (err) {
      console.error("[InviteCollaboratorsPopover] member access update failed", err);
      toast.error("Failed to update access");
    }
  };

  const handleSelectUser = (user: DepartmentMemberPreview) => {
    setSelectedUser(user);
    setUserSearch(user.name);
    setUserPickerOpen(false);
  };

  const handleSelectTeam = (team: TeamRow) => {
    setSelectedTeam(team);
    setTeamSearch(team.name);
    setTeamPickerOpen(false);
  };

  const handleSendInvites = async () => {
    if (!isPersisted || !resourceType || !resourceId) {
      onClose();
      return;
    }
    if (!selectedUser && !selectedTeam) {
      toast.error("Select a team member or team first");
      return;
    }
    setSending(true);
    try {
      const successParts: string[] = [];
      if (selectedUser) {
        const count = await inviteAppUsersToResource({
          resourceType,
          resourceId,
          accessUi: memberAccessLevel,
          appUserIds: [selectedUser.id],
          appUsers,
        });
        successParts.push(count > 0 ? `Invited ${selectedUser.name}` : `${selectedUser.name} already has access`);
      }
      if (selectedTeam) {
        const emails = (selectedTeam.member_emails ?? [])
          .map((e) => e?.trim().toLowerCase())
          .filter((e): e is string => !!e);
        const emailSet = new Set(emails);
        const teamUserIds = appUsers
          .filter((u) => u.email && emailSet.has(u.email.toLowerCase()))
          .map((u) => u.id);
        if (teamUserIds.length === 0) {
          successParts.push(`Team "${selectedTeam.name}" has no matching users`);
        } else {
          const count = await inviteAppUsersToResource({
            resourceType,
            resourceId,
            accessUi: teamAccessLevel,
            appUserIds: teamUserIds,
            appUsers,
          });
          successParts.push(`Invited ${count} from "${selectedTeam.name}"`);
        }
      }
      await reload();
      setSelectedUser(null);
      setUserSearch("");
      setSelectedTeam(null);
      setTeamSearch("");
      notifyCollaborationChange();
      toast.success(successParts.join(" · "));
    } catch (err) {
      console.error("[InviteCollaboratorsPopover] send invites failed", err);
      toast.error("Failed to send invites");
    } finally {
      setSending(false);
    }
  };

  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className={cx(
        "invite-collaborators-dropdown fixed inset-x-3 bottom-3 top-16 z-50 flex min-h-0 flex-col overflow-hidden rounded-2xl border border-secondary bg-primary shadow-xl sm:left-auto sm:right-4 sm:w-[400px]",
        className,
      )}
    >
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-secondary px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-lg border border-secondary bg-primary shadow-xs">
            <UserPlus01 className="size-5 text-fg-quaternary" />
          </div>
          <h3 className="text-lg font-semibold text-primary">Invite collaborators</h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="cursor-pointer rounded-md p-1 text-fg-quaternary hover:text-fg-quaternary_hover"
        >
          <X className="size-5" />
        </button>
      </div>

      <div className="min-h-0 min-w-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="min-w-0 px-5 pt-4">
        <label className="text-sm font-medium text-secondary">Share link</label>
        <div className="mt-1.5 flex min-w-0 gap-2">
          <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden rounded-lg border border-secondary bg-primary px-3 py-2.5">
            <span className="min-w-0 flex-1 truncate text-sm text-tertiary">{shareLink}</span>
            <button
              type="button"
              aria-label="Copy"
              onClick={() => {
                void handleCopyLink();
              }}
              className="shrink-0 cursor-pointer text-fg-quaternary hover:text-fg-quaternary_hover"
            >
              <Copy01 className="size-4" />
            </button>
          </div>
          <div className="shrink-0">
            <AccessDropdown
              value={shareVisibility}
              onChange={(v) => {
                void handleShareVisibilityChange(v);
              }}
              options={SHARE_VISIBILITY_OPTIONS}
            />
          </div>

        </div>
      </div>

      {canInviteCollaborators ? (
        <>
      <div className="my-4 flex items-center gap-3 px-5">
        <div className="h-px flex-1 bg-border-secondary" />
        <span className="text-sm text-tertiary">Or</span>
        <div className="h-px flex-1 bg-border-secondary" />
      </div>

      <div className="px-5">
        <div className="flex items-center gap-1">
          <label className="text-sm font-medium text-secondary">Team member</label>
          <span className="text-sm text-error-primary">*</span>
          <HelpCircle className="size-4 text-fg-quaternary" />
        </div>
        <div className="mt-1.5 flex gap-2">
          <div ref={userPickerRef} className="relative min-w-0 flex-1">
            <div className="flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2.5 focus-within:ring-2 focus-within:ring-[#1565ef]/30">
              <SearchLg className="size-4 shrink-0 text-fg-quaternary" />
              <input
                type="text"
                value={userSearch}
                onChange={(e) => {
                  setUserSearch(e.target.value);
                  setSelectedUser(null);
                  setUserPickerOpen(true);
                }}
                onFocus={() => setUserPickerOpen(true)}
                placeholder="Search by name…"
                className="min-w-0 flex-1 bg-transparent text-sm text-primary outline-none placeholder:text-tertiary"
              />
            </div>
            {userPickerOpen ? (
              <div className="absolute inset-x-0 top-[calc(100%+4px)] z-10 max-h-[200px] overflow-y-auto rounded-lg border border-secondary bg-primary py-1 shadow-lg">
                {configLoading ? (
                  <p className="px-3 py-2 text-sm text-tertiary">Loading users…</p>
                ) : filteredUsers.length === 0 ? (
                  <p className="px-3 py-2 text-sm text-tertiary">
                    {userSearch.trim() ? "No users match your search" : "No users available"}
                  </p>
                ) : (
                  <ul>
                    {filteredUsers.map((user) => (
                      <li key={user.id}>
                        <button
                          type="button"
                          onClick={() => handleSelectUser(user)}
                          className={cx(
                            "flex w-full items-center gap-2.5 px-3 py-2 text-left hover:bg-primary_hover",
                            selectedUser?.id === user.id && "bg-primary_hover",
                          )}
                        >
                          <Avatar size="sm" initials={user.initials} alt={user.name} src={user.avatar ?? undefined} />
                          <div className="min-w-0 leading-tight">
                            <p className="truncate text-sm font-medium text-primary">{user.name}</p>
                            <p className="truncate text-xs text-tertiary">{user.role}</p>
                          </div>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ) : null}
          </div>
          <AccessDropdown value={memberAccessLevel} onChange={setMemberAccessLevel} options={MEMBER_ACCESS_OPTIONS} />
        </div>
        {selectedUser ? (
          <div className="mt-2 flex items-center gap-2">
            <Avatar
              size="sm"
              initials={selectedUser.initials}
              alt={selectedUser.name}
              src={selectedUser.avatar ?? undefined}
            />
            <p className="truncate text-sm text-secondary">
              Selected: <span className="font-medium text-primary">{selectedUser.name}</span>
            </p>
          </div>
        ) : null}

      </div>



      <div className="mt-4 px-5">
        <div className="flex items-center gap-1">
          <label className="text-sm font-medium text-secondary">Team</label>
          <HelpCircle className="size-4 text-fg-quaternary" />
        </div>
        <div className="mt-1.5 flex gap-2">
          <div ref={teamPickerRef} className="relative min-w-0 flex-1">
            <div className="flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2.5 focus-within:ring-2 focus-within:ring-[#1565ef]/30">
              <SearchLg className="size-4 shrink-0 text-fg-quaternary" />
              <input
                type="text"
                value={teamSearch}
                onChange={(e) => {
                  setTeamSearch(e.target.value);
                  setSelectedTeam(null);
                  setTeamPickerOpen(true);
                }}
                onFocus={() => setTeamPickerOpen(true)}
                placeholder="Search teams…"
                className="min-w-0 flex-1 bg-transparent text-sm text-primary outline-none placeholder:text-tertiary"
              />
            </div>
            {teamPickerOpen ? (
              <div className="absolute inset-x-0 top-[calc(100%+4px)] z-10 max-h-[200px] overflow-y-auto rounded-lg border border-secondary bg-primary py-1 shadow-lg">
                {configLoading ? (
                  <p className="px-3 py-2 text-sm text-tertiary">Loading teams…</p>
                ) : filteredTeams.length === 0 ? (
                  <p className="px-3 py-2 text-sm text-tertiary">
                    {teamSearch.trim() ? "No teams match your search" : "No teams available"}
                  </p>
                ) : (
                  <ul>
                    {filteredTeams.map((team) => {
                      const count = team.user_count ?? team.member_emails.length;
                      return (
                        <li key={team.id}>
                          <button
                            type="button"
                            onClick={() => handleSelectTeam(team)}
                            className={cx(
                              "flex w-full items-center gap-2.5 px-3 py-2 text-left hover:bg-primary_hover",
                              selectedTeam?.id === team.id && "bg-primary_hover",
                            )}
                          >
                            <span className="grid size-8 shrink-0 place-items-center rounded-full bg-[#1565EF]/10 text-[#1565EF]">
                              <Users01 className="size-4" />
                            </span>
                            <div className="min-w-0 leading-tight">
                              <p className="truncate text-sm font-medium text-primary">{team.name}</p>
                              <p className="truncate text-xs text-tertiary">{count} member{count === 1 ? "" : "s"}</p>
                            </div>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            ) : null}
          </div>
          <AccessDropdown value={teamAccessLevel} onChange={setTeamAccessLevel} options={MEMBER_ACCESS_OPTIONS} />
        </div>
        {selectedTeam ? (
          <div className="mt-3 rounded-lg border border-secondary bg-secondary/40 p-3">
            <div className="flex items-center gap-2">
              <span className="grid size-8 shrink-0 place-items-center rounded-full bg-[#1565EF]/10 text-[#1565EF]">
                <Users01 className="size-4" />
              </span>
              <div className="min-w-0 leading-tight">
                <p className="text-xs text-tertiary">Team name</p>
                <p className="truncate text-sm font-semibold text-primary">{selectedTeam.name}</p>
              </div>
            </div>
            {(() => {
              const emails = (selectedTeam.member_emails ?? [])
                .map((e) => e?.trim().toLowerCase())
                .filter((e): e is string => !!e);
              const emailSet = new Set(emails);
              const teamMembers = appUsers.filter((u) => u.email && emailSet.has(u.email.toLowerCase()));
              if (teamMembers.length === 0) {
                return (
                  <p className="mt-2 text-xs text-tertiary">No matching users found for this team.</p>
                );
              }
              return (
                <ul className="mt-3 flex max-h-[180px] flex-col gap-2 overflow-y-auto pr-1">
                  {teamMembers.map((u) => {
                    const name = u.name || u.email || "Unknown";
                    const initials = (name.trim().split(/\s+/).slice(0, 2).map((p: string) => p[0]?.toUpperCase() ?? "").join("")) || "U";
                    return (
                      <li key={u.id} className="flex items-center gap-2.5">
                        <Avatar size="sm" initials={initials} alt={name} src={u.avatar ?? undefined} />
                        <div className="min-w-0 leading-tight">
                          <p className="truncate text-sm font-medium text-primary">{name}</p>
                          {u.email ? <p className="truncate text-xs text-tertiary">{u.email}</p> : null}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              );
            })()}
          </div>
        ) : null}

      </div>

      <div className="mt-4 px-5">
        <p className="text-sm font-medium text-secondary">Members With Access</p>
        {!isPersisted ? (
          <p className="mt-3 text-sm text-tertiary">Save this item to a project to manage collaborators.</p>
        ) : members.length === 0 ? (
          <p className="mt-3 text-sm text-tertiary">No collaborators yet. Invite a team member above.</p>
        ) : (
          <ul className="mt-3 flex max-h-[260px] flex-col gap-4 overflow-y-auto pr-1">
            {members.map((m) => (
              <li key={m.id} className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <Avatar size="md" initials={m.initials} alt={m.name} />
                  <div className="leading-tight">
                    <p className="text-sm font-semibold text-primary">{m.name}</p>
                    <p className="text-sm text-tertiary">{m.role}</p>
                  </div>
                </div>
                <MemberAccessDropdown
                  value={memberAccess[m.id] ?? "View"}
                  onChange={(v) => {
                    void handleMemberAccessChange(m.id, v);
                  }}
                />
              </li>
            ))}
          </ul>
        )}
      </div>
        </>
      ) : null}




      {analysisId ? (
        <div className="mt-5 border-t border-secondary px-5 pt-4">
          <p className="text-sm font-semibold text-primary">Email Results</p>
          <p className="mt-1 text-xs text-tertiary">
            Recipients receive a password-protected PDF. Share the password separately.
          </p>
          <div className="mt-3 space-y-3">
            <div>
              <label htmlFor="ic-email-recipients" className="text-xs font-medium text-secondary">
                Recipient email(s)
              </label>
              <textarea
                id="ic-email-recipients"
                value={emailRecipientsText}
                onChange={(e) => setEmailRecipientsText(e.target.value)}
                rows={2}
                placeholder="name@example.com, another@example.com"
                disabled={emailSending}
                className="mt-1 w-full resize-none rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary shadow-xs placeholder:text-tertiary focus:border-[#1565ef] focus:outline-none"
              />
              {emailRecipientsText.trim() && !emailsAllValid ? (
                <p className="mt-1 text-xs text-error-primary">Enter valid email addresses separated by commas.</p>
              ) : null}
            </div>
            <div>
              <label htmlFor="ic-email-password" className="text-xs font-medium text-secondary">
                PDF password
              </label>
              <div className="mt-1 flex items-center gap-2">
                <input
                  id="ic-email-password"
                  type={showPdfPassword ? "text" : "password"}
                  value={pdfPassword}
                  onChange={(e) => setPdfPassword(e.target.value)}
                  placeholder="At least 6 characters"
                  disabled={emailSending}
                  autoComplete="new-password"
                  className="w-full rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary shadow-xs placeholder:text-tertiary focus:border-[#1565ef] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => setShowPdfPassword((v) => !v)}
                  className="shrink-0 text-xs font-medium text-[#1565ef] hover:underline"
                >
                  {showPdfPassword ? "Hide" : "Show"}
                </button>
              </div>
              {pdfPassword && !passwordValid ? (
                <p className="mt-1 text-xs text-error-primary">Password must be 6–128 characters.</p>
              ) : (
                <p className="mt-1 text-xs text-tertiary">Share this password with recipients separately.</p>
              )}
            </div>
            <div>
              <label htmlFor="ic-email-message" className="text-xs font-medium text-secondary">
                Message <span className="text-tertiary">(optional)</span>
              </label>
              <textarea
                id="ic-email-message"
                value={emailMessage}
                onChange={(e) => setEmailMessage(e.target.value)}
                rows={3}
                placeholder="Add a short note…"
                disabled={emailSending}
                className="mt-1 w-full resize-none rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary shadow-xs placeholder:text-tertiary focus:border-[#1565ef] focus:outline-none"
              />
            </div>
            <div className="flex justify-end">
              <Button
                color="primary"
                size="sm"
                isDisabled={!emailsAllValid || !passwordValid || emailSending}
                isLoading={emailSending}
                onClick={() => {
                  void handleSendEmailResults();
                }}
              >
                {emailSending ? "Sending…" : "Send email"}
              </Button>
            </div>
          </div>
        </div>
      ) : null}


      {analysisId ? (
        <div className="mt-5 border-t border-secondary px-5 pt-4">
          <p className="text-sm font-semibold text-primary">Generate Public Link</p>
          <p className="mt-1 text-xs text-tertiary">
            Anyone with this link can sign in and view these results.
          </p>
          <div className="mt-3 space-y-3">
            {!generatedLink ? (
              <div className="flex items-center gap-2">
                <AccessDropdown
                  value={linkAccess}
                  onChange={(v) => setLinkAccess(v === "Edit" ? "Edit" : "View")}
                  options={["View", "Edit"]}
                />
                <Button
                  color="primary"
                  size="sm"
                  isDisabled={linkBusy}
                  isLoading={linkBusy}
                  onClick={() => {
                    void handleGenerateShareLink();
                  }}
                >
                  {linkBusy ? "Generating…" : "Generate link"}
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2">
                  <input
                    readOnly
                    value={generatedLink.url}
                    className="min-w-0 flex-1 bg-transparent text-sm text-primary outline-none"
                    onFocus={(e) => e.currentTarget.select()}
                  />
                </div>
                <div className="flex items-center justify-end gap-2">
                  <Button
                    color="secondary"
                    size="sm"
                    isDisabled={linkBusy}
                    onClick={() => {
                      void handleCopyGeneratedLink();
                    }}
                  >
                    {linkCopied ? "Copied" : "Copy"}
                  </Button>
                  <Button
                    color="secondary-destructive"
                    size="sm"
                    isDisabled={linkBusy}
                    isLoading={linkBusy}
                    onClick={() => {
                      void handleRevokeShareLink();
                    }}
                  >
                    Revoke
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : null}




      </div>

      <div className="flex shrink-0 items-center justify-end gap-3 border-t border-secondary bg-primary px-5 py-4">
        <Button color="secondary" size="md" onClick={onClose}>
          Cancel
        </Button>
        {canInviteCollaborators ? (
          <Button
            color="primary"
            size="md"
            onClick={() => {
              void handleSendInvites();
            }}
            isDisabled={sending || !isPersisted || (!selectedUser && !selectedTeam)}
          >
            {sending ? "Sending…" : "Send invites"}
          </Button>
        ) : null}
      </div>
    </div>,
    document.body,
  );
}
