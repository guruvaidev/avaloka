import { useEffect, useMemo, useState } from "react";
import { ChevronDown, CheckCircle2, X, HelpCircle, XCircle, Check } from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { TeamsIcon } from "@/components/dashboard/icons/ConfigSectionIcons";
import { cn } from "@/lib/utils";
import { listAppUsers, listBranches, upsertTeam } from "@/lib/configurations.functions";

type Action = "update" | "deactivate" | "activate";
type Step = "form" | "confirm" | "success";

type UserOpt = { id: string; name: string; role: string; avatar: string; email: string };

export interface EditTeam {
  id: string;
  name: string;
  description?: string | null;
  lead: { id: string | null; name: string; email: string; avatar: string };
  branch: string;
  status: "Active" | "Inactive";
  member_ids?: string[];
}

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  team: EditTeam | null;
}

export function EditTeamModal({ open, onOpenChange, team }: Props) {
  const qc = useQueryClient();
  const listUsersFn = useServerFn(listAppUsers);
  const listBranchesFn = useServerFn(listBranches);
  const upsertTeamFn = useServerFn(upsertTeam);

  const usersQ = useQuery<UserOpt[]>({
    queryKey: ["config", "users", "active"],
    queryFn: async () => {
      const rows = (await listUsersFn()) as Array<{
        id: string; name: string; email: string; role: string; avatar: string | null; status?: string | null;
      }>;
      return rows
        .filter((u) => (u.status ?? "Active") === "Active")
        .map((u) => ({ id: u.id, name: u.name, role: u.role, email: u.email, avatar: u.avatar ?? "" }));
    },
    enabled: open,
  });

  const branchesQ = useQuery<string[]>({
    queryKey: ["config", "branches", "names"],
    queryFn: async () => {
      const rows = (await listBranchesFn()) as Array<{ name: string; location: string }>;
      return rows.map((b) => `${b.name}${b.location ? `, ${b.location}` : ""}`);
    },
    enabled: open,
  });

  const USERS = usersQ.data ?? [];
  const BRANCHES_OPT = branchesQ.data ?? [];

  const [step, setStep] = useState<Step>("form");
  const [action, setAction] = useState<Action>("update");
  const [submitting, setSubmitting] = useState(false);

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [lead, setLead] = useState<UserOpt | null>(null);
  const [branch, setBranch] = useState("");
  const [members, setMembers] = useState<UserOpt[]>([]);

  const [leadOpen, setLeadOpen] = useState(false);
  const [branchOpen, setBranchOpen] = useState(false);
  const [membersOpen, setMembersOpen] = useState(false);

  const inactive = team?.status === "Inactive";

  const [initial, setInitial] = useState({
    name: "",
    description: "",
    leadName: "",
    branch: "",
    memberNames: "",
  });

  useEffect(() => {
    if (open && team) {
      setStep("form");
      const seedLead: UserOpt = {
        id: team.lead.id ?? "",
        name: team.lead.name,
        email: team.lead.email,
        role: "Team Lead",
        avatar: team.lead.avatar,
      };
      setName(team.name);
      setDescription(team.description ?? "");
      setLead(team.lead.name ? seedLead : null);
      setBranch(team.branch);
      const ids = team.member_ids ?? [];
      const seedMembers: UserOpt[] = ids.map((id) => ({
        id,
        name: id,
        email: "",
        role: "",
        avatar: "",
      }));
      setMembers(seedMembers);
      setInitial({
        name: team.name,
        description: team.description ?? "",
        leadName: team.lead.name,
        branch: team.branch,
        memberNames: ids.slice().sort().join(","),
      });
    }
  }, [open, team]);

  // Once USERS load, hydrate members with full info (name/role/avatar)
  useEffect(() => {
    if (!open || !team || USERS.length === 0) return;
    const ids = team.member_ids ?? [];
    if (ids.length === 0) return;
    const hydrated: UserOpt[] = ids
      .map((id) => USERS.find((u) => u.id === id))
      .filter((u): u is UserOpt => Boolean(u));
    if (hydrated.length > 0) {
      setMembers(hydrated);
      setInitial((prev) => ({ ...prev, memberNames: hydrated.map((m) => m.id).slice().sort().join(",") }));
    }
    // Also hydrate lead if we only had partial info
    if (team.lead.id) {
      const fullLead = USERS.find((u) => u.id === team.lead.id);
      if (fullLead) setLead(fullLead);
    }
  }, [open, team, USERS]);

  const toggleMember = (m: UserOpt) => {
    if (inactive) return;
    setMembers((prev) =>
      prev.find((x) => x.id === m.id) ? prev.filter((x) => x.id !== m.id) : [...prev, m],
    );
  };

  const removeMember = (id: string) => {
    if (inactive) return;
    setMembers((prev) => prev.filter((m) => m.id !== id));
  };

  const hasChanges = useMemo(
    () =>
      name !== initial.name ||
      description !== initial.description ||
      (lead?.name ?? "") !== initial.leadName ||
      branch !== initial.branch ||
      members.map((m) => m.id).slice().sort().join(",") !== initial.memberNames,
    [name, description, lead, branch, members, initial],
  );

  const canUpdate = Boolean(name && lead && branch && hasChanges);

  const startConfirm = (a: Action) => {
    setAction(a);
    setStep("confirm");
  };

  const handleConfirm = async () => {
    if (!team) return;
    setSubmitting(true);
    try {
      if (action === "deactivate" || action === "activate") {
        await upsertTeamFn({
          data: {
            id: team.id,
            name: team.name,
            lead_user_id: team.lead.id ?? null,
            branch: team.branch,
            status: action === "activate" ? "Active" : "Inactive",
          },
        });
      } else {
        await upsertTeamFn({
          data: {
            id: team.id,
            name: name.trim(),
            description: description.trim() || null,
            lead_user_id: lead?.id ?? null,
            member_ids: members.map((m) => m.id),
            branch,
            status: team.status,
          },
        });
      }
      await qc.invalidateQueries({ queryKey: ["config", "teams"] });
      setStep("success");
      setTimeout(() => onOpenChange(false), 1500);
    } catch (e) {
      console.error(e);
      setStep("form");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {step === "form" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 max-h-[92vh] w-[760px] max-w-[94vw] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-2xl bg-white dark:bg-[#0f1216] dark:border dark:border-white/10 shadow-xl duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
            <div className="flex items-center justify-between border-b border-border dark:border-white/10 px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border dark:border-white/10 bg-white dark:bg-white/5">
                  <TeamsIcon className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">Team</h2>
                {inactive && (
                  <span className="inline-flex items-center rounded-full border border-border bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                    Inactive
                  </span>
                )}
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="max-h-[68vh] space-y-4 overflow-y-auto px-5 py-5">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Team Name</label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={inactive}
                  className="h-10 disabled:opacity-70"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Description <span className="font-normal text-muted-foreground">(Optional)</span>
                </label>
                <Textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  disabled={inactive}
                  className="min-h-[96px] resize-none disabled:opacity-70"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Team Lead</label>
                  <Popover open={leadOpen} onOpenChange={setLeadOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        disabled={inactive}
                        className="flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm text-foreground disabled:opacity-70"
                      >
                        {lead ? (
                          <span className="flex items-center gap-2">
                            <Avatar className="h-6 w-6">
                              <AvatarImage src={lead.avatar} alt={lead.name} />
                              <AvatarFallback>{lead.name[0]}</AvatarFallback>
                            </Avatar>
                            {lead.name}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">Select team member</span>
                        )}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="max-h-72 w-[var(--radix-popover-trigger-width)] overflow-y-auto p-1" align="start">
                      {USERS.length === 0 ? (
                        <div className="px-2.5 py-2 text-sm text-muted-foreground">No users available</div>
                      ) : (
                        USERS.map((l) => (
                          <button
                            key={l.email || l.name}
                            onClick={() => {
                              setLead(l);
                              setLeadOpen(false);
                            }}
                            className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                          >
                            <Avatar className="h-6 w-6">
                              <AvatarImage src={l.avatar} alt={l.name} />
                              <AvatarFallback>{l.name[0]}</AvatarFallback>
                            </Avatar>
                            <span className="flex flex-col items-start">
                              <span className="text-sm text-foreground">{l.name}</span>
                              <span className="text-xs text-muted-foreground">{l.role}</span>
                            </span>
                          </button>
                        ))
                      )}
                    </PopoverContent>
                  </Popover>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Branch</label>
                  <Popover open={branchOpen} onOpenChange={setBranchOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        disabled={inactive}
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm disabled:opacity-70",
                          branch ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {branch || "Select Branch"}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="max-h-72 w-[var(--radix-popover-trigger-width)] overflow-y-auto p-1" align="start">
                      {BRANCHES_OPT.length === 0 ? (
                        <div className="px-2.5 py-2 text-sm text-muted-foreground">No branches available</div>
                      ) : (
                        BRANCHES_OPT.map((b) => (
                          <button
                            key={b}
                            onClick={() => {
                              setBranch(b);
                              setBranchOpen(false);
                            }}
                            className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                          >
                            {b}
                            {branch === b && <Check className="h-4 w-4 text-primary" />}
                          </button>
                        ))
                      )}
                    </PopoverContent>
                  </Popover>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Team member</label>
                <Popover open={membersOpen} onOpenChange={setMembersOpen}>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      disabled={inactive}
                      className="flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm text-foreground disabled:opacity-70"
                    >
                      {members[0] ? (
                        <span className="flex items-center gap-2">
                          <Avatar className="h-6 w-6">
                            <AvatarImage src={members[0].avatar} alt={members[0].name} />
                            <AvatarFallback>{members[0].name[0]}</AvatarFallback>
                          </Avatar>
                          {members[0].name}
                          {members.length > 1 && (
                            <span className="text-xs text-muted-foreground">+{members.length - 1}</span>
                          )}
                        </span>
                      ) : (
                        <span className="text-muted-foreground">Select Team Members</span>
                      )}
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="max-h-72 w-[var(--radix-popover-trigger-width)] overflow-y-auto p-1" align="start">
                    {USERS.length === 0 ? (
                      <div className="px-2.5 py-2 text-sm text-muted-foreground">No users available</div>
                    ) : (
                      USERS.map((m) => {
                        const selected = !!members.find((x) => x.name === m.name);
                        return (
                          <button
                            key={m.email || m.name}
                            onClick={() => toggleMember(m)}
                            className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                          >
                            <Avatar className="h-6 w-6">
                              <AvatarImage src={m.avatar} alt={m.name} />
                              <AvatarFallback>{m.name[0]}</AvatarFallback>
                            </Avatar>
                            <span className="flex flex-1 flex-col items-start">
                              <span className="text-sm text-foreground">{m.name}</span>
                              <span className="text-xs text-muted-foreground">{m.role}</span>
                            </span>
                            {selected && <Check className="h-4 w-4 text-primary" />}
                          </button>
                        );
                      })
                    )}
                  </PopoverContent>
                </Popover>
              </div>

              {members.length > 1 && (
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-foreground">Members With Access</h4>
                  <ul className="space-y-3">
                    {members.map((m) => (
                      <li key={m.name} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Avatar className="h-9 w-9">
                            <AvatarImage src={m.avatar} alt={m.name} />
                            <AvatarFallback>{m.name[0]}</AvatarFallback>
                          </Avatar>
                          <div className="flex flex-col">
                            <span className="text-sm font-semibold text-foreground">{m.name}</span>
                            <span className="text-xs text-muted-foreground">{m.role}</span>
                          </div>
                        </div>
                        {!inactive && (
                          <button
                            onClick={() => removeMember(m.id)}
                            className="text-rose-500 hover:text-rose-600"
                            aria-label={`Remove ${m.name}`}
                          >
                            <X className="h-4 w-4" />
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="flex justify-end gap-3 border-t border-border dark:border-white/10 px-5 py-4">
              {inactive ? (
                <Button
                  onClick={() => startConfirm("activate")}
                  className="h-11 gap-2 bg-[#1565EF] px-5 font-semibold text-white hover:bg-[#1257cf]"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  Activate
                </Button>
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={() => startConfirm("deactivate")}
                    className="h-11 gap-2 font-semibold"
                  >
                    <XCircle className="h-4 w-4" />
                    Deactivate
                  </Button>
                  <Button
                    onClick={() => startConfirm("update")}
                    disabled={!canUpdate}
                    className="h-11 gap-2 bg-[#1565EF] px-5 font-semibold text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] dark:disabled:bg-white/5 disabled:text-muted-foreground"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    Update
                  </Button>
                </>
              )}
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}

      {step === "confirm" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[420px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white dark:bg-[#0f1216] dark:border dark:border-white/10 p-6 shadow-xl duration-200 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95">
            <DialogPrimitive.Close className="absolute right-4 top-4 text-muted-foreground hover:text-foreground">
              <X className="h-5 w-5" />
            </DialogPrimitive.Close>
            <div className="flex flex-col items-center text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50 dark:bg-blue-500/15">
                <HelpCircle className="h-6 w-6 text-[#1565EF]" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-foreground">
                Are you sure you want to {action} team
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{name}?</p>
              <div className="mt-6 grid w-full grid-cols-2 gap-3">
                <Button
                  variant="outline"
                  onClick={() => setStep("form")}
                  disabled={submitting}
                  className="h-11 gap-2 font-semibold"
                >
                  <XCircle className="h-4 w-4" />
                  Cancel
                </Button>
                <Button
                  onClick={handleConfirm}
                  disabled={submitting}
                  className="h-11 gap-2 bg-[#1565EF] font-semibold text-white hover:bg-[#1257cf]"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  {submitting ? "Saving..." : "Confirm"}
                </Button>
              </div>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}

      {step === "success" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[400px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white dark:bg-[#0f1216] dark:border dark:border-white/10 p-8 shadow-xl duration-200 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95">
            <div className="flex flex-col items-center text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-500/15">
                <CheckCircle2 className="h-7 w-7 text-emerald-700 dark:text-emerald-400" strokeWidth={2} />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-foreground">
                Successfully{" "}
                {action === "update" ? "updated" : action === "activate" ? "activated" : "deactivated"} team
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{name}</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
