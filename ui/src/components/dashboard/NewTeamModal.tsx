import { useMemo, useState } from "react";
import { ChevronDown, CheckCircle2, X, HelpCircle, XCircle, Plus, Check, User as UserIconLucide } from "lucide-react";
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

type Step = "form" | "confirm" | "success";

type UserOpt = { id: string; name: string; role: string; avatar: string; email: string };

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function NewTeamModal({ open, onOpenChange }: Props) {
  const qc = useQueryClient();
  const listUsersFn = useServerFn(listAppUsers);
  const listBranchesFn = useServerFn(listBranches);
  const upsertTeamFn = useServerFn(upsertTeam);

  const usersQ = useQuery<UserOpt[]>({
    queryKey: ["config", "users", "active"],
    queryFn: async () => {
      const rows = (await listUsersFn()) as Array<{
        id: string;
        name: string;
        email: string;
        role: string;
        avatar: string | null;
        team_id?: string | null;
        status?: string | null;
      }>;
      return rows
        .filter((u) => (u.status ?? "Active") === "Active")
        .map((u) => ({
        id: u.id,
        name: u.name,
        role: u.role,
        email: u.email,
        avatar: u.avatar ?? "",
        team_id: u.team_id ?? null,
      }));
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
  const BRANCHES = branchesQ.data ?? [];
  const AVAILABLE_MEMBERS = USERS.filter((u) => !(u as any).team_id);


  const [step, setStep] = useState<Step>("form");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [lead, setLead] = useState<UserOpt | null>(null);
  const [branch, setBranch] = useState("");
  const [members, setMembers] = useState<UserOpt[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const [leadOpen, setLeadOpen] = useState(false);
  const [branchOpen, setBranchOpen] = useState(false);
  const [membersOpen, setMembersOpen] = useState(false);

  const reset = () => {
    setStep("form");
    setName("");
    setDescription("");
    setLead(null);
    setBranch("");
    setMembers([]);
  };

  const handleOpenChange = (o: boolean) => {
    if (!o) reset();
    onOpenChange(o);
  };

  const canSubmit = useMemo(
    () => Boolean(name.trim() && lead && branch),
    [name, lead, branch],
  );

  const toggleMember = (m: UserOpt) => {
    setMembers((prev) =>
      prev.find((x) => x.name === m.name)
        ? prev.filter((x) => x.name !== m.name)
        : [...prev, m],
    );
  };

  const removeMember = (n: string) =>
    setMembers((prev) => prev.filter((m) => m.name !== n));

  const handleConfirm = async () => {
    if (!lead) return;
    setSubmitting(true);
    try {
      await upsertTeamFn({
        data: {
          name: name.trim(),
          description: description.trim() || null,
          lead_user_id: lead.id,
          member_ids: members.map((m) => m.id),
          branch,
          status: "Active",
        },
      });
      await qc.invalidateQueries({ queryKey: ["config", "teams"] });
      setStep("success");
      setTimeout(() => handleOpenChange(false), 1500);
    } catch (e) {
      console.error(e);
      setStep("form");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
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
                  placeholder="Enter Team Name"
                  className="h-10"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Description <span className="font-normal text-muted-foreground">(Optional)</span>
                </label>
                <Textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Enter Description"
                  className="min-h-[96px] resize-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Team Lead</label>
                  <Popover open={leadOpen} onOpenChange={setLeadOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className="flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm text-foreground"
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
                          <span className="flex items-center gap-2 text-muted-foreground">
                            <UserIconLucide className="h-4 w-4" />
                            Select team member
                          </span>
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
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm",
                          branch ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {branch || "Select Branch"}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="max-h-72 w-[var(--radix-popover-trigger-width)] overflow-y-auto p-1" align="start">
                      {BRANCHES.length === 0 ? (
                        <div className="px-2.5 py-2 text-sm text-muted-foreground">No branches available</div>
                      ) : (
                        BRANCHES.map((b) => (
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
                      className="flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm text-foreground"
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
                    {AVAILABLE_MEMBERS.length === 0 ? (
                      <div className="px-2.5 py-2 text-sm text-muted-foreground">No unassigned users available</div>
                    ) : (
                      AVAILABLE_MEMBERS.map((m) => {
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

              {members.length > 0 && (
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
                        <button
                          onClick={() => removeMember(m.name)}
                          className="text-rose-500 hover:text-rose-600"
                          aria-label={`Remove ${m.name}`}
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="flex justify-end border-t border-border dark:border-white/10 px-5 py-4">
              <Button
                onClick={() => setStep("confirm")}
                disabled={!canSubmit}
                className="h-11 gap-2 bg-[#1565EF] px-5 font-semibold text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] dark:disabled:bg-white/5 disabled:text-muted-foreground"
              >
                <Plus className="h-4 w-4" />
                Add Team
              </Button>
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
                Are you sure you want to add team
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
              <h3 className="mt-4 text-lg font-semibold text-foreground">Successfully added team</h3>
              <p className="mt-1 text-sm text-muted-foreground">{name}</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
