import { useEffect, useMemo, useState } from "react";
import { ChevronDown, CheckCircle2, X, HelpCircle, XCircle, Plus, Check } from "lucide-react";
import { useServerFn } from "@tanstack/react-start";
import { toast } from "sonner";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { UserIcon } from "@/components/dashboard/icons/ConfigSectionIcons";
import { cn } from "@/lib/utils";
import { CircularLoader } from "@/components/ui/circular-loader";
import {
  inviteAppUser,
  checkInviteEmail,
  listDepartments,
  listConfigRoles,
  listBranches,
} from "@/lib/configurations.functions";

type Step = "form" | "confirm" | "success";
const COUNTRY_CODES = [
  { code: "IN", dial: "+91" },
  { code: "US", dial: "+1" },
  { code: "UK", dial: "+44" },
  { code: "AU", dial: "+61" },
];

const COMPANIES = ["User Management", "Analysis", "Projects", "Dashboard", "Data Configuration"];
const ACTIONS = ["View", "Create", "Edit", "Approve"] as const;
type ActionKey = (typeof ACTIONS)[number];

type Perms = Record<string, Record<ActionKey, boolean>>;

// Build the permission matrix from a role row's `permissions` array
// (stored as "Module:action" tokens, e.g. "User Management:view").
const permsFromRoleTokens = (tokens: unknown): Perms => {
  const base = emptyPerms();
  if (!Array.isArray(tokens)) return base;
  const moduleByLower = new Map(COMPANIES.map((c) => [c.toLowerCase(), c]));
  const actionByLower = new Map(ACTIONS.map((a) => [a.toLowerCase(), a] as const));
  for (const token of tokens) {
    if (typeof token !== "string") continue;
    const [rawModule, rawAction] = token.split(":");
    if (!rawModule || !rawAction) continue;
    const mod = moduleByLower.get(rawModule.trim().toLowerCase());
    const act = actionByLower.get(rawAction.trim().toLowerCase());
    if (mod && act) base[mod][act] = true;
  }
  return base;
};

// Serialize the matrix to lowercase-keyed JSON for storage.
const permsToLowerJson = (perms: Perms): Record<string, Record<string, boolean>> => {
  const out: Record<string, Record<string, boolean>> = {};
  for (const c of COMPANIES) {
    const row: Record<string, boolean> = {};
    // Analysis "View" is always granted and cannot be turned off.
    for (const a of ACTIONS)
      row[a.toLowerCase()] =
        c === "Analysis" && a === "View" ? true : Boolean(perms[c]?.[a]);
    out[c.toLowerCase()] = row;
  }
  return out;
};



const emptyPerms = (): Perms =>
  COMPANIES.reduce((acc, c) => {
    acc[c] = { View: c === "Analysis", Create: false, Edit: false, Approve: false };
    return acc;
  }, {} as Perms);

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function NewUserModal({ open, onOpenChange }: Props) {
  const inviteUserFn = useServerFn(inviteAppUser);
  const qc = useQueryClient();

  const listDeptsFn = useServerFn(listDepartments);
  const listRolesFn = useServerFn(listConfigRoles);
  const listBranchesFn = useServerFn(listBranches);

  const { data: deptRows = [] } = useQuery({
    queryKey: ["config", "departments"],
    queryFn: () => listDeptsFn({}),
    enabled: open,
  });
  const { data: roleRows = [] } = useQuery({
    queryKey: ["config", "roles"],
    queryFn: () => listRolesFn({}),
    enabled: open,
  });
  const { data: branchRows = [] } = useQuery({
    queryKey: ["config", "branches"],
    queryFn: () => listBranchesFn({}),
    enabled: open,
  });

  const baseRoles: string[] = (roleRows as any[]).map((r) => r.role).filter(Boolean);
  const ROLES: string[] = baseRoles.some((r) => r.toLowerCase() === "custom")
    ? baseRoles
    : [...baseRoles, "Custom"];
  const DEPARTMENTS: string[] = (deptRows as any[]).map((d) => d.department).filter(Boolean);
  const LOCATIONS: string[] = (branchRows as any[])
    .map((b) => b.location || b.name)
    .filter(Boolean);

  // Analyst-like defaults: view enabled on every module, others off.
  const buildCustomDefaults = (): Perms => {
    const base = emptyPerms();
    for (const c of COMPANIES) base[c].View = true;
    return base;
  };

  const [step, setStep] = useState<Step>("form");

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const fullName = `${firstName.trim()} ${lastName.trim()}`.trim();
  // user_id_code is auto-generated server-side from the selected department.
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("");
  const [department, setDepartment] = useState("");
  const [location, setLocation] = useState("");
  const [country, setCountry] = useState(COUNTRY_CODES[0]);
  const [phone, setPhone] = useState("");
  const [perms, setPerms] = useState<Perms>(emptyPerms);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [roleOpen, setRoleOpen] = useState(false);
  const [deptOpen, setDeptOpen] = useState(false);
  const [locOpen, setLocOpen] = useState(false);
  const [ccOpen, setCcOpen] = useState(false);

  const resetAll = () => {
    setStep("form");
    setFirstName("");
    setLastName("");
    setEmail("");
    setRole("");
    setDepartment("");
    setLocation("");
    setCountry(COUNTRY_CODES[0]);
    setPhone("");
    setPerms(emptyPerms());
    setError(null);
    setSubmitting(false);
    setLinked(null);
  };


  const handleOpenChange = (o: boolean) => {
    if (!o) resetAll();
    onOpenChange(o);
  };

  const selectRole = (r: string) => {
    setRole(r);
    setRoleOpen(false);
    if (r.toLowerCase() === "custom") {
      setPerms(buildCustomDefaults());
      return;
    }
    const roleRow = (roleRows as any[]).find((row) => row?.role === r);
    setPerms(permsFromRoleTokens(roleRow?.permissions));
  };


  const togglePerm = (company: string, action: ActionKey) => {
    if (!role || (company === "Analysis" && action === "View")) return;
    setPerms((prev) => ({
      ...prev,
      [company]: { ...prev[company], [action]: !prev[company][action] },
    }));
    // Any manual permission tweak reclassifies the user as "Custom".
    if (role.toLowerCase() !== "custom") setRole("Custom");
  };

  // Debounced inline check: is this email already used by another organization?
  const checkEmailFn = useServerFn(checkInviteEmail);
  const [emailError, setEmailError] = useState<string | null>(null);
  const [checkingEmail, setCheckingEmail] = useState(false);

  useEffect(() => {
    const value = email.trim().toLowerCase();
    if (!value || !value.includes("@")) {
      setEmailError(null);
      setCheckingEmail(false);
      return;
    }
    let cancelled = false;
    setCheckingEmail(true);
    const t = setTimeout(async () => {
      try {
        const res: any = await checkEmailFn({ data: { email: value } });
        if (cancelled) return;
        setEmailError(res?.available === false ? res.message : null);
      } catch {
        if (!cancelled) setEmailError(null);
      } finally {
        if (!cancelled) setCheckingEmail(false);
      }
    }, 500);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [email, checkEmailFn]);

  const canSubmit = useMemo(
    () =>
      Boolean(
        fullName.trim() &&
          email.trim() &&
          role &&
          department &&
          location &&
          !emailError &&
          !checkingEmail,
      ),
    [fullName, email, role, department, location, emailError, checkingEmail],
  );

  // Preview the auto-generated User ID based on the selected department's code.
  const selectedDeptCode = useMemo(() => {
    const row = (deptRows as any[]).find((d) => d.department === department);
    const raw = (row?.code || department || "").toString();
    return raw.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 6);
  }, [deptRows, department]);
  const userIdPreview = department
    ? selectedDeptCode
      ? `${selectedDeptCode}-###`
      : "Auto-generated"
    : "Select a division first";

  const [linked, setLinked] = useState<boolean | null>(null);

  const handleConfirm = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const res: any = await inviteUserFn({
        data: {
          name: fullName.trim(),
          email: email.trim(),
          role,
          department,
          branch: location,
          permissions: permsToLowerJson(perms),
          redirectTo:
            typeof window !== "undefined" ? window.location.origin : undefined,
        },
      });
      if (res && res.ok === false) {
        const msg = res.message || "Failed to add user";
        setError(msg);
        toast.error(msg);
        setStep("form");
        return;
      }
      await qc.invalidateQueries({ queryKey: ["config", "users"] });
      await qc.invalidateQueries({ queryKey: ["config", "departments"] });
      await qc.invalidateQueries({ queryKey: ["config", "branches"] });
      setLinked(Boolean(res?.linked));
      setStep("success");
      setTimeout(() => handleOpenChange(false), 2000);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to add user";
      setError(msg);
      toast.error(msg);
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
                  <UserIcon className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">User</h2>
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="max-h-[68vh] space-y-4 overflow-y-auto px-5 py-5">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">First Name</label>
                  <Input
                    value={firstName}
                    onChange={(e) => setFirstName(e.target.value)}
                    placeholder="Enter First Name"
                    className="h-10"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Last Name</label>
                  <Input
                    value={lastName}
                    onChange={(e) => setLastName(e.target.value)}
                    placeholder="Enter Last Name"
                    className="h-10"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">User ID</label>
                  <Input
                    value={userIdPreview}
                    readOnly
                    disabled
                    placeholder="Auto-generated from division"
                    className="h-10 cursor-not-allowed bg-muted/40 dark:bg-white/5 text-muted-foreground"
                  />
                  <p className="text-xs text-muted-foreground">
                    Generated automatically from the selected department code.
                  </p>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Email</label>
                  <Input
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter Email ID"
                    aria-invalid={Boolean(emailError)}
                    className={cn(
                      "h-10",
                      emailError && "border-destructive focus-visible:ring-destructive",
                    )}
                  />
                  {emailError ? (
                    <p className="text-xs text-destructive">{emailError}</p>
                  ) : null}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Role</label>
                  <Popover open={roleOpen} onOpenChange={setRoleOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm",
                          role ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {role || "Select Role"}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                      {ROLES.map((r) => (
                        <button
                          key={r}
                          onClick={() => selectRole(r)}
                          className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                        >
                          {r}
                          {role === r && <Check className="h-4 w-4 text-primary" />}
                        </button>
                      ))}
                    </PopoverContent>
                  </Popover>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Division</label>
                  <Popover open={deptOpen} onOpenChange={setDeptOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm",
                          department ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {department || "Select Division"}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                      {DEPARTMENTS.map((d) => (
                        <button
                          key={d}
                          onClick={() => {
                            setDepartment(d);
                            setDeptOpen(false);
                          }}
                          className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                        >
                          {d}
                          {department === d && <Check className="h-4 w-4 text-primary" />}
                        </button>
                      ))}
                    </PopoverContent>
                  </Popover>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Branch Location</label>
                  <Popover open={locOpen} onOpenChange={setLocOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm",
                          location ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {location || "Select Location"}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                      {LOCATIONS.map((l) => (
                        <button
                          key={l}
                          onClick={() => {
                            setLocation(l);
                            setLocOpen(false);
                          }}
                          className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:hover:bg-white/5"
                        >
                          {l}
                          {location === l && <Check className="h-4 w-4 text-primary" />}
                        </button>
                      ))}
                    </PopoverContent>
                  </Popover>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Phone number</label>
                  <div className="flex h-10 items-center rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-2 text-sm">
                    <Popover open={ccOpen} onOpenChange={setCcOpen}>
                      <PopoverTrigger asChild>
                        <button
                          type="button"
                          className="flex items-center gap-1 pr-2 text-foreground"
                        >
                          {country.code}
                          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                        </button>
                      </PopoverTrigger>
                      <PopoverContent className="w-32 p-1" align="start">
                        {COUNTRY_CODES.map((c) => (
                          <button
                            key={c.code}
                            onClick={() => {
                              setCountry(c);
                              setCcOpen(false);
                            }}
                            className="flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-sm hover:bg-muted dark:hover:bg-white/5"
                          >
                            <span>{c.code}</span>
                            <span className="text-muted-foreground">{c.dial}</span>
                          </button>
                        ))}
                      </PopoverContent>
                    </Popover>
                    <span className="border-l border-border pl-2 pr-1 text-muted-foreground">
                      {country.dial}
                    </span>
                    <input
                      value={phone}
                      onChange={(e) => setPhone(e.target.value.replace(/[^\d]/g, ""))}
                      className="h-full flex-1 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-2 pt-2">
                <h3 className="text-sm font-medium text-foreground">Permissions</h3>
                <div className="overflow-hidden rounded-lg border border-border dark:border-white/10">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <th className="px-4 py-3 font-medium">Company</th>
                        {ACTIONS.map((a) => (
                          <th key={a} className="px-4 py-3 text-center font-medium">
                            {a}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {COMPANIES.map((c) => (
                        <tr key={c} className="border-t border-border dark:border-white/10">
                          <td className="px-4 py-3 text-sm text-foreground">{c}</td>
                          {ACTIONS.map((a) => (
                            <td key={a} className="px-4 py-3">
                              <div className="flex justify-center">
                                <Switch
                                  checked={
                                    c === "Analysis" && a === "View"
                                      ? true
                                      : (perms[c]?.[a] ?? false)
                                  }
                                  onCheckedChange={() => togglePerm(c, a)}
                                  disabled={(c === "Analysis" && a === "View") || !role}
                                  className="data-[state=checked]:bg-[#1565EF]"
                                />
                              </div>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="flex justify-end border-t border-border dark:border-white/10 px-5 py-4">
              <Button
                onClick={() => setStep("confirm")}
                disabled={!canSubmit}
                className="h-11 gap-2 bg-[#1565EF] px-5 font-semibold text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] dark:disabled:bg-white/5 disabled:text-muted-foreground"
              >
                <Plus className="h-4 w-4" />
                Add User
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
                Are you sure you want to add user
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{fullName}?</p>
              <div className="mt-6 grid w-full grid-cols-2 gap-3">
                <Button
                  variant="outline"
                  onClick={() => setStep("form")}
                  className="h-11 gap-2 font-semibold"
                >
                  <XCircle className="h-4 w-4" />
                  Cancel
                </Button>
                <Button
                  onClick={handleConfirm}
                  disabled={submitting}
                  className="h-11 gap-2 bg-[#1565EF] font-semibold text-white hover:bg-[#1257cf] disabled:opacity-70"
                >
                  {submitting ? (
                    <>
                      <CircularLoader size={16} strokeWidth={2} color="#ffffff" trackColor="rgba(255,255,255,0.35)" />
                      Adding...
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="h-4 w-4" />
                      Confirm
                    </>
                  )}
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
                {linked ? "User added to your organization" : "Invitation sent"}
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">
                {linked
                  ? `${fullName} now has access.`
                  : `We emailed a signup link to ${email}.`}
              </p>

            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
