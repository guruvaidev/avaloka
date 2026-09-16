import { useEffect, useMemo, useState } from "react";
import { ChevronDown, CheckCircle2, X, HelpCircle, XCircle, Check } from "lucide-react";
import { useServerFn } from "@tanstack/react-start";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { UserIcon } from "@/components/dashboard/icons/ConfigSectionIcons";
import { cn } from "@/lib/utils";
import {
  upsertAppUser,
  listDepartments,
  listConfigRoles,
  listBranches,
} from "@/lib/configurations.functions";

type Action = "update" | "deactivate" | "activate";
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

const seedPerms = (): Perms => ({
  "User Management": { View: true, Create: true, Edit: true, Approve: false },
  Analysis: { View: true, Create: true, Edit: true, Approve: false },
  Projects: { View: true, Create: true, Edit: true, Approve: true },
  Dashboard: { View: false, Create: true, Edit: true, Approve: false },
  "Data Configuration": { View: false, Create: false, Edit: false, Approve: false },
});

export interface EditUser {
  id: string;
  name: string;
  email: string;
  userId: string;
  role: string;
  department: string;
  branch: string;
  status: "Active" | "Inactive" | "Invited";
  permissions?: Perms | null;
  phone?: string | null;
  country?: string | null;
}

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  user: EditUser | null;
  readOnly?: boolean;
}

export function EditUserModal({ open, onOpenChange, user, readOnly = false }: Props) {
  const upsertUserFn = useServerFn(upsertAppUser);
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

  const ROLES: string[] = (roleRows as any[]).map((r) => r.role).filter(Boolean);
  const DEPARTMENTS: string[] = (deptRows as any[]).map((d) => d.department).filter(Boolean);
  const LOCATIONS: string[] = (branchRows as any[])
    .map((b) => b.location || b.name)
    .filter(Boolean);

  const [step, setStep] = useState<Step>("form");
  const [action, setAction] = useState<Action>("update");

  const [fullName, setFullName] = useState("");
  const [userId, setUserId] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("");
  const [department, setDepartment] = useState("");
  const [location, setLocation] = useState("");
  const [country, setCountry] = useState(COUNTRY_CODES[0]);
  const [phone, setPhone] = useState("");
  const [perms, setPerms] = useState<Perms>(seedPerms);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [roleOpen, setRoleOpen] = useState(false);
  const [deptOpen, setDeptOpen] = useState(false);
  const [locOpen, setLocOpen] = useState(false);
  const [ccOpen, setCcOpen] = useState(false);

  const inactive = user?.status === "Inactive";
  const pending = user?.status === "Invited";
  const locked = inactive || pending || readOnly;

  const [initial, setInitial] = useState({
    fullName: "",
    userId: "",
    email: "",
    role: "",
    department: "",
    location: "",
    phone: "",
    perms: JSON.stringify(seedPerms()),
  });

  useEffect(() => {
    if (open && user) {
      setStep("form");
      const merged = seedPerms();
      if (user.permissions) {
        for (const c of Object.keys(merged)) {
          const saved = user.permissions[c];
          if (saved) merged[c] = { ...merged[c], ...saved } as typeof merged[string];
        }
      }
      const seed = {
        fullName: user.name,
        userId: user.userId,
        email: user.email,
        role: user.role,
        department: user.department,
        location: user.branch,
        phone: user.phone ?? "",
        perms: JSON.stringify(merged),
      };
      const matchedCountry =
        COUNTRY_CODES.find((c) => c.code === user.country) ?? COUNTRY_CODES[0];
      setFullName(seed.fullName);
      setUserId(seed.userId);
      setEmail(seed.email);
      setRole(seed.role);
      setDepartment(seed.department);
      setLocation(seed.location);
      setPhone(seed.phone);
      setCountry(matchedCountry);
      setPerms(merged);
      setInitial(seed);
      setError(null);
    }
  }, [user?.id, open]);

  const togglePerm = (company: string, a: ActionKey) => {
    if (locked || (company === "Analysis" && a === "View")) return;
    setPerms((prev) => ({
      ...prev,
      [company]: { ...prev[company], [a]: !prev[company][a] },
    }));
  };

  const hasChanges = useMemo(
    () =>
      fullName !== initial.fullName ||
      userId !== initial.userId ||
      email !== initial.email ||
      role !== initial.role ||
      department !== initial.department ||
      location !== initial.location ||
      phone !== initial.phone ||
      JSON.stringify(perms) !== initial.perms,
    [fullName, userId, email, role, department, location, phone, perms, initial],
  );

  const canUpdate =
    !locked &&
    Boolean(fullName && userId && email && role && department && location && hasChanges);

  const startConfirm = (a: Action) => {
    setAction(a);
    setStep("confirm");
  };

  const handleConfirm = async () => {
    if (!user) return;
    setSubmitting(true);
    setError(null);
    try {
      const payload =
        action === "deactivate"
          ? {
              id: user.id,
              name: user.name,
              email: user.email,
              user_id_code: user.userId,
              role: user.role,
              department: user.department,
              branch: user.branch,
              status: "Inactive" as const,
            }
          : action === "activate"
          ? {
              id: user.id,
              name: user.name,
              email: user.email,
              user_id_code: user.userId,
              role: user.role,
              department: user.department,
              branch: user.branch,
              status: "Active" as const,
            }
          : {
              id: user.id,
              name: fullName.trim(),
              email: email.trim(),
              user_id_code: userId.trim(),
              role,
              department,
              branch: location,
              status: user.status,
              permissions: Object.fromEntries(
                Object.entries(perms).map(([c, row]) => [
                  c,
                  c === "Analysis" ? { ...row, View: true } : row,
                ]),
              ),
            };
      await upsertUserFn({ data: payload });
      await qc.invalidateQueries({ queryKey: ["config", "users"] });
      await qc.invalidateQueries({ queryKey: ["config", "departments"] });
      await qc.invalidateQueries({ queryKey: ["config", "branches"] });
      setStep("success");
      setTimeout(() => onOpenChange(false), 1500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save");
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
                  <UserIcon className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">User</h2>
                {inactive && (
                  <span className="inline-flex items-center rounded-full border border-border bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                    Inactive
                  </span>
                )}
                {pending && (
                  <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-700">
                    Pending
                  </span>
                )}
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="max-h-[68vh] space-y-4 overflow-y-auto px-5 py-5">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Full Name</label>
                  <Input
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    disabled={locked}
                    className="h-10 disabled:opacity-70"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">User ID</label>
                  <Input
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    disabled={locked}
                    className="h-10 disabled:opacity-70"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Email</label>
                <Input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={locked}
                  className="h-10 disabled:opacity-70"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Role</label>
                  <Popover open={roleOpen} onOpenChange={setRoleOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                        disabled={locked}
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm disabled:opacity-70",
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
                          onClick={() => {
                            setRole(r);
                            setRoleOpen(false);
                          }}
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
                        disabled={locked}
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm disabled:opacity-70",
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
                        disabled={locked}
                        className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm disabled:opacity-70",
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
                  <div
                    className={cn(
                      "flex h-10 items-center rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-2 text-sm",
                      locked && "opacity-70",
                    )}
                  >
                    <Popover open={ccOpen} onOpenChange={setCcOpen}>
                      <PopoverTrigger asChild>
                        <button
                          type="button"
                          disabled={locked}
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
                      disabled={locked}
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
                                  disabled={(c === "Analysis" && a === "View") || locked}
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

            <div className="flex justify-end gap-3 border-t border-border dark:border-white/10 px-5 py-4">
              {readOnly || pending ? (
                <Button
                  variant="outline"
                  onClick={() => onOpenChange(false)}
                  className="h-11 gap-2 font-semibold"
                >
                  Close
                </Button>
              ) : inactive ? (
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
                Are you sure you want to {action} user
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
                  className="h-11 gap-2 bg-[#1565EF] font-semibold text-white hover:bg-[#1257cf]"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  Confirm
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
                {action === "update" ? "updated" : action === "activate" ? "activated" : "deactivated"} user
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{fullName}</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
