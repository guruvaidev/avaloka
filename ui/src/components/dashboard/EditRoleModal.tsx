import { useEffect, useMemo, useState } from "react";
import { ChevronDown, CheckCircle2, X, HelpCircle, XCircle, Check } from "lucide-react";
import { useServerFn } from "@tanstack/react-start";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { RolesIcon } from "@/components/dashboard/icons/ConfigSectionIcons";
import { upsertConfigRole, listDepartments } from "@/lib/configurations.functions";
import { cn } from "@/lib/utils";

type Action = "update" | "deactivate" | "activate";
type Step = "form" | "confirm" | "success";
type PermKey = "view" | "create" | "edit" | "approve";

const MODULES = ["User Management", "Analysis", "Projects", "Dashboard", "Data Configuration"];
const PERM_COLS: { key: PermKey; label: string }[] = [
  { key: "view", label: "View" },
  { key: "create", label: "Create" },
  { key: "edit", label: "Edit" },
  { key: "approve", label: "Approve" },
];

type PermissionState = Record<string, Record<PermKey, boolean>>;

const buildPerms = (granted: string[]): PermissionState => {
  const acc = MODULES.reduce((a, m) => {
    a[m] = { view: false, create: false, edit: false, approve: false };
    return a;
  }, {} as PermissionState);
  for (const g of granted) {
    const [m, k] = g.split(":") as [string, PermKey];
    if (acc[m] && (k === "view" || k === "create" || k === "edit" || k === "approve")) {
      acc[m][k] = true;
    }
  }
  return acc;
};

export interface EditRole {
  id: string;
  name: string;
  department: string;
  permissions: string[];
}

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  role: EditRole | null;
}

export function EditRoleModal({ open, onOpenChange, role }: Props) {
  const [step, setStep] = useState<Step>("form");
  const [action, setAction] = useState<Action>("update");
  const [name, setName] = useState("");
  const [department, setDepartment] = useState("");
  const [deptOpen, setDeptOpen] = useState(false);
  const [perms, setPerms] = useState<PermissionState>(() => buildPerms([]));
  const [initialPerms, setInitialPerms] = useState<PermissionState>(() => buildPerms([]));
  const [submitting, setSubmitting] = useState(false);

  const upsertRoleFn = useServerFn(upsertConfigRole);
  const listDeptsFn = useServerFn(listDepartments);
  const { data: deptRows } = useQuery({
    queryKey: ["config", "departments"],
    queryFn: () => listDeptsFn(),
  });
  const deptOptions = useMemo(
    () => (deptRows ?? []).map((d: any) => d.department).filter(Boolean),
    [deptRows],
  );
  const qc = useQueryClient();

  useEffect(() => {
    if (open && role) {
      setStep("form");
      setSubmitting(false);
      setName(role.name);
      setDepartment(role.department);
      const seed = buildPerms(role.permissions);
      setPerms(seed);
      setInitialPerms(seed);
    }
  }, [open, role]);

  const hasChanges = useMemo(() => {
    if (!role) return false;
    if (name !== role.name || department !== role.department) return true;
    return MODULES.some((m) =>
      PERM_COLS.some((c) => perms[m]?.[c.key] !== initialPerms[m]?.[c.key]),
    );
  }, [role, name, department, perms, initialPerms]);

  const canUpdate = Boolean(name && department && hasChanges);

  const togglePerm = (m: string, k: PermKey) =>
    setPerms((prev) => ({ ...prev, [m]: { ...prev[m], [k]: !prev[m][k] } }));

  const startConfirm = (a: Action) => {
    setAction(a);
    setStep("confirm");
  };

  const handleConfirm = async () => {
    if (!role || submitting) return;
    setSubmitting(true);
    try {
      const permissions: string[] = [];
      for (const m of MODULES) {
        for (const c of PERM_COLS) {
          if (perms[m]?.[c.key]) permissions.push(`${m}:${c.key}`);
        }
      }
      await upsertRoleFn({
        data: { id: role.id, role: name, department, permissions },
      });
      await qc.invalidateQueries({ queryKey: ["config", "roles"] });
      setStep("success");
      setTimeout(() => onOpenChange(false), 1500);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save role");
      setSubmitting(false);
      setStep("form");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {step === "form" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[720px] max-w-[94vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white dark:bg-[#0f1216] dark:border dark:border-white/10 shadow-xl duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
            <div className="flex items-center justify-between border-b border-border dark:border-white/10 px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border dark:border-white/10 bg-white dark:bg-white/5">
                  <RolesIcon className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">Role</h2>
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="space-y-4 px-5 py-5">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Role Name</label>
                  <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="h-10 disabled:opacity-70"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-foreground">Division</label>
                  <Popover open={deptOpen} onOpenChange={setDeptOpen}>
                    <PopoverTrigger asChild>
                      <button
                        type="button"
                            className={cn(
                          "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm disabled:opacity-70",
                          department ? "text-foreground" : "text-muted-foreground",
                        )}
                      >
                        {department || "Select Division"}
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent
                      className="w-[var(--radix-popover-trigger-width)] p-1"
                      align="start"
                    >
                      {deptOptions.length === 0 ? (
                        <div className="px-2.5 py-2 text-sm text-muted-foreground">
                          No divisions found
                        </div>
                      ) : (
                        deptOptions.map((d: string) => (
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
                        ))
                      )}
                    </PopoverContent>
                  </Popover>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Permissions</label>
                <div className="overflow-hidden rounded-lg border border-border dark:border-white/10">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border dark:border-white/10 bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <th className="px-4 py-3 font-medium">Company</th>
                        {PERM_COLS.map((c) => (
                          <th key={c.key} className="px-4 py-3 font-medium">
                            {c.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {MODULES.map((m) => (
                        <tr key={m} className="border-b border-border dark:border-white/10 last:border-0">
                          <td className="px-4 py-3 text-sm text-foreground">{m}</td>
                          {PERM_COLS.map((c) => (
                            <td key={c.key} className="px-4 py-3">
                              <Switch
                                checked={perms[m]?.[c.key] ?? false}
                                onCheckedChange={() => togglePerm(m, c.key)}
                                            className="data-[state=checked]:bg-[#1565EF]"
                              />
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
              <Button
                onClick={() => startConfirm("update")}
                disabled={!canUpdate}
                className="h-11 gap-2 bg-[#1565EF] px-5 font-semibold text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] dark:disabled:bg-white/5 disabled:text-muted-foreground"
              >
                <CheckCircle2 className="h-4 w-4" />
                Update
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
                Are you sure you want to {action}
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{name} Role?</p>
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
                Successfully {action === "update" ? "updated" : action === "activate" ? "activated" : "deactivated"}
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{name} Role</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
