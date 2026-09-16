import { useEffect, useMemo, useState } from "react";
import { ChevronDown, CheckCircle2, X, HelpCircle, XCircle } from "lucide-react";
import { useServerFn } from "@tanstack/react-start";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { DepartmentIcon } from "@/components/dashboard/icons/ConfigSectionIcons";
import { upsertDepartment, listAppUsers } from "@/lib/configurations.functions";
import { cn } from "@/lib/utils";

type Action = "update";
type Step = "form" | "confirm" | "success";

const DEPARTMENT_OPTIONS = [
  "Data Analytics",
  "Data Research",
  "Data Engineering",
  "Data Science",
  "Data Operations",
];

export interface EditDept {
  id: string;
  name: string;
  code: string;
  head?: { name: string; email: string; avatar?: string; profile_id?: string | null } | null;
}

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  department: EditDept | null;
}

export function EditDepartmentModal({ open, onOpenChange, department }: Props) {
  const [step, setStep] = useState<Step>("form");
  const action: Action = "update";
  const [name, setName] = useState("");
  const [code, setCode] = useState("");
  const [head, setHead] = useState<{ name: string; email: string; avatar?: string; profile_id?: string | null } | null>(null);
  const [deptOpen, setDeptOpen] = useState(false);
  const [headOpen, setHeadOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const inactive = false;


  const upsertDeptFn = useServerFn(upsertDepartment);
  const listUsersFn = useServerFn(listAppUsers);
  const qc = useQueryClient();

  const { data: appUsers } = useQuery({
    queryKey: ["config", "appUsers", "active"],
    queryFn: () => listUsersFn(),
    enabled: open,
  });

  const userOptions = useMemo(() => {
    if (!appUsers || !Array.isArray(appUsers)) return [];
    return appUsers
      .filter((u: any) => (u.status ?? "Active") === "Active")
      .filter((u: any) => !name || (u.department ?? "").toLowerCase() === name.toLowerCase())
      .map((u: any) => ({
        name: u.name,
        email: u.email,
        avatar: u.avatar ?? "",
        profile_id: u.profile_id ?? null,
      }));
  }, [appUsers, name]);

  useEffect(() => {
    if (open && department) {
      setStep("form");
      setSubmitting(false);
      setName(department.name);
      setCode(department.code);
      setHead(
        department.head
          ? {
              name: department.head.name,
              email: department.head.email,
              avatar: department.head.avatar ?? "",
              profile_id: department.head.profile_id ?? null,
            }
          : null,
      );
    }
  }, [open, department]);

  const hasChanges = useMemo(() => {
    if (!department) return false;
    return (
      name !== department.name ||
      code !== department.code ||
      (head?.email ?? null) !== (department.head?.email ?? null)
    );
  }, [department, name, code, head]);

  const canUpdate = Boolean(name && code && hasChanges);

  const startConfirm = (_a: Action) => {
    setStep("confirm");
  };

  const handleConfirm = async () => {
    if (!department || submitting) return;
    setSubmitting(true);
    try {
      await upsertDeptFn({
        data: {
          id: department.id,
          department: name,
          code,
          head_profile_id: head?.profile_id ?? null,
        },
      });
      await qc.invalidateQueries({ queryKey: ["config", "departments"] });
      setStep("success");
      setTimeout(() => onOpenChange(false), 1500);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to save division");
      setSubmitting(false);
      setStep("form");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {step === "form" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[440px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white dark:bg-[#0f1216] dark:border dark:border-white/10 shadow-xl duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border dark:border-white/10 px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border dark:border-white/10 bg-white dark:bg-white/5">
                  <DepartmentIcon className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">Division</h2>
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            {/* Body */}
            <div className="space-y-4 px-5 py-5">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Division Name</label>
                <Popover open={deptOpen} onOpenChange={(o) => !inactive && setDeptOpen(o)}>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      disabled={inactive}
                      className="flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm text-foreground disabled:opacity-70"
                    >
                      {name || "Select Division"}
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                    {DEPARTMENT_OPTIONS.map((d) => (
                      <button
                        key={d}
                        onClick={() => {
                          setName(d);
                          if (head && d !== name) setHead(null);
                          setDeptOpen(false);
                        }}
                        className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:bg-white/10 dark:hover:bg-white/5"
                      >
                        {d}
                      </button>
                    ))}
                  </PopoverContent>
                </Popover>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Division Code</label>
                <Input
                  value={code}
                  onChange={(e) => setCode(e.target.value.toUpperCase())}
                  disabled={inactive}
                  className="h-10 disabled:opacity-70"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Division Head{" "}
                  <span className="font-normal text-muted-foreground">(Optional)</span>
                </label>
                <Popover open={headOpen} onOpenChange={(o) => !inactive && setHeadOpen(o)}>
                  <PopoverTrigger asChild>
                    <button
                      type="button"
                      disabled={inactive}
                      className={cn(
                        "flex h-10 w-full items-center justify-between rounded-md border border-border dark:border-white/10 bg-white dark:bg-white/5 px-3 text-sm disabled:opacity-70",
                        head ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {head ? (
                        <span className="flex items-center gap-2">
                          <Avatar className="h-6 w-6">
                            <AvatarImage src={head.avatar} alt={head.name} />
                            <AvatarFallback>{head.name[0]}</AvatarFallback>
                          </Avatar>
                          {head.name}
                        </span>
                      ) : (
                        "Select Division Head"
                      )}
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                    {userOptions.length === 0 ? (
                      <div className="px-2.5 py-2 text-sm text-muted-foreground">No users found</div>
                    ) : (
                      userOptions.map((h: { name: string; email: string; avatar?: string }) => (
                        <button
                          key={h.email}
                          onClick={() => {
                            setHead(h);
                            setHeadOpen(false);
                          }}
                          className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-sm hover:bg-muted dark:bg-white/10 dark:hover:bg-white/5"
                        >
                          <Avatar className="h-6 w-6">
                            <AvatarImage src={h.avatar} alt={h.name} />
                            <AvatarFallback>{h.name[0]}</AvatarFallback>
                          </Avatar>
                          <span className="flex flex-col items-start">
                            <span className="text-sm text-foreground">{h.name}</span>
                            <span className="text-xs text-muted-foreground">{h.email}</span>
                          </span>
                        </button>
                      ))
                    )}
                  </PopoverContent>
                </Popover>
              </div>
            </div>

            {/* Footer */}
            <div className="flex justify-end gap-3 border-t border-border dark:border-white/10 px-5 py-4">
              <Button
                onClick={() => startConfirm("update")}
                disabled={!canUpdate}
                className="h-11 w-full gap-2 bg-[#1565EF] font-semibold text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] dark:disabled:bg-white/5 disabled:text-muted-foreground"
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
              <p className="mt-1 text-sm text-muted-foreground">{name} Division?</p>
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
                Successfully updated
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{name} Division</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
