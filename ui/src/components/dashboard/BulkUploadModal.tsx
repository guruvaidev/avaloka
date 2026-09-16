import { useEffect, useMemo, useRef, useState } from "react";
import {
  Upload,
  X,
  HelpCircle,
  CheckCircle2,
  Trash2,
  FileText,
  Download,
  AlertCircle,
} from "lucide-react";
import { useServerFn } from "@tanstack/react-start";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  bulkUpsertDepartments,
  bulkUpsertConfigRoles,
  bulkUpsertBranches,
  inviteAppUser,
} from "@/lib/configurations.functions";

type Step = "upload" | "confirm" | "success";

export type BulkEntityKey =
  | "departments"
  | "roles"
  | "branches"
  | "users"
  | "teams"
  | "models";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  entity?: string;
  entityKey?: BulkEntityKey;
}

interface UploadedFile {
  name: string;
  size: number;
  progress: number;
  text?: string;
}

interface DeptRow {
  department: string;
  code: string;
  head_name: string | null;
  head_email: string | null;
}

interface RoleRow {
  role: string;
  department: string;
  permissions: string[];
}

interface BranchRow {
  name: string;
  location: string;
  head_name: string | null;
  head_email: string | null;
  employees: number;
}

interface UserRow {
  name: string;
  email: string;
  role: string;
  department: string;
  branch: string;
}

type AnyRow = DeptRow | RoleRow | BranchRow | UserRow;

const DEPT_TEMPLATE = `department,code,head_name,head_email
Data Analytics,DA,Olivia Rhye,olivia@avaloka.ai
Data Research,DR,,
`;

const ROLES_TEMPLATE = `role,department,permissions
Analyst,Data Analytics,"read,write"
Viewer,Data Research,read
`;

const BRANCHES_TEMPLATE = `name,location,head_name,head_email,employees
HQ,New York,Olivia Rhye,olivia@avaloka.ai,42
Remote,Worldwide,,,0
`;

const USERS_TEMPLATE = `name,email,role,department,branch
Olivia Rhye,olivia@avaloka.ai,Analyst,Data Analytics,HQ
Phoenix Baker,phoenix@avaloka.ai,Admin,Data Research,HQ
`;

// Minimal CSV parser supporting quoted values and "" escapes.
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let cur: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") {
        cur.push(field);
        field = "";
      } else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        cur.push(field);
        field = "";
        rows.push(cur);
        cur = [];
      } else field += c;
    }
  }
  if (field.length || cur.length) {
    cur.push(field);
    rows.push(cur);
  }
  return rows.filter((r) => r.some((v) => v.trim().length));
}

function downloadBlob(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function BulkUploadModal({
  open,
  onOpenChange,
  entity = "Divisions",
  entityKey,
}: Props) {
  const [step, setStep] = useState<Step>("upload");
  const [file, setFile] = useState<UploadedFile | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [insertedCount, setInsertedCount] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const bulkDeptFn = useServerFn(bulkUpsertDepartments);
  const bulkRolesFn = useServerFn(bulkUpsertConfigRoles);
  const bulkBranchesFn = useServerFn(bulkUpsertBranches);
  const inviteFn = useServerFn(inviteAppUser);
  const qc = useQueryClient();
  const isDepartments = entityKey === "departments" || entityKey === ("department" as BulkEntityKey);
  const isRoles = entityKey === "roles";
  const isBranches = entityKey === "branches";
  const isUsers = entityKey === "users" || entityKey === ("user" as BulkEntityKey);
  const isCsv = isDepartments || isRoles || isBranches || isUsers;

  useEffect(() => {
    if (!open) {
      setTimeout(() => {
        setStep("upload");
        setFile(null);
        setSubmitting(false);
        setInsertedCount(0);
      }, 200);
    }
  }, [open]);

  // Parse + validate CSV per entity
  const { rows, errors } = useMemo(() => {
    if (!isCsv || !file?.text) return { rows: [] as AnyRow[], errors: [] as string[] };
    const parsed = parseCsv(file.text);
    if (!parsed.length) return { rows: [] as AnyRow[], errors: ["File is empty"] };
    const header = parsed[0].map((h) => h.trim().toLowerCase());
    const idx = (k: string) => header.indexOf(k);
    const errs: string[] = [];
    const out: AnyRow[] = [];

    if (isDepartments) {
      const iDept = idx("department");
      const iCode = idx("code");
      if (iDept < 0 || iCode < 0) {
        return { rows: [], errors: ["Header must include 'department' and 'code' columns"] };
      }
      const iName = idx("head_name");
      const iEmail = idx("head_email");
      for (let r = 1; r < parsed.length; r++) {
        const row = parsed[r];
        const department = (row[iDept] ?? "").trim();
        const code = (row[iCode] ?? "").trim().toUpperCase();
        if (!department || !code) {
          errs.push(`Line ${r + 1}: department and code are required`);
          continue;
        }
        out.push({
          department,
          code,
          head_name: iName >= 0 ? (row[iName] ?? "").trim() || null : null,
          head_email: iEmail >= 0 ? (row[iEmail] ?? "").trim().toLowerCase() || null : null,
        });
      }
    } else if (isRoles) {
      const iRole = idx("role");
      const iDept = idx("department");
      if (iRole < 0 || iDept < 0) {
        return { rows: [], errors: ["Header must include 'role' and 'department' columns"] };
      }
      const iPerms = idx("permissions");
      for (let r = 1; r < parsed.length; r++) {
        const row = parsed[r];
        const role = (row[iRole] ?? "").trim();
        const department = (row[iDept] ?? "").trim();
        if (!role || !department) {
          errs.push(`Line ${r + 1}: role and department are required`);
          continue;
        }
        const permsRaw = iPerms >= 0 ? (row[iPerms] ?? "").trim() : "";
        const permissions = permsRaw
          ? permsRaw.split(/[,;|]/).map((p) => p.trim()).filter(Boolean)
          : [];
        out.push({
          role,
          department,
          permissions,
        });
      }
    } else if (isBranches) {
      const iName = idx("name");
      const iLoc = idx("location");
      if (iName < 0 || iLoc < 0) {
        return { rows: [], errors: ["Header must include 'name' and 'location' columns"] };
      }
      const iHeadName = idx("head_name");
      const iHeadEmail = idx("head_email");
      const iEmp = idx("employees");
      for (let r = 1; r < parsed.length; r++) {
        const row = parsed[r];
        const name = (row[iName] ?? "").trim();
        const location = (row[iLoc] ?? "").trim();
        if (!name || !location) {
          errs.push(`Line ${r + 1}: name and location are required`);
          continue;
        }
        const empRaw = iEmp >= 0 ? (row[iEmp] ?? "").trim() : "";
        const employees = empRaw ? Math.max(0, parseInt(empRaw, 10) || 0) : 0;
        out.push({
          name,
          location,
          head_name: iHeadName >= 0 ? (row[iHeadName] ?? "").trim() || null : null,
          head_email: iHeadEmail >= 0 ? (row[iHeadEmail] ?? "").trim().toLowerCase() || null : null,
          employees,
        });
      }
    } else if (isUsers) {
      const iName = idx("name");
      const iEmail = idx("email");
      const iRole = idx("role");
      const iDept = idx("department");
      const iBranch = idx("branch");
      if (iName < 0 || iEmail < 0) {
        return { rows: [], errors: ["Header must include 'name' and 'email' columns"] };
      }
      const seen = new Set<string>();
      for (let r = 1; r < parsed.length; r++) {
        const row = parsed[r];
        const name = (row[iName] ?? "").trim();
        const email = (row[iEmail] ?? "").trim().toLowerCase();
        if (!name || !email) {
          errs.push(`Line ${r + 1}: name and email are required`);
          continue;
        }
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
          errs.push(`Line ${r + 1}: invalid email "${email}"`);
          continue;
        }
        if (seen.has(email)) {
          errs.push(`Line ${r + 1}: duplicate email "${email}"`);
          continue;
        }
        seen.add(email);
        out.push({
          name,
          email,
          role: iRole >= 0 ? (row[iRole] ?? "").trim() || "Analyst" : "Analyst",
          department: iDept >= 0 ? (row[iDept] ?? "").trim() : "",
          branch: iBranch >= 0 ? (row[iBranch] ?? "").trim() : "",
        });
      }
    }
    if (!out.length && !errs.length) errs.push("No data rows found");
    return { rows: out, errors: errs };
  }, [isCsv, isDepartments, isRoles, isBranches, isUsers, file?.text]);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const f = files[0];
    if (isCsv) {
      const text = await f.text();
      setFile({ name: f.name, size: f.size, progress: 100, text });
    } else {
      setFile({ name: f.name, size: f.size, progress: 100 });
    }
  };

  const handleDownloadTemplate = () => {
    if (isDepartments) {
      downloadBlob("departments-template.csv", DEPT_TEMPLATE, "text/csv");
    } else if (isRoles) {
      downloadBlob("roles-template.csv", ROLES_TEMPLATE, "text/csv");
    } else if (isBranches) {
      downloadBlob("branches-template.csv", BRANCHES_TEMPLATE, "text/csv");
    } else if (isUsers) {
      downloadBlob("users-template.csv", USERS_TEMPLATE, "text/csv");
    } else {
      toast.info("Template not available for this section yet.");
    }
  };

  const handleConfirm = async () => {
    if (!isCsv) {
      setStep("success");
      setTimeout(() => onOpenChange(false), 1800);
      return;
    }
    if (submitting) return;
    setSubmitting(true);
    try {
      let res: any;
      let invalidateKey: string[] = [];
      if (isDepartments) {
        res = await bulkDeptFn({ data: { rows: rows as DeptRow[] } });
        invalidateKey = ["config", "departments"];
      } else if (isRoles) {
        res = await bulkRolesFn({ data: { rows: rows as RoleRow[] } });
        invalidateKey = ["config", "roles"];
      } else if (isBranches) {
        res = await bulkBranchesFn({ data: { rows: rows as BranchRow[] } });
        invalidateKey = ["config", "branches"];
      } else if (isUsers) {
        const userRows = rows as UserRow[];
        const redirectTo =
          typeof window !== "undefined" ? `${window.location.origin}/auth` : undefined;
        let inserted = 0;
        const failures: string[] = [];
        for (const u of userRows) {
          try {
            const result: any = await inviteFn({
              data: {
                name: u.name,
                email: u.email,
                role: u.role,
                department: u.department,
                branch: u.branch,
                permissions: {},
                redirectTo,
              },
            });
            if (result?.ok === false) {
              failures.push(`${u.email}: ${result.message ?? "skipped"}`);
            } else {
              inserted++;
            }
          } catch (err) {
            failures.push(`${u.email}: ${err instanceof Error ? err.message : "failed"}`);
          }
        }
        res = { inserted };
        invalidateKey = ["config", "users"];
        if (failures.length) {
          toast.warning(`${failures.length} row(s) skipped`, {
            description: failures.slice(0, 4).join("\n"),
          });
        }
      }
      setInsertedCount(res?.inserted ?? rows.length);
      await qc.invalidateQueries({ queryKey: invalidateKey });
      setStep("success");
      setTimeout(() => onOpenChange(false), 1800);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Bulk upload failed");
      setSubmitting(false);
    }
  };


  const canSubmit =
    !!file &&
    file.progress >= 100 &&
    (!isCsv || (rows.length > 0 && errors.length === 0));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {step === "upload" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[520px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white dark:bg-[#0f1216] dark:border dark:border-white/10 shadow-xl duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border dark:border-white/10 px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border dark:border-white/10 bg-white dark:bg-white/5">
                  <Upload className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">Bulk Upload</h2>
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="space-y-4 px-5 py-5">
              {/* Dropzone */}
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  handleFiles(e.dataTransfer.files);
                }}
                className={cn(
                  "flex flex-col items-center justify-center gap-2 rounded-xl border bg-white px-6 py-8 text-center transition",
                  dragOver || file ? "border-[#1565EF] border-2" : "border-border dark:border-white/10",
                )}
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border dark:border-white/10 bg-white dark:bg-white/5">
                  <Upload className="h-4 w-4 text-foreground" />
                </div>
                <div className="text-sm">
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    className="font-semibold text-[#1565EF] hover:underline"
                  >
                    Select
                  </button>{" "}
                  <span className="text-muted-foreground">or drag and drop file</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  {isCsv ? "CSV only (use the template below)" : "CSV or Excel file"}
                </p>
                <input
                  ref={inputRef}
                  type="file"
                  className="hidden"
                  accept={isCsv ? ".csv,text/csv" : ".csv,.xlsx,.xls"}
                  onChange={(e) => handleFiles(e.target.files)}
                />
              </div>

              {/* File card */}
              {file && (
                <div className="flex items-center gap-3 rounded-xl border border-border dark:border-white/10 bg-white dark:bg-white/5 p-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-red-50 dark:bg-red-500/15">
                    <FileText className="h-5 w-5 text-red-500 dark:text-red-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <p className="truncate text-sm font-medium text-foreground">{file.name}</p>
                      <button
                        onClick={() => setFile(null)}
                        className="text-muted-foreground hover:text-foreground"
                        aria-label="Remove"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                    <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                      <span>{formatSize(file.size)}</span>
                      {file.progress >= 100 && (
                        <span className="flex items-center gap-1 text-emerald-600">
                          <span className="text-muted-foreground">|</span>
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          {isCsv ? `${rows.length} row${rows.length === 1 ? "" : "s"} parsed` : "Complete"}
                        </span>
                      )}
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-[#1565EF] transition-all"
                          style={{ width: `${file.progress}%` }}
                        />
                      </div>
                      <span className="w-10 text-right text-xs font-medium text-foreground">
                        {file.progress}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Validation errors */}
              {isCsv && file && errors.length > 0 && (
                <div className="rounded-xl border border-red-200 dark:border-red-500/30 bg-red-50 dark:bg-red-500/15 p-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-red-700 dark:text-red-400">
                    <AlertCircle className="h-4 w-4" />
                    {errors.length} issue{errors.length === 1 ? "" : "s"} found
                  </div>
                  <ul className="mt-1.5 ml-6 list-disc space-y-0.5 text-xs text-red-700 dark:text-red-400">
                    {errors.slice(0, 6).map((e, i) => (
                      <li key={i}>{e}</li>
                    ))}
                    {errors.length > 6 && <li>...and {errors.length - 6} more</li>}
                  </ul>
                </div>
              )}

              {/* Steps */}
              <div className="space-y-2">
                <p className="text-center text-sm text-muted-foreground">Steps to Bulk Upload</p>
                <button
                  type="button"
                  onClick={handleDownloadTemplate}
                  className="flex w-full items-start gap-3 rounded-xl border border-border dark:border-white/10 bg-white dark:bg-white/5 px-4 py-3 text-left transition hover:border-[#1565EF]"
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border dark:border-white/10 text-xs font-semibold text-foreground">
                    01
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-foreground">Download CSV Template</p>
                    <p className="text-xs text-muted-foreground">
                      Click here to download the {entity} template
                    </p>
                  </div>
                  <Download className="mt-1 h-4 w-4 text-[#1565EF]" />
                </button>
                <StepCard n="02" title="Fill Data" desc="Fill in the required details in downloaded template" />
                <StepCard n="03" title="Upload" desc="Verify details and upload the completed CSV." />
              </div>
            </div>

            <div className="border-t border-border dark:border-white/10 px-5 py-4">
              <Button
                onClick={() => setStep("confirm")}
                disabled={!canSubmit}
                className="h-11 w-full gap-2 bg-[#1565EF] text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] dark:disabled:bg-white/5 disabled:text-muted-foreground"
              >
                <Upload className="h-4 w-4" />
                Bulk Upload
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
                Are you sure you want to bulk upload
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">
                {isCsv ? `${rows.length} ${entity}?` : `${entity}?`}
              </p>
              <div className="mt-6 grid w-full grid-cols-2 gap-3">
                <Button
                  variant="outline"
                  onClick={() => setStep("upload")}
                  disabled={submitting}
                  className="h-11 gap-2"
                >
                  <X className="h-4 w-4" />
                  Cancel
                </Button>
                <Button
                  onClick={handleConfirm}
                  disabled={submitting}
                  className="h-11 gap-2 bg-[#1565EF] text-white hover:bg-[#1257cf]"
                >
                  <CheckCircle2 className="h-4 w-4" />
                  {submitting ? "Uploading..." : "Confirm"}
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
                {isCsv ? `Successfully added ${insertedCount}` : "Successfully bulk uploaded"}
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">{entity}</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}

function StepCard({ n, title, desc }: { n: string; title: string; desc: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-border dark:border-white/10 bg-white dark:bg-white/5 px-4 py-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border dark:border-white/10 text-xs font-semibold text-foreground">
        {n}
      </div>
      <div>
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <p className="text-xs text-muted-foreground">{desc}</p>
      </div>
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
