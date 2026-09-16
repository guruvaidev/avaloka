import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { listDepartments, listConfigRoles, listBranches, listAppUsers, listTeams, resendAppUserInvite, deleteAppUser } from "@/lib/configurations.functions";
import { toast } from "sonner";

import { supabase } from "@/integrations/supabase/client";
import { PlanGate } from "@/components/dashboard/PlanGate";
import { usePlanFeatures } from "@/lib/use-plan-features";

import {
  Search,
  Upload,
  Plus,
  ChevronsUpDown,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Trash2,
} from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";



import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

import { NewDepartmentModal } from "@/components/dashboard/NewDepartmentModal";
import { BulkUploadModal } from "@/components/dashboard/BulkUploadModal";
import { EditDepartmentModal, type EditDept } from "@/components/dashboard/EditDepartmentModal";
import { EditRoleModal, type EditRole } from "@/components/dashboard/EditRoleModal";
import { NewRoleModal } from "@/components/dashboard/NewRoleModal";
import { NewBranchModal } from "@/components/dashboard/NewBranchModal";
import { EditBranchModal, type EditBranch } from "@/components/dashboard/EditBranchModal";
import { NewUserModal } from "@/components/dashboard/NewUserModal";
import { NewTeamModal } from "@/components/dashboard/NewTeamModal";
import { EditTeamModal, type EditTeam } from "@/components/dashboard/EditTeamModal";
import { RegisterModelModal } from "@/components/dashboard/RegisterModelModal";
import { ModelDetailView } from "@/components/dashboard/ModelDetailView";
import { backendApi, type BackendModel } from "@/lib/api/backendApi";
import { EditUserModal, type EditUser } from "@/components/dashboard/EditUserModal";
import {
  DepartmentIcon,
  RolesIcon,
  BranchesIcon,
  UserIcon,
  TeamsIcon,
  ModelsIcon,
} from "@/components/dashboard/icons/ConfigSectionIcons";
import {
  MOCK_DEPARTMENTS,
  MOCK_ROLES,
  MOCK_BRANCHES,
  MOCK_APP_USERS,
  MOCK_TEAMS,
} from "@/lib/configurations.mock";
import { cn } from "@/lib/utils";
import { useUserMgmtPerms } from "@/lib/use-user-mgmt-perms";
import { getCachedAuthUser } from "@/lib/auth-user";

function formatModelDate(input: string | null): string | null {
  if (!input) return null;
  const trimmed = input.trim();
  if (!trimmed) return null;
  // Numeric epoch (ms or seconds)
  const num = Number(trimmed);
  let d: Date | null = null;
  if (Number.isFinite(num) && trimmed.match(/^\d+$/)) {
    d = new Date(num > 1e12 ? num : num * 1000);
  } else {
    const parsed = new Date(trimmed);
    if (!Number.isNaN(parsed.getTime())) d = parsed;
  }
  if (!d || Number.isNaN(d.getTime())) return trimmed;
  const now = Date.now();
  const diffSec = Math.round((now - d.getTime()) / 1000);
  const abs = Math.abs(diffSec);
  if (abs < 60) return "just now";
  if (abs < 3600) return `${Math.round(abs / 60)}m ago`;
  if (abs < 86400) return `${Math.round(abs / 3600)}h ago`;
  if (abs < 86400 * 7) return `${Math.round(abs / 86400)}d ago`;
  return d.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
}


export const Route = createFileRoute("/_authenticated/configurations")({
  head: () => ({
    meta: [
      { title: "Configurations · Avaloka AI" },
      { name: "description", content: "Manage divisions, roles, branches, users, teams and models." },
    ],
  }),
  component: () => (
    <PlanGate>
      <ConfigurationsPage />
    </PlanGate>
  ),
});

type SectionId = "department" | "roles" | "branches" | "user" | "teams" | "models";

const SECTIONS: { id: SectionId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "department", label: "Division", icon: DepartmentIcon },
  { id: "roles", label: "Roles", icon: RolesIcon },
  { id: "branches", label: "Branches", icon: BranchesIcon },
  { id: "user", label: "User", icon: UserIcon },
  { id: "teams", label: "Teams", icon: TeamsIcon },
  { id: "models", label: "Models", icon: ModelsIcon },
];


type DbStatus = "Active" | "Inactive";

type Dept = {
  id: string;
  department: string;
  code: string;
  head_name: string | null;
  head_email: string | null;
  head_avatar: string | null;
  employees: number;
};

type Role = {
  id: string;
  role: string;
  department: string;
  permissions: string[];
};

type Branch = {
  id: string;
  name: string;
  location: string;
  head_profile_id: string | null;
  head_name: string | null;
  head_avatar: string | null;
  employees: number;
};

type AppUser = {
  id: string;
  name: string;
  email: string;
  avatar: string | null;
  user_id_code: string;
  role: string;
  department: string;
  branch: string;
  status: "Active" | "Inactive" | "Invited";
  phone?: string | null;
  country?: string | null;
  permissions?: Record<string, Record<string, boolean>> | null;
};

type Team = {
  id: string;
  name: string;
  description: string | null;
  lead_user_id: string | null;
  lead_name: string | null;
  lead_email: string | null;
  lead_avatar: string | null;
  user_count: number;
  user_avatars: string[];
  member_ids: string[];
  branch: string;
  status: DbStatus;
};

type Model = BackendModel;

const PERMISSION_STYLES: Record<string, string> = {
  Analysis: "bg-blue-50 text-blue-700 ring-blue-200",
  Projects: "bg-blue-50 text-blue-700 ring-blue-200",
  Reports: "bg-amber-50 text-amber-700 ring-amber-200",
  Cost: "bg-pink-50 text-pink-700 ring-pink-200",
  Revenue: "bg-emerald-50 text-emerald-700 ring-emerald-200",
};


function ConfigurationsPage() {
  const [section, setSection] = useState<SectionId>("department");
  const [tableSearch, setTableSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<DbStatus>("Active");
  const [page, setPage] = useState(1);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [newDeptOpen, setNewDeptOpen] = useState(false);
  const [newRoleOpen, setNewRoleOpen] = useState(false);
  const [newBranchOpen, setNewBranchOpen] = useState(false);
  const [newUserOpen, setNewUserOpen] = useState(false);
  const [newTeamOpen, setNewTeamOpen] = useState(false);
  const [registerModelOpen, setRegisterModelOpen] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [editTeam, setEditTeam] = useState<EditTeam | null>(null);
  const [editUser, setEditUser] = useState<EditUser | null>(null);
  const [editBranch, setEditBranch] = useState<EditBranch | null>(null);
  const [editRole, setEditRole] = useState<EditRole | null>(null);
  const [bulkOpen, setBulkOpen] = useState(false);
  const [editDept, setEditDept] = useState<EditDept | null>(null);

  const userPerms = useUserMgmtPerms();
  const planFeatures = usePlanFeatures();
  const gateUserCreate = section === "user" && !userPerms.canCreate;

  // Professional plan: Models only. Otherwise hide the User tab when the
  // current user lacks View permission.
  const allowedSections = useMemo(
    () =>
      planFeatures.canUseAllConfigSections
        ? SECTIONS.filter((s) => s.id !== "user" || userPerms.canView)
        : SECTIONS.filter((s) => s.id === "models"),
    [userPerms.canView, planFeatures.canUseAllConfigSections],
  );

  useEffect(() => {
    if (!allowedSections.some((s) => s.id === section)) {
      setSection(allowedSections[0]?.id ?? "models");
    }
  }, [allowedSections, section]);

  useEffect(() => {
    if (section === "user" && !userPerms.loading && !userPerms.canView) {
      setSection("department");
    }
  }, [section, userPerms.loading, userPerms.canView]);

  const listDepartmentsFn = useServerFn(listDepartments);
  const listConfigRolesFn = useServerFn(listConfigRoles);
  const listBranchesFn = useServerFn(listBranches);
  const listAppUsersFn = useServerFn(listAppUsers);
  const listTeamsFn = useServerFn(listTeams);

  const deptQ = useQuery<Dept[]>({
    queryKey: ["config", "departments"],
    queryFn: async () => (await listDepartmentsFn()) as Dept[],
  });
  const rolesQ = useQuery<Role[]>({
    queryKey: ["config", "roles"],
    queryFn: async () => (await listConfigRolesFn()) as Role[],
    enabled: section === "roles",
  });
  const branchesQ = useQuery<Branch[]>({
    queryKey: ["config", "branches"],
    queryFn: async () => (await listBranchesFn()) as Branch[],
    enabled: section === "branches",
  });
  const usersQ = useQuery<AppUser[]>({
    queryKey: ["config", "users"],
    queryFn: async () => (await listAppUsersFn()) as AppUser[],
    enabled: section === "user",
  });
  const resendInviteFn = useServerFn(resendAppUserInvite);
  const [resendingId, setResendingId] = useState<string | null>(null);
  const handleResendInvite = async (userId: string) => {
    setResendingId(userId);
    try {
      const res: any = await resendInviteFn({
        data: {
          id: userId,
          redirectTo: typeof window !== "undefined" ? window.location.origin : undefined,
        },
      });
      if (res?.ok === false) {
        toast.message(res.message ?? "This user has already accepted the invitation.");
      } else if (res?.activated) {
        toast.success("This user already accepted — marked as active.");
      } else {
        toast.success(`Invitation resent to ${res?.email ?? "the user"}`);
      }
      usersQ.refetch();
    } catch (e: any) {
      toast.error(e?.message ?? "Could not resend the invitation.");
    } finally {
      setResendingId(null);
    }
  };

  const deleteAppUserFn = useServerFn(deleteAppUser);
  const [pendingRemove, setPendingRemove] = useState<AppUser | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const handleRemoveUser = async () => {
    const target = pendingRemove;
    if (!target) return;
    setDeletingId(target.id);
    try {
      await deleteAppUserFn({ data: { id: target.id } });
      toast.success(`${target.name || target.email} removed from the organization`);
      setPendingRemove(null);
      usersQ.refetch();
      deptQ.refetch();
      branchesQ.refetch();
    } catch (e: any) {
      toast.error(e?.message ?? "Could not remove this user.");
    } finally {
      setDeletingId(null);
    }
  };



  const teamsQ = useQuery<Team[]>({
    queryKey: ["config", "teams"],
    queryFn: async () => (await listTeamsFn()) as Team[],
    enabled: section === "teams",
  });
  const modelsQ = useQuery<Model[]>({
    queryKey: ["config", "models"],
    queryFn: async () => {
      const res = await backendApi.listModels();
      return Array.isArray(res.models) ? res.models : [];
    },
    enabled: section === "models",
  });



  const departments = deptQ.data ?? [];
  const rolesData = rolesQ.data ?? [];
  const branchesData = branchesQ.data ?? [];
  const usersData = usersQ.data ?? [];
  const teamsData = teamsQ.data ?? [];
  const modelsData = modelsQ.data ?? [];

  const uidQ = useQuery({
    queryKey: ["auth", "uid"],
    queryFn: async () => (await getCachedAuthUser())?.id ?? null,
    staleTime: 5 * 60_000,
  });
  const selfAppUserId = useMemo(() => {
    const uid = uidQ.data;
    if (!uid) return null;
    const me = usersData.find((u: any) => u.auth_user_id === uid);
    return me?.id ?? null;
  }, [uidQ.data, usersData]);
  const isEditingSelf = !!editUser && !!selfAppUserId && editUser.id === selfAppUserId;
  const userReadOnly = (section === "user" && !userPerms.canEdit) || isEditingSelf;

  const filteredRows = useMemo(() => {
    const q = tableSearch.toLowerCase();
    return departments.filter((d) => {
      if (!q) return true;
      return (
        d.department.toLowerCase().includes(q) ||
        d.code.toLowerCase().includes(q) ||
        (d.head_name ?? "").toLowerCase().includes(q) ||
        (d.head_email ?? "").toLowerCase().includes(q)
      );
    });
  }, [tableSearch, departments]);

  const filteredRoles = useMemo(
    () =>
      rolesData.filter((r) => {
        if (!tableSearch) return true;
        const q = tableSearch.toLowerCase();
        return r.role.toLowerCase().includes(q) || r.department.toLowerCase().includes(q);
      }),
    [rolesData, tableSearch],
  );

  const filteredBranches = useMemo(
    () =>
      branchesData.filter((b) => {
        if (!tableSearch) return true;
        const q = tableSearch.toLowerCase();
        return (
          b.name.toLowerCase().includes(q) ||
          b.location.toLowerCase().includes(q) ||
          (b.head_name ?? "").toLowerCase().includes(q)
        );
      }),
    [branchesData, tableSearch],
  );

  const filteredUsers = useMemo(
    () =>
      usersData
        .filter((u) => {
          // Active tab includes fully active users plus invited-but-not-yet-accepted users.
          if (statusFilter === "Active" && u.status !== "Active" && u.status !== "Invited") return false;
          if (statusFilter === "Inactive" && u.status !== "Inactive") return false;
          if (!tableSearch) return true;
          const q = tableSearch.toLowerCase();
          return (
            u.name.toLowerCase().includes(q) ||
            u.email.toLowerCase().includes(q) ||
            u.user_id_code.toLowerCase().includes(q) ||
            u.role.toLowerCase().includes(q) ||
            u.department.toLowerCase().includes(q) ||
            u.branch.toLowerCase().includes(q)
          );
        })
        // Organization owner always appears first.
        .sort((a, b) => {
          const rank = (r: string) => (String(r).toLowerCase() === "owner" ? 0 : 1);
          return rank(a.role) - rank(b.role);
        }),
    [usersData, tableSearch, statusFilter],
  );

  const filteredTeams = useMemo(
    () =>
      teamsData.filter((t) => {
        if (t.status !== statusFilter) return false;
        if (!tableSearch) return true;
        const q = tableSearch.toLowerCase();
        return (
          t.name.toLowerCase().includes(q) ||
          (t.description ?? "").toLowerCase().includes(q) ||
          (t.lead_name ?? "").toLowerCase().includes(q) ||
          t.branch.toLowerCase().includes(q)
        );
      }),
    [teamsData, tableSearch, statusFilter],
  );

  const filteredModels = useMemo(
    () =>
      modelsData.filter((m) => {
        if (!tableSearch) return true;
        const q = tableSearch.toLowerCase();
        const raw = m as unknown as Record<string, unknown>;
        const name = String(raw.name ?? raw.model_name ?? raw.run_name ?? raw.registered_model_name ?? raw.display_name ?? "");
        return name.toLowerCase().includes(q) || String(m.model_type ?? "").toLowerCase().includes(q);
      }),
    [modelsData, tableSearch],
  );

  const sectionLoading =
    (section === "department" && deptQ.isLoading) ||
    (section === "roles" && rolesQ.isLoading) ||
    (section === "branches" && branchesQ.isLoading) ||
    (section === "user" && usersQ.isLoading) ||
    (section === "teams" && teamsQ.isLoading) ||
    (section === "models" && modelsQ.isLoading);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 bg-[#f4f6fb] text-foreground dark:bg-[#0b0d10]">



      {sidebarOpen && (
        <aside className="flex w-[280px] shrink-0 flex-col border-r border-border bg-white dark:bg-[#111418]">
          <div className="flex items-center justify-between px-5 pt-5 pb-3">
            <h2 className="text-base font-semibold tracking-tight">Configurations</h2>
            <button
              onClick={() => setSidebarOpen(false)}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Collapse sidebar"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12.5 2.5V17.5M6.5 2.5H13.5C14.9001 2.5 15.6002 2.5 16.135 2.77248C16.6054 3.01217 16.9878 3.39462 17.2275 3.86502C17.5 4.3998 17.5 5.09987 17.5 6.5V13.5C17.5 14.9001 17.5 15.6002 17.2275 16.135C16.9878 16.6054 16.6054 16.9878 16.135 17.2275C15.6002 17.5 14.9001 17.5 13.5 17.5H6.5C5.09987 17.5 4.3998 17.5 3.86502 17.2275C3.39462 16.9878 3.01217 16.6054 2.77248 16.135C2.5 15.6002 2.5 14.9001 2.5 13.5V6.5C2.5 5.09987 2.5 4.3998 2.77248 3.86502C3.01217 3.39462 3.39462 3.01217 3.86502 2.77248C4.3998 2.5 5.09987 2.5 6.5 2.5Z" stroke="currentColor" strokeWidth="1.66667" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
          <nav className="flex-1 overflow-y-auto px-3 py-1">
            {allowedSections.map((s) => {
              const Icon = s.icon;
              const active = s.id === section;
              return (
                <button
                  key={s.id}
                  onClick={() => setSection(s.id)}
                  className={cn(
                    "mb-1 flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                    active
                      ? "bg-[#f4f6fb] font-semibold text-foreground dark:bg-white/10"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {s.label}
                </button>
              );
            })}
          </nav>
        </aside>
      )}


      {/* Main panel */}
      <main className="flex flex-1 flex-col overflow-y-auto bg-[#f4f6fb] dark:bg-[#0b0d10]">
        {section === "models" && selectedModelId ? (
          <ModelDetailView runId={selectedModelId} onBack={() => setSelectedModelId(null)} />
        ) : (
        <div className="m-4 flex w-auto flex-1 flex-col rounded-2xl border border-border bg-white shadow-sm dark:bg-[#111418] sm:m-6">

          {/* Header */}
          <header className="flex items-center justify-between border-b border-border px-6 py-4">
            <div className="flex items-center gap-3">
              {(() => {
                const Icon = SECTIONS.find((s) => s.id === section)?.icon ?? DepartmentIcon;
                return <Icon className="h-5 w-5 text-foreground" />;
              })()}
              <div>
                <h1 className="text-lg font-semibold tracking-tight">
                  {(() => {
                    const l = SECTIONS.find((s) => s.id === section)?.label ?? "";
                    return l.endsWith("s") ? l : `${l}s`;
                  })()}
                </h1>
                <p className="text-xs text-muted-foreground">
                  {section === "models"
                    ? "View all the Registered Models"
                    : `View all the created ${(() => {
                        const l = SECTIONS.find((s) => s.id === section)?.label ?? "";
                        return l.endsWith("s") ? l : `${l}s`;
                      })()}`}
                </p>

              </div>
            </div>
            <div className="flex items-center gap-2">
              {!["models", "department", "roles", "branches"].includes(section) && !gateUserCreate && (
                <Button variant="outline" onClick={() => setBulkOpen(true)} className="gap-2">
                  <Upload className="h-4 w-4" />
                  Bulk Upload
                </Button>
              )}
              {!gateUserCreate && (
                <Button
                  onClick={() => {
                    if (section === "department") setNewDeptOpen(true);
                    else if (section === "roles") setNewRoleOpen(true);
                    else if (section === "branches") setNewBranchOpen(true);
                    else if (section === "user") setNewUserOpen(true);
                    else if (section === "teams") setNewTeamOpen(true);
                    else if (section === "models") setRegisterModelOpen(true);
                  }}
                  className="gap-2 bg-[#1565EF] hover:bg-[#1257cf]"
                >
                  <Plus className="h-4 w-4" />
                  {section === "models" ? "Register Model" : `New ${SECTIONS.find((s) => s.id === section)?.label}`}
                </Button>
              )}
            </div>
          </header>

          {/* Table card */}
          <section className="flex flex-1 flex-col p-6">
            <div className="flex flex-1 flex-col rounded-xl border border-border bg-white dark:bg-[#111418]">
              <div className="flex items-center justify-between border-b border-border px-5 py-3">
                <div className="relative w-[320px] max-w-full">
                  <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={tableSearch}
                    onChange={(e) => {
                      setTableSearch(e.target.value);
                      setPage(1);
                    }}
                    placeholder="Search"
                    className="h-9 pl-9 pr-12 text-sm"
                  />
                  <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                    ⌘K
                  </span>
                </div>
                {section !== "models" && section !== "department" && section !== "roles" && section !== "branches" && (
                  <div className="inline-flex items-center rounded-md border border-border bg-white p-0.5 shadow-sm dark:bg-white/5">
                    <button
                      type="button"
                      onClick={() => {
                        setStatusFilter("Inactive");
                        setPage(1);
                      }}
                      className={cn(
                        "rounded px-3 py-1 text-sm font-medium transition-colors",
                        statusFilter === "Inactive"
                          ? "bg-[#f4f6fb] text-foreground dark:bg-[#0b0d10]"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      Inactive
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setStatusFilter("Active");
                        setPage(1);
                      }}
                      className={cn(
                        "rounded px-3 py-1 text-sm font-medium transition-colors",
                        statusFilter === "Active"
                          ? "bg-[#f4f6fb] text-foreground dark:bg-[#0b0d10]"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      Active
                    </button>
                  </div>
                )}
              </div>

              {/* Table */}
              <div className="flex-1 overflow-x-auto">
                {sectionLoading ? (
                  <div className="flex items-center justify-center px-6 py-16 text-sm text-muted-foreground">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading…
                  </div>
                ) : section === "roles" ? (
                  <table className="w-full text-sm">
                    <thead>

                      <tr className="border-b border-border bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <Th label="Role" sortable />
                        <Th label="Division" sortable />
                        <Th label="Permissions" />
                      </tr>
                    </thead>
                    <tbody>
                      {filteredRoles.map((r) => (
                        <tr
                          key={r.id}
                          onClick={() =>
                            setEditRole({
                              id: r.id,
                              name: r.role,
                              department: r.department,
                              permissions: r.permissions,
                            })
                          }
                          className="cursor-pointer border-b border-border last:border-0 hover:bg-muted/30"
                        >
                          <td className="px-5 py-3.5 text-sm text-foreground">{r.role}</td>
                          <td className="px-5 py-3.5 text-sm text-foreground">{r.department}</td>
                          <td className="px-5 py-3.5">
                            <div className="flex flex-wrap items-center gap-1.5">
                              {r.permissions.slice(0, 2).map((p) => (
                                <span
                                  key={p}
                                  className={cn(
                                    "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
                                    PERMISSION_STYLES[p] ?? "bg-muted text-foreground ring-border",
                                  )}
                                >
                                  {p}
                                </span>
                              ))}
                              {r.permissions.length > 2 && (
                                <span className="inline-flex items-center rounded-full border border-border bg-white dark:bg-white/5 px-2 py-0.5 text-xs font-medium text-muted-foreground">
                                  +{r.permissions.length - 2}
                                </span>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : section === "branches" ? (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <Th label="Branch Name" sortable />
                        <Th label="Location" sortable />
                        <Th label="Branch Head" />
                        <Th label="No of Employees" />
                      </tr>
                    </thead>
                    <tbody>
                      {filteredBranches.map((b) => (
                        <tr
                          key={b.id}
                          onClick={() =>
                            setEditBranch({
                              id: b.id,
                              name: b.name,
                              location: b.location,
                              employees: b.employees,
                              head: b.head_profile_id
                                ? {
                                    name: b.head_name ?? "",
                                    avatar: b.head_avatar ?? undefined,
                                    profile_id: b.head_profile_id,
                                  }
                                : null,
                            })
                          }
                          className="cursor-pointer border-b border-border last:border-0 hover:bg-muted/30"
                        >
                          <td className="px-5 py-3.5 text-sm text-foreground">{b.name}</td>
                          <td className="max-w-[220px] px-5 py-3.5 text-sm text-foreground">
                            <span className="block truncate" title={b.location}>
                              {b.location}
                            </span>
                          </td>
                          <td className="px-5 py-3.5">
                            {b.head_name ? (
                              <div className="flex items-center gap-2.5">
                                <Avatar className="h-8 w-8">
                                  <AvatarImage src={b.head_avatar ?? undefined} alt={b.head_name} />
                                  <AvatarFallback>{b.head_name.charAt(0)}</AvatarFallback>
                                </Avatar>
                                <span className="text-sm font-medium text-foreground">{b.head_name}</span>

                              </div>
                            ) : (
                              <span className="text-sm text-muted-foreground">NA</span>
                            )}
                          </td>
                          <td className="px-5 py-3.5 text-sm text-foreground">{b.employees}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : section === "user" ? (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <Th label="User" sortable />
                        <Th label="User ID" sortable />
                        <Th label="Role" />
                        <Th label="Division" />
                        <Th label="Branch" />
                        <Th label="Status" sortable />
                      </tr>
                    </thead>
                    <tbody>
                      {filteredUsers.map((u) => {
                        const isOwnerRow =
                          String(u.role ?? "").toLowerCase() === "owner";
                        return (
                        <tr
                          key={u.id}
                          onClick={() => {
                            if (isOwnerRow) return;
                            setEditUser({
                              id: u.id,
                              name: u.name,
                              email: u.email,
                              userId: u.user_id_code,
                              role: u.role,
                              department: u.department,
                              branch: u.branch,
                              status: u.status,
                              permissions: u.permissions ?? null,
                              phone: u.phone ?? null,
                              country: u.country ?? null,
                            });
                          }}
                          className={`border-b border-border last:border-0 ${isOwnerRow ? "cursor-default bg-muted/20" : "cursor-pointer hover:bg-muted/30"}`}
                        >
                          <td className="px-5 py-3.5">
                            <div className="flex items-center gap-2.5">
                              <Avatar className="h-8 w-8">
                                <AvatarImage src={u.avatar ?? undefined} alt={u.name} />
                                <AvatarFallback>{u.name.charAt(0)}</AvatarFallback>
                              </Avatar>
                              <div className="flex flex-col">
                                <span className="text-sm font-medium text-foreground">{u.name}</span>
                                <span className="text-xs text-muted-foreground">{u.email}</span>
                              </div>
                            </div>
                          </td>
                          <td className="px-5 py-3.5 text-sm text-foreground">{u.user_id_code}</td>
                          <td className="px-5 py-3.5 text-sm text-foreground">{u.role}</td>
                          <td className="px-5 py-3.5 text-sm text-foreground">{u.department}</td>
                          <td className="px-5 py-3.5 text-sm text-foreground">{u.branch}</td>
                          <td className="px-5 py-3.5">
                            <div className="flex items-center gap-2">
                              <StatusBadge status={u.status} />
                              {u.status === "Invited" && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  disabled={resendingId === u.id}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleResendInvite(u.id);
                                  }}
                                  className="h-7 px-2 text-xs"
                                >
                                  {resendingId === u.id ? "Sending…" : "Resend invite"}
                                </Button>
                              )}
                              {userPerms.canEdit &&
                                String(u.role ?? "").toLowerCase() !== "owner" &&
                                u.id !== selfAppUserId && (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    aria-label={`Remove ${u.name}`}
                                    title="Remove from organization"
                                    disabled={deletingId === u.id}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setPendingRemove(u);
                                    }}
                                    className="h-7 w-7 p-0 text-destructive hover:bg-destructive/10 hover:text-destructive"
                                  >
                                    {deletingId === u.id ? (
                                      <Loader2 className="h-4 w-4 animate-spin" />
                                    ) : (
                                      <Trash2 className="h-4 w-4" />
                                    )}
                                  </Button>
                                )}
                            </div>

                          </td>

                        </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : section === "teams" ? (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <Th label="Team Name" sortable />
                        <Th label="Team Lead" sortable />
                        <Th label="Users" />
                        <Th label="Branch" />
                        <Th label="Status" />
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTeams.map((t) => {
                        const extra = Math.max(0, t.user_count - t.user_avatars.length);
                        const leadName = t.lead_name ?? "";
                        const leadEmail = t.lead_email ?? "";
                        const leadAvatar = t.lead_avatar ?? "";
                        return (
                          <tr
                            key={t.id}
                            onClick={() =>
                              setEditTeam({
                                id: t.id,
                                name: t.name,
                                description: t.description,
                                lead: { id: t.lead_user_id, name: leadName, email: leadEmail, avatar: leadAvatar },
                                branch: t.branch,
                                status: t.status,
                                member_ids: t.member_ids ?? [],
                              })
                            }
                            className="cursor-pointer border-b border-border last:border-0 hover:bg-muted/30"
                          >
                            <td className="px-5 py-3.5 text-sm text-foreground">{t.name}</td>
                            <td className="px-5 py-3.5">
                              <div className="flex items-center gap-2.5">
                                <Avatar className="h-8 w-8">
                                  <AvatarImage src={leadAvatar} alt={leadName} />
                                  <AvatarFallback>{leadName.charAt(0)}</AvatarFallback>
                                </Avatar>
                                <div className="flex flex-col">
                                  <span className="text-sm font-medium text-foreground">{leadName}</span>
                                  <span className="text-xs text-muted-foreground">{leadEmail}</span>
                                </div>
                              </div>
                            </td>
                            <td className="px-5 py-3.5">
                              <div className="flex items-center gap-2">
                                <span className="text-sm text-foreground">
                                  {String(t.user_count).padStart(2, "0")} users
                                </span>
                                <div className="flex -space-x-2">
                                  {t.user_avatars.map((a: string, i: number) => (
                                    <Avatar key={i} className="h-7 w-7 ring-2 ring-white">
                                      <AvatarImage src={a} alt="" />
                                      <AvatarFallback>U</AvatarFallback>
                                    </Avatar>
                                  ))}
                                  {extra > 0 && (
                                    <span className="z-10 inline-flex h-7 min-w-7 items-center justify-center rounded-full bg-[#f4f6fb] dark:bg-white/10 px-1.5 text-[11px] font-medium text-muted-foreground ring-2 ring-white dark:ring-[#111418]">
                                      +{extra}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </td>
                            <td className="px-5 py-3.5 text-sm text-foreground">{t.branch}</td>
                            <td className="px-5 py-3.5">
                              <StatusBadge status={t.status} />
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : section === "models" ? (
                  modelsQ.isError ? (
                    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
                      <div className="text-sm font-medium text-foreground">Failed to load models</div>
                      <div className="text-xs text-muted-foreground">
                        {modelsQ.error instanceof Error ? modelsQ.error.message : "Unknown error"}
                      </div>
                      <Button variant="outline" size="sm" onClick={() => modelsQ.refetch()}>
                        Retry
                      </Button>
                    </div>
                  ) : filteredModels.length === 0 ? (
                    <div className="flex flex-col items-center justify-center gap-2 py-16 text-center">
                      <div className="text-sm font-medium text-foreground">No models registered yet</div>
                      <div className="text-xs text-muted-foreground">Registered models will appear here.</div>
                    </div>
                  ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <Th label="Model Name" sortable />
                        <Th label="Task Type" sortable />
                        <Th label="Latest Version" />
                        <Th label="Last Updated" />
                      </tr>
                    </thead>
                    <tbody>
                      {filteredModels.map((m) => {
                        const raw = m as unknown as Record<string, unknown>;
                        const pick = (...keys: string[]): string | null => {
                          for (const k of keys) {
                            const v = raw[k];
                            if (v !== undefined && v !== null && v !== "") return String(v);
                          }
                          return null;
                        };
                        const name = pick("name", "model_name", "run_name", "registered_model_name", "display_name");
                        const version = pick("version", "latest_version", "model_version", "current_version");
                        const updatedRaw = pick(
                          "updated_at",
                          "last_updated",
                          "last_updated_timestamp",
                          "updated_timestamp",
                          "creation_timestamp",
                          "created_at",
                          "start_time",
                          "end_time",
                        );
                        const updated = formatModelDate(updatedRaw);
                        const taskType = m.model_type ? String(m.model_type) : "";
                        const taskLabel = taskType ? taskType.charAt(0).toUpperCase() + taskType.slice(1) : "—";
                        const runId = pick("mlflow_run_id", "run_id", "run_uuid", "runId", "id");
                        return (
                          <tr
                            key={runId ?? m.run_id ?? name ?? Math.random().toString(36)}
                            onClick={runId ? () => { console.log("[Models] row click run id:", runId); setSelectedModelId(runId); } : undefined}
                            role={runId ? "button" : undefined}
                            tabIndex={runId ? 0 : undefined}
                            onKeyDown={runId ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); console.log("[Models] row key run id:", runId); setSelectedModelId(runId); } } : undefined}
                            className={`border-b border-border last:border-0 ${runId ? "cursor-pointer hover:bg-muted/30" : ""}`}
                          >
                            <td className="px-5 py-3.5 text-sm text-foreground">{name ?? "—"}</td>
                            <td className="px-5 py-3.5 text-sm text-foreground">{taskLabel}</td>
                            <td className="px-5 py-3.5 text-sm text-muted-foreground">
                              {version ? (
                                <>
                                  Version <span className="text-foreground">{version}</span>
                                </>
                              ) : (
                                "—"
                              )}
                            </td>
                            <td className="px-5 py-3.5 text-sm text-foreground">{updated ?? "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  )


                ) : (
                  <table className="w-full text-sm">

                    <thead>
                      <tr className="border-b border-border bg-[#f9fafb] dark:bg-white/5 text-left text-xs font-medium text-muted-foreground">
                        <Th label="Division" sortable />
                        <Th label="Division Code" sortable />
                        <Th label="Division Head" />
                        <Th label="No of Employees" />
                      </tr>
                    </thead>
                    <tbody>
                      {filteredRows.map((d) => {
                        const headName = d.head_name ?? "";
                        const headEmail = d.head_email ?? "";
                        const headAvatar = d.head_avatar ?? undefined;
                        return (
                        <tr
                          key={d.id}
                          onClick={() => {
                            setEditDept({
                              id: d.id,
                              name: d.department,
                              code: d.code,
                              head: { name: headName, email: headEmail, avatar: headAvatar, profile_id: (d as any).head_profile_id ?? null },
                            });
                          }}
                          className={cn(
                            "border-b border-border last:border-0 cursor-pointer hover:bg-muted/30",
                          )}
                        >
                          <td className="px-5 py-3.5 text-sm text-foreground">{d.department}</td>
                          <td className="px-5 py-3.5">
                            <span className="inline-flex items-center rounded-full border border-border bg-white dark:bg-white/5 px-2.5 py-0.5 text-xs font-medium text-foreground">
                              {d.code}
                            </span>
                          </td>
                          <td className="px-5 py-3.5">
                            {headName ? (
                              <div className="flex items-center gap-2.5">
                                <Avatar className="h-8 w-8">
                                  <AvatarImage src={headAvatar} alt={headName} />
                                  <AvatarFallback>{headName.charAt(0)}</AvatarFallback>
                                </Avatar>
                                <div className="flex flex-col">
                                  <span className="text-sm font-medium text-foreground">{headName}</span>
                                  <span className="text-xs text-muted-foreground">{headEmail}</span>
                                </div>
                              </div>
                            ) : (
                              <span className="text-sm text-muted-foreground">-</span>
                            )}
                          </td>

                          <td className="px-5 py-3.5 text-sm text-foreground">{d.employees}</td>
                        </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </div>


              {/* Pagination */}
              {section !== "models" && (() => {
                const currentCount =
                  section === "department" ? filteredRows.length :
                  section === "roles" ? filteredRoles.length :
                  section === "branches" ? filteredBranches.length :
                  section === "user" ? filteredUsers.length :
                  section === "teams" ? filteredTeams.length : 0;
                const pageSize = 10;
                const totalPages = Math.max(1, Math.ceil(currentCount / pageSize));
                if (totalPages <= 1) return null;
                return (
                  <div className="flex items-center justify-between border-t border-border px-5 py-3">
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1"
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page <= 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                      Previous
                    </Button>
                    <Pager page={Math.min(page, totalPages)} totalPages={totalPages} onChange={setPage} />
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1"
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      disabled={page >= totalPages}
                    >
                      Next
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                );
              })()}
            </div>
          </section>
        </div>
        )}
      </main>


      <NewDepartmentModal open={newDeptOpen} onOpenChange={setNewDeptOpen} />
      <NewRoleModal open={newRoleOpen} onOpenChange={setNewRoleOpen} />
      <NewBranchModal open={newBranchOpen} onOpenChange={setNewBranchOpen} />
      <NewUserModal open={newUserOpen} onOpenChange={setNewUserOpen} />
      <NewTeamModal open={newTeamOpen} onOpenChange={setNewTeamOpen} />
      <RegisterModelModal open={registerModelOpen} onOpenChange={setRegisterModelOpen} />
      <EditTeamModal
        open={!!editTeam}
        onOpenChange={(o) => !o && setEditTeam(null)}
        team={editTeam}
      />
      <EditUserModal
        open={!!editUser}
        onOpenChange={(o) => !o && setEditUser(null)}
        user={editUser}
        readOnly={userReadOnly}
      />
      <EditBranchModal
        open={!!editBranch}
        onOpenChange={(o) => !o && setEditBranch(null)}
        branch={editBranch}
      />
      <BulkUploadModal
        open={bulkOpen}
        onOpenChange={setBulkOpen}
        entity={`${SECTIONS.find((s) => s.id === section)?.label}s`}
        entityKey={section as any}
      />

      <EditDepartmentModal
        open={!!editDept}
        onOpenChange={(o) => !o && setEditDept(null)}
        department={editDept}
      />
      <EditRoleModal
        open={!!editRole}
        onOpenChange={(o) => !o && setEditRole(null)}
        role={editRole}
      />
      <AlertDialog
        open={!!pendingRemove}
        onOpenChange={(o) => !o && setPendingRemove(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Remove {pendingRemove?.name || pendingRemove?.email} from the organization?
            </AlertDialogTitle>
            <AlertDialogDescription>
              They keep their account, but lose access to this organization and its
              data, and their plan drops back to Free.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={!!deletingId}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={!!deletingId}
              onClick={(e) => {
                e.preventDefault();
                handleRemoveUser();
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deletingId ? "Removing…" : "Remove"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>

  );
}

function Th({ label, sortable }: { label: string; sortable?: boolean }) {
  return (
    <th className="px-5 py-3 font-medium">
      <span className="inline-flex items-center gap-1">
        {label}
        {sortable && <ChevronsUpDown className="h-3 w-3 opacity-60" />}
      </span>
    </th>
  );
}

function StatusBadge({ status }: { status: "Active" | "Inactive" | "Invited" }) {
  const isActive = status === "Active";
  const isPending = status === "Invited";
  const label = isPending ? "Invited" : status;
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        isActive
          ? "bg-emerald-50 text-emerald-700 ring-1 ring-inset ring-emerald-200"
          : isPending
            ? "bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-200"
            : "bg-gray-100 text-gray-700 ring-1 ring-inset ring-gray-200",
      )}
    >
      <span
        className={cn(
          "mr-1.5 h-1.5 w-1.5 rounded-full",
          isActive ? "bg-emerald-500" : isPending ? "bg-amber-500" : "bg-gray-400",
        )}
      />
      {label}
    </span>
  );
}

function Pager({
  page,
  totalPages,
  onChange,
}: {
  page: number;
  totalPages: number;
  onChange: (n: number) => void;
}) {
  const pages: (number | "...")[] = [];
  const first = [1, 2, 3];
  const last = [totalPages - 2, totalPages - 1, totalPages];
  pages.push(...first);
  if (totalPages > 6) pages.push("...");
  pages.push(...last.filter((n) => !first.includes(n)));

  return (
    <div className="flex items-center gap-1">
      {pages.map((p, i) =>
        p === "..." ? (
          <span key={`e${i}`} className="px-2 text-sm text-muted-foreground">
            …
          </span>
        ) : (
          <button
            key={p}
            onClick={() => onChange(p)}
            className={cn(
              "h-8 min-w-8 rounded-md px-2 text-sm font-medium",
              page === p
                ? "bg-[#f4f6fb] text-foreground dark:bg-[#0b0d10]"
                : "text-muted-foreground hover:bg-muted",
            )}
          >
            {p}
          </button>
        ),
      )}
    </div>
  );
}
