import type { FC } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  MessageChatCircle,
  Folder,
  File02,
  Database01,
  Star01,
  LifeBuoy01,
  Settings01,
  Users01,
  ChevronDown,
  LogOut01,
  PieChart03,
  ClockRewind,
  BarChart03,
  ArrowLeft,
  Diamond01,
} from "@untitledui/icons";
import { useActivePlan } from "@/lib/use-active-plan";
import { Button as AriaButton } from "react-aria-components";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { Avatar } from "@/components/base/avatar/avatar";
import { Dropdown } from "@/components/base/dropdown/dropdown";
import { Logomark } from "@/components/brand/Logomark";
import { cx } from "@/lib/utils/cx";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { supabase } from "@/integrations/supabase/client";
import { useProjectsPerms, useDashboardPerms } from "@/lib/use-user-mgmt-perms";

import { ReportIcon } from "./icons/ReportIcon";
import { ConfigurationsIcon } from "./icons/ConfigurationsIcon";
import { NotificationsPopover } from "./NotificationsPopover";
import { getCachedAuthUser } from "@/lib/auth-user";
import { SUPPORT_ADMIN_EMAIL } from "@/lib/support-admin";

type RailItem = { id: string; icon: FC<{ className?: string }>; label: string };

const topItems: RailItem[] = [
  { id: "chat", icon: MessageChatCircle, label: "Analysis" },
  { id: "projects", icon: Folder, label: "Projects" },
  { id: "reports", icon: ReportIcon, label: "Reports" },
];

const middleItems: RailItem[] = [{ id: "sources", icon: Database01, label: "Data Set" }];

const analysisProjectItems: RailItem[] = [
  { id: "chat", icon: MessageChatCircle, label: "Analysis" },
  { id: "projects", icon: PieChart03, label: "Dashboard" },
  { id: "sources", icon: Database01, label: "Dataset" },
  { id: "restore", icon: ClockRewind, label: "Restore" },
  { id: "reports", icon: BarChart03, label: "Insights" },
];

const bottomItems: RailItem[] = [
  { id: "settings", icon: ConfigurationsIcon, label: "Configurations" },
  { id: "upgrade", icon: Diamond01, label: "Upgrade" },
];

const FREE_LOCKED_IDS = new Set(["projects", "reports", "sources", "settings"]);

// Default destinations for rail items. Keeps every page's behavior consistent.
const DEFAULT_NAV: Record<string, string> = {
  chat: "/proanalysis",
  projects: "/dashboard",
  files: "/dashboard",
  reports: "/reports",
  sources: "/datasets",
  history: "/scheduled-analysis",
  settings: "/configurations",
  upgrade: "/settings",
  "db-config": "/db-tables",
};

function activeFromPath(pathname: string): string | undefined {
  if (pathname.startsWith("/scheduled-analysis") || pathname.startsWith("/proanalysis/scheduled-analysis")) return "history";
  if (pathname.startsWith("/proanalysis")) return "chat";
  if (pathname.startsWith("/analysis")) return "chat";
  if (pathname.startsWith("/datasets") || pathname.startsWith("/database")) return "sources";
  if (pathname.startsWith("/reports")) return "reports";
  if (pathname.startsWith("/db-tables")) return "db-config";
  if (pathname.startsWith("/configurations")) return "settings";
  if (pathname.startsWith("/settings")) return "settings";
  if (pathname.startsWith("/dashboard")) return "projects";
  return undefined;
}

export function IconRail({
  active,
  onItemClick,
  skipNavIds,
  variant = "default",
}: {
  /** Optional override; otherwise derived from the current route. */
  active?: string;
  /** Optional extra side-effect; default navigation still runs. */
  onItemClick?: (id: string) => void;
  /** Rail item ids that should not trigger route navigation (e.g. in-page panes). */
  skipNavIds?: string[];
  /** `analysis-project` shows the in-analysis project pane icons; `default` is the app rail. */
  variant?: "default" | "analysis-project";
}) {
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const resolvedActive = active ?? activeFromPath(pathname);
  const isAnalysisProject = variant === "analysis-project";
  const projectsPerms = useProjectsPerms();
  const dashboardPerms = useDashboardPerms();
  const visibleAnalysisProjectItems = analysisProjectItems.filter(
    (item) =>
      item.id !== "projects" ||
      dashboardPerms.loading ||
      dashboardPerms.canView,
  );

  const planQ = useActivePlan();
  const tier = planQ.data?.plan ?? "free";
  const isFree = tier === "free";
  const isEnterprise = tier === "enterprise";

  const visibleTopItems = topItems.filter(
    (item) =>
      item.id !== "projects" ||
      projectsPerms.loading ||
      projectsPerms.canView,
  );
  const visibleBottomItems = bottomItems.filter((item) => {
    if (item.id === "upgrade" && isEnterprise) return false;
    if (item.id === "settings" && isFree) return false;
    return true;
  });

  const lockedIds = isFree ? FREE_LOCKED_IDS : new Set<string>();

  const handleClick = (id: string) => {
    // Free users can now OPEN premium sections (the diamond badge still shows);
    // individual actions inside each section prompt to upgrade instead.
    onItemClick?.(id);
    if (skipNavIds?.includes(id)) return;
    const to = DEFAULT_NAV[id];
    if (to && to !== pathname) navigate({ to });
  };


  return (
    <aside className="flex h-full w-[64px] shrink-0 flex-col items-center border-r border-secondary bg-primary py-3">
      {!isAnalysisProject ? (
      <button
        type="button"
        aria-label="Go to analysis"
        onClick={() => {
          if (pathname !== "/proanalysis") navigate({ to: "/proanalysis" });
        }}
        className="mb-4 shrink-0 outline-none"
      >
        <Logomark size={36} />
      </button>
      ) : (
        <ButtonUtility
          size="sm"
          color="tertiary"
          icon={ArrowLeft}
          tooltip="Back to project"
          tooltipPlacement="right"
          onClick={() => {
            if (pathname !== "/dashboard") navigate({ to: "/dashboard" });
          }}
          className="mb-4 size-10 *:data-icon:size-[18px]"
        />
      )}

      <div className="flex min-h-0 w-full flex-1 flex-col items-center">
        {isAnalysisProject ? (
          <RailGroup items={visibleAnalysisProjectItems} active={resolvedActive} onItemClick={handleClick} />

        ) : (
          <>
            <RailGroup items={visibleTopItems} active={resolvedActive} onItemClick={handleClick} lockedIds={lockedIds} />
            <div className="my-3 h-px w-8 shrink-0 bg-border-secondary" />
            <RailGroup items={middleItems} active={resolvedActive} onItemClick={handleClick} lockedIds={lockedIds} />
          </>
        )}
      </div>

      <div className="mt-auto flex w-full shrink-0 flex-col items-center gap-1">
        {!isAnalysisProject ? (
          <>
            <RailGroup items={visibleBottomItems} active={resolvedActive} onItemClick={handleClick} lockedIds={lockedIds} />
            <NotificationsPopover />
          </>
        ) : null}
        <div className={cx("relative", !isAnalysisProject && "mt-2")}>
          <ProfileAvatarMenu />
        </div>
      </div>
    </aside>
  );
}

export function ProfileAvatarMenu() {
  const navigate = useNavigate();

  const profileQ = useQuery({
    queryKey: ["rail", "profile"],
    staleTime: 5 * 60_000,
    gcTime: 30 * 60_000,
    queryFn: async () => {
      const user = await getCachedAuthUser();
      const email = String(user?.email ?? "").trim().toLowerCase();
      const isSupportAdmin = email === SUPPORT_ADMIN_EMAIL;
      if (!user) return { name: null as string | null, avatar: null as string | null, isSupportAdmin: false };

      const { data: profile } = await supabase
        .from("profiles")
        .select("full_name, avatar_url")
        .eq("user_id", user.id)
        .maybeSingle();

      if (!profile) return { name: user.email || "User", avatar: null as string | null, isSupportAdmin };

      let avatar: string | null = null;
      if (profile.avatar_url) {
        if (/^https?:\/\//i.test(profile.avatar_url)) {
          avatar = profile.avatar_url;
        } else {
          const { data: signed } = await supabase.storage
            .from("avatars")
            .createSignedUrl(profile.avatar_url, 60 * 60);
          avatar = signed?.signedUrl ?? null;
        }
      }
      return { name: (profile as any).full_name || user.email || "User", avatar, isSupportAdmin };
    },
  });

  const userName: string | null = profileQ.data?.name ?? null;
  const avatarUrl: string | null = profileQ.data?.avatar ?? null;
  const isSupportAdmin = Boolean((profileQ.data as any)?.isSupportAdmin);

  const initials = userName
    ? userName
        .split(" ")
        .map((n) => n[0])
        .join("")
        .slice(0, 2)
        .toUpperCase()
    : undefined;


  return (
    <Dropdown.Root>
      <AriaButton
        aria-label="Open profile menu"
        className={({ isPressed, isFocusVisible }) =>
          cx(
            "group relative inline-flex cursor-pointer rounded-full outline-offset-2 outline-focus-ring",
            (isPressed || isFocusVisible) && "outline-2",
          )
        }
      >
        <Avatar size="sm" src={avatarUrl} initials={initials} status="online" alt={userName ?? "User"} />
      </AriaButton>

      <Dropdown.Popover className="w-60" placement="right bottom" offset={12}>
        <Dropdown.Menu>
          {!isSupportAdmin && (
            <Dropdown.Item icon={LifeBuoy01} onAction={() => navigate({ to: "/support" })}>
              Support
            </Dropdown.Item>
          )}
          {isSupportAdmin && (
            <Dropdown.Item
              icon={MessageChatCircle}
              onAction={() => navigate({ to: "/support-queries" })}
            >
              Support queries
            </Dropdown.Item>
          )}

          <Dropdown.Item icon={Settings01} onAction={() => navigate({ to: "/settings" })}>
            Settings
          </Dropdown.Item>

          <Dropdown.Separator />

          <Dropdown.Item icon={LogOut01} onAction={() => navigate({ to: "/logout" })}>
            Sign out
          </Dropdown.Item>
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown.Root>
  );
}

function RailGroup({
  items,
  active,
  onItemClick,
  lockedIds,
}: {
  items: RailItem[];
  active?: string;
  onItemClick?: (id: string) => void;
  lockedIds?: Set<string>;
}) {
  return (
    <div className="flex flex-col items-center gap-1">
      {items.map((item) => {
        const isActive = active === item.id;
        const isLocked = lockedIds?.has(item.id) ?? false;
        return (
          <div key={item.id} className="relative">
            <ButtonUtility
              size="sm"
              color="tertiary"
              icon={item.icon}
              tooltip={isLocked ? `${item.label} — Upgrade to unlock` : item.label}
              tooltipPlacement="right"
              onClick={() => onItemClick?.(item.id)}
              className={cx("size-10 *:data-icon:size-[18px]", isActive && "bg-primary_hover text-fg-primary")}
            />
            {isLocked ? (
              <span
                aria-hidden="true"
                className="pointer-events-none absolute -right-0.5 -top-0.5 flex size-3.5 items-center justify-center rounded-full bg-brand-solid text-white ring-2 ring-primary"
              >
                <Diamond01 className="size-2.5" />
              </span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
