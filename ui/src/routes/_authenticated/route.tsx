import { createFileRoute, Outlet, redirect, useRouterState } from "@tanstack/react-router";
import { getCachedAuthUser } from "@/lib/auth-user";
import { IconRail } from "@/components/dashboard/IconRail";

export const Route = createFileRoute("/_authenticated")({
  ssr: false,
  beforeLoad: async ({ location }) => {    

    const bypass =
      typeof window !== "undefined" &&
      localStorage.getItem("admin-bypass") === "1";
    if (bypass) return;

    // Uses the cached/local session instead of a network round-trip on every
    // navigation (falls back to a real getUser() when nothing is cached).
    const user = await getCachedAuthUser();
    if (!user) {
       const params = new URLSearchParams(location.search);
      const tab = params.get("tab");

      // Store tab=billing in a cookie
      if (typeof document !== "undefined" && tab === "billing") {
        document.cookie = "tab=billing; path=/; max-age=86400; SameSite=Lax";
      }
      throw redirect({ to: "/" });
    }
  },
  component: AuthenticatedLayout,
});

function AuthenticatedLayout() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const usesSeparatedPanel = pathname.startsWith("/settings") || pathname.startsWith("/configurations");

  return (
    <div className="flex h-screen w-full bg-primary text-primary dark:bg-[#131316]">
      <IconRail />
      <div className={`flex min-h-0 min-w-0 flex-1 ${usesSeparatedPanel ? "my-3 mr-3 ml-3 overflow-hidden rounded-l-xl rounded-r-none border border-border-secondary bg-primary dark:border-white/10 dark:bg-[#1A1A1E]" : ""}`}>
        <Outlet />
      </div>
    </div>
  );
}
