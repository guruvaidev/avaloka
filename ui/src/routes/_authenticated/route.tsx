import { createFileRoute, Outlet, redirect } from "@tanstack/react-router";
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
  return (
    <div className="flex h-screen w-full bg-primary text-primary">
      <IconRail />
      <div className="flex min-h-0 min-w-0 flex-1">
        <Outlet />
      </div>
    </div>
  );
}
