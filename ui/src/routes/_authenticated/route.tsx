import { createFileRoute, Outlet, redirect } from "@tanstack/react-router";
import { getCachedAuthUser } from "@/lib/auth-user";
import { IconRail } from "@/components/dashboard/IconRail";

export const Route = createFileRoute("/_authenticated")({
  ssr: false,
  beforeLoad: async () => {
    const bypass =
      typeof window !== "undefined" &&
      localStorage.getItem("admin-bypass") === "1";
    if (bypass) return;

    // Uses the cached/local session instead of a network round-trip on every
    // navigation (falls back to a real getUser() when nothing is cached).
    const user = await getCachedAuthUser();
    if (!user) {
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
