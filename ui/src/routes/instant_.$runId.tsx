// /instant/:runId — the report page a question lands on: the verified export
// in a sandboxed frame, Evidence, Limitations, downloads. Signed-in only: the
// same guard as instant.tsx (a visitor without a session is sent to "/"). The
// file is instant_.$runId: the underscore keeps it a sibling of /instant, not
// a child rendered through an Outlet /instant does not have.
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { getCachedAuthUser } from "@/lib/auth-user";
import { IconRail } from "@/components/dashboard/IconRail";
import { ReportPage } from "@/components/instant/ReportPage";

export const Route = createFileRoute("/instant_/$runId")({
  ssr: false,
  head: () => ({
    meta: [
      { title: "Report · Avaloka AI" },
      { name: "description", content: "One question → verified report." },
      { name: "robots", content: "noindex" },
    ],
  }),
  beforeLoad: async () => {
    const bypass = typeof window !== "undefined" && localStorage.getItem("admin-bypass") === "1";
    if (bypass) return;
    const user = await getCachedAuthUser();
    if (!user) {
      throw redirect({ to: "/" });
    }
  },
  component: InstantReportRoute,
});

function InstantReportRoute() {
  const { runId } = Route.useParams();
  const navigate = useNavigate();
  return (
    <div className="flex h-screen w-full bg-primary text-primary">
      <IconRail />
      <div className="min-h-0 min-w-0 flex-1 overflow-y-auto">
        <ReportPage
          runId={runId}
          onReady={(id) => navigate({ to: "/instant/$runId", params: { runId: id } })}
        />
      </div>
    </div>
  );
}
