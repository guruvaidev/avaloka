// /instant — Instant Analysis: upload or pick a sample, see the data profile,
// ask one question. Signed-in only: the same guard as _authenticated/route.tsx
// (a visitor without a session is sent to the sign-in page at "/").
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { getCachedAuthUser } from "@/lib/auth-user";
import { IconRail } from "@/components/dashboard/IconRail";
import { InstantFlow } from "@/components/instant/InstantFlow";

export const Route = createFileRoute("/instant")({
  ssr: false,
  head: () => ({
    meta: [
      { title: "Ask · Avaloka AI" },
      { name: "description", content: "One question → verified report." },
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
  component: InstantPage,
});

function InstantPage() {
  const navigate = useNavigate();
  return (
    <div className="flex h-screen w-full bg-primary text-primary">
      <IconRail />
      <div className="min-h-0 min-w-0 flex-1 overflow-y-auto">
        <InstantFlow
          onReady={(r) => navigate({ to: "/instant/$runId", params: { runId: r.run_id } })}
        />
      </div>
    </div>
  );
}
