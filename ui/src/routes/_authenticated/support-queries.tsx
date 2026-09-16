import { createFileRoute, redirect } from "@tanstack/react-router";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { LifeBuoy01, SearchLg } from "@untitledui/icons";
import { getCachedAuthUser } from "@/lib/auth-user";
import { listAllSupportTickets } from "@/lib/support-tickets.functions";
import { SUPPORT_ADMIN_EMAIL } from "@/lib/support-admin";
import { SupportThreadPanel, SupportStatusPill } from "@/components/support/SupportThreadPanel";
import { Input } from "@/components/ui/input";

export const Route = createFileRoute("/_authenticated/support-queries")({
  // UI-level gate; the server functions independently verify the caller's
  // email before returning any client data.
  beforeLoad: async () => {
    const user = await getCachedAuthUser();
    const email = String(user?.email ?? "").trim().toLowerCase();
    if (email !== SUPPORT_ADMIN_EMAIL) throw redirect({ to: "/support" });
  },
  head: () => ({
    meta: [
      { title: "Support queries · Avaloka AI" },
      { name: "description", content: "Review and reply to client support queries." },
      { property: "og:title", content: "Support queries · Avaloka AI" },
      { property: "og:description", content: "Review and reply to client support queries." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: AdminSupportQueriesPage,
});

function AdminSupportQueriesPage() {
  const [openTicketId, setOpenTicketId] = useState<string | null>(null);
  const [q, setQ] = useState("");

  const { data, isLoading, error } = useQuery({
    queryKey: ["support-tickets", "all"],
    queryFn: () => listAllSupportTickets(),
    refetchInterval: 60_000,
  });

  const tickets = (data?.tickets ?? []).filter((t) => {
    const needle = q.trim().toLowerCase();
    if (!needle) return true;
    return [t.subject, t.category, t.status, t.requester_email, t.requester_name, t.message]
      .filter(Boolean)
      .some((v) => String(v).toLowerCase().includes(needle));
  });

  return (
    <div className="flex min-h-0 min-w-0 flex-1 overflow-y-auto bg-background text-foreground">
      <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 lg:px-8 lg:py-12">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-brand-600">
              <LifeBuoy01 className="size-5" />
              <span className="text-xs font-medium uppercase tracking-wide">Support desk</span>
            </div>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight text-foreground">
              Client support queries
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Open a query to read the full conversation and reply to the client.
            </p>
          </div>
          <div className="relative w-full max-w-xs">
            <SearchLg className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search queries…"
              className="h-11 rounded-xl pl-9 text-sm"
            />
          </div>
        </div>

        <div className="mt-6 overflow-hidden rounded-2xl border border-border bg-card">
          {isLoading && <p className="px-5 py-6 text-sm text-muted-foreground">Loading queries…</p>}
          {error && (
            <p className="px-5 py-6 text-sm text-destructive">
              Could not load support queries.
            </p>
          )}
          {!isLoading && !error && tickets.length === 0 && (
            <p className="px-5 py-6 text-sm text-muted-foreground">No support queries found.</p>
          )}
          <ul className="divide-y divide-border">
            {tickets.map((t) => (
              <li key={t.id}>
                <button
                  type="button"
                  onClick={() => setOpenTicketId(t.id)}
                  className="flex w-full items-start justify-between gap-4 px-5 py-4 text-left transition-colors hover:bg-muted/50"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium text-foreground">{t.subject}</span>
                    <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                      {t.requester_name || t.requester_email || "Unknown client"} · {t.category} ·{" "}
                      {new Date(t.created_at).toLocaleString()} · {t.reply_count ?? 0} replies
                    </span>
                  </span>
                  <SupportStatusPill status={t.status} isAdmin />
                </button>
              </li>
            ))}
          </ul>
        </div>
      </main>

      {openTicketId && (
        <SupportThreadPanel
          ticketId={openTicketId}
          isAdmin
          onClose={() => setOpenTicketId(null)}
        />
      )}
    </div>
  );
}
