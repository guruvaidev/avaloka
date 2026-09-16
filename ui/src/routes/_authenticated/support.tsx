import { createFileRoute, redirect } from "@tanstack/react-router";
import { getCachedAuthUser } from "@/lib/auth-user";
import { SUPPORT_ADMIN_EMAIL } from "@/lib/support-admin";

import { useState } from "react";
import { toast } from "sonner";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { LifeBuoy01, Send01, Paperclip, XClose, Clock, Mail01, Shield01, MessageChatCircle } from "@untitledui/icons";
import { supabase } from "@/integrations/supabase/client";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { sendSupportQueryEmail } from "@/lib/support.functions";
import { listMySupportTickets } from "@/lib/support-tickets.functions";
import { SupportThreadPanel, SupportStatusPill } from "@/components/support/SupportThreadPanel";
import { AnalysisPickerModal } from "@/components/support/AnalysisPickerModal";


export const Route = createFileRoute("/_authenticated/support")({
  // The support desk account has no client-facing support page; it goes
  // straight to the support queries dashboard.
  beforeLoad: async () => {
    const user = await getCachedAuthUser();
    const email = String(user?.email ?? "").trim().toLowerCase();
    if (email === SUPPORT_ADMIN_EMAIL) throw redirect({ to: "/support-queries" });
  },

  head: () => ({
    meta: [
      { title: "Support · Avaloka AI" },
      { name: "description", content: "Send your question to the Avaloka support team." },
      { property: "og:title", content: "Support · Avaloka AI" },
      { property: "og:description", content: "Send your question to the Avaloka support team." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: SupportPage,
});

const CATEGORIES = [
  { id: "general", label: "General question" },
  { id: "billing", label: "Billing & plans" },
  { id: "technical", label: "Technical issue" },
  { id: "feature", label: "Feature request" },
];

const MAX_TOTAL_BYTES = 10 * 1024 * 1024;

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result ?? "");
      resolve(result.slice(result.indexOf(",") + 1));
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

type PickableAnalysis = {
  id: string;
  name: string;
  project_id: string | null;
  project_name: string | null;
};

type PickerData = { analyses: PickableAnalysis[]; projects: { id: string; name: string }[] };

async function fetchAnalysisRows(): Promise<Array<{ id: string; name: string; project_id: string | null }>> {
  // Some environments don't expose every optional column, so fall back to a plain select.
  const base = () => supabase.from("analyses").select("id,name,project_id,created_at").order("created_at", { ascending: false }).limit(300);
  const { data, error } = await base().is("parent_analysis_id", null);
  if (!error) return (data ?? []) as any[];
  const { data: plain, error: plainError } = await base();
  if (plainError) throw plainError;
  return (plain ?? []) as any[];
}

async function listMyAnalyses(): Promise<PickerData> {
  const [rows, projectsRes] = await Promise.all([
    fetchAnalysisRows(),
    supabase.from("projects").select("id,name").is("deleted_at", null).order("name"),
  ]);

  // Include analyses shared through the organization owner view when available.
  let extra: any[] = [];
  try {
    const { listOrgMemberAnalyses } = await import("@/lib/org-owner-access.functions");
    const [standalone, all] = await Promise.all([
      listOrgMemberAnalyses({ data: { projectId: null } }).catch(() => []),
      Promise.resolve([]),
    ]);
    extra = [...((standalone as any[]) ?? []), ...all];
  } catch {
    extra = [];
  }

  const byId = new Map<string, { id: string; name: string; project_id: string | null }>();
  for (const r of [...rows, ...extra]) {
    if (r?.id && !byId.has(r.id)) byId.set(r.id, { id: r.id, name: r.name ?? "Untitled analysis", project_id: r.project_id ?? null });
  }

  const projectRows = (projectsRes.data ?? []) as Array<{ id: string; name: string }>;
  const names = new Map(projectRows.map((p) => [p.id, p.name]));

  const analyses = Array.from(byId.values()).map((r) => ({
    id: r.id,
    name: r.name,
    project_id: r.project_id,
    project_name: r.project_id ? names.get(r.project_id) ?? "Project" : null,
  }));

  return { analyses, projects: projectRows };
}


/** Creates (or reuses) a share link for the analysis so support can open it. */
async function createAnalysisShareLink(analysisId: string): Promise<string | null> {
  const { data } = await supabase.auth.getSession();
  const accessToken = data.session?.access_token;
  if (!accessToken) return null;
  const res = await fetch(`/api/analysis/${analysisId}/share-link`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${accessToken}` },
    body: JSON.stringify({ access_level: "view" }),
  });
  if (!res.ok) return null;
  const json = (await res.json()) as { url?: string };
  return json?.url ?? null;
}

function SupportPage() {
  const [subject, setSubject] = useState("");
  const [category, setCategory] = useState("general");
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [sending, setSending] = useState(false);
  const [analysisIds, setAnalysisIds] = useState<string[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [openTicketId, setOpenTicketId] = useState<string | null>(null);
  const qc = useQueryClient();

  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ["support-tickets", "mine"],
    queryFn: () => listMySupportTickets(),
  });
  const tickets = history?.tickets ?? [];

  const { data: pickerData } = useQuery({
    queryKey: ["support", "my-analyses"],
    queryFn: listMyAnalyses,
    staleTime: 60_000,
  });
  const analysisList = pickerData?.analyses ?? [];



  const addFiles = (list: FileList | null) => {
    if (!list?.length) return;
    const next = [...files, ...Array.from(list)].slice(0, 5);
    const total = next.reduce((n, f) => n + f.size, 0);
    if (total > MAX_TOTAL_BYTES) {
      toast.error("Attachments must be under 10MB in total.");
      return;
    }
    setFiles(next);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedSubject = subject.trim();
    const trimmedMessage = message.trim();
    if (!trimmedSubject || !trimmedMessage) {
      toast.error("Please add a subject and describe your query.");
      return;
    }
    if (trimmedSubject.length > 150) {
      toast.error("Subject must be under 150 characters.");
      return;
    }
    if (trimmedMessage.length > 4000) {
      toast.error("Message must be under 4000 characters.");
      return;
    }

    setSending(true);
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (!user) {
      setSending(false);
      toast.error("You need to be signed in to contact support.");
      return;
    }

    let finalMessage = trimmedMessage;
    if (analysisIds.length) {
      const lines: string[] = [];
      let failed = 0;
      for (const id of analysisIds) {
        const picked = analysisList.find((a) => a.id === id);
        const url = await createAnalysisShareLink(id);
        const label = picked
          ? `${picked.name}${picked.project_name ? ` (${picked.project_name})` : " (Pro analysis)"}`
          : "Analysis";
        if (url) lines.push(`Analysis: ${label}\nLink: ${url}`);
        else failed += 1;
      }
      if (lines.length) finalMessage = `${trimmedMessage}\n\n---\n${lines.join("\n\n")}`;
      if (failed) toast.warning("Some analysis links could not be attached.");
    }

    const { error } = await supabase.from("support_tickets").insert({
      user_id: user.id,
      subject: trimmedSubject,
      category,
      message: finalMessage,
    });

    if (error) {
      setSending(false);
      toast.error("Could not send your query. Please try again.");
      return;
    }

    try {
      const attachments = await Promise.all(
        files.map(async (f) => ({
          filename: f.name,
          contentType: f.type || "application/octet-stream",
          base64: await fileToBase64(f),
        })),
      );
      const res = await sendSupportQueryEmail({
        data: { subject: trimmedSubject, category, message: finalMessage, attachments },
      });

      if (!res?.ok) {
        console.error("support email error", res?.error);
        toast.warning(`Query saved, but email failed: ${res?.error ?? "unknown error"}`);
      } else {
        toast.success("Your query has been sent to support@avaloka.ai.");
      }
    } catch (err) {
      console.error("support email exception", err);
      toast.warning("Query saved, but the notification email could not be delivered.");
    }


    setSending(false);
    setSubject("");
    setMessage("");
    setFiles([]);
    setCategory("general");
    setAnalysisIds([]);

    qc.invalidateQueries({ queryKey: ["support-tickets", "mine"] });
  };




  return (
    <div className="flex min-h-0 min-w-0 flex-1 overflow-hidden bg-background text-foreground">
      <main className="mx-auto flex h-full w-full max-w-7xl flex-col overflow-y-auto px-3 py-3 sm:px-6 sm:py-4 lg:overflow-hidden lg:px-8">
        {/* Top: compact support hero */}
        <section className="relative flex shrink-0 flex-col gap-3 overflow-hidden rounded-2xl border border-border bg-gradient-to-r from-brand-600 to-brand-800 px-4 py-3 text-white shadow-sm sm:flex-row sm:items-center sm:justify-between sm:px-5 sm:py-4">
          <div
            aria-hidden
            className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-white/10 blur-2xl"
          />
          <div className="relative flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/15">
              <LifeBuoy01 className="size-5" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-tight sm:text-lg">How can we help?</h1>
              <p className="text-[11px] text-brand-100 sm:text-xs">
                Our team replies within 24 hours, Mon–Fri.
              </p>
            </div>
          </div>
          <a
            href="mailto:support@avaloka.ai"
            className="relative inline-flex w-fit items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium backdrop-blur transition-colors hover:bg-white/20"
          >
            <Mail01 className="size-3.5" />
            support@avaloka.ai
          </a>
        </section>

        <div className="mt-3 grid min-h-0 flex-1 items-stretch gap-3 sm:mt-4 sm:gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,360px)]">
          {/* Left: form */}
          <form
            onSubmit={submit}
            className="flex min-h-0 flex-col rounded-2xl border border-border bg-card p-3 shadow-sm sm:p-4"
          >
            <div className="flex flex-col gap-3 overflow-y-auto pr-1">

              <div>
                <h2 className="text-base font-semibold tracking-tight text-foreground">Submit a query</h2>
                <p className="text-xs text-muted-foreground">More detail helps us resolve it faster.</p>
              </div>

              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-foreground">What is this about?</span>
                <div className="flex flex-wrap gap-1">
                  {CATEGORIES.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      onClick={() => setCategory(c.id)}
                      aria-pressed={category === c.id}
                      className={cn(
                        "rounded-full border px-2 py-0.5 text-left text-[11px] leading-5 transition-all",
                        "border-border bg-background text-muted-foreground hover:border-brand-600/40 hover:text-foreground",
                        category === c.id &&
                          "border-brand-600 bg-brand-600/10 font-medium text-brand-600 ring-1 ring-brand-600/20",
                      )}
                    >
                      {c.label}
                    </button>
                  ))}
                </div>
              </div>


              <div className="flex flex-col gap-1">
                <label htmlFor="support-subject" className="text-xs font-medium text-foreground">
                  Subject
                </label>
                <Input
                  id="support-subject"
                  value={subject}
                  maxLength={150}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="Short summary"
                  className="h-9 rounded-lg text-sm"
                />
              </div>

              <div className="flex flex-col gap-1">
                <label htmlFor="support-message" className="text-xs font-medium text-foreground">
                  Your query
                </label>
                <textarea
                  id="support-message"
                  value={message}
                  maxLength={4000}
                  onChange={(e) => setMessage(e.target.value)}
                  rows={3}
                  placeholder="Describe your question or issue…"
                  className="w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground focus:border-brand-600 focus:ring-2 focus:ring-brand-600/10"
                />
                <span className="self-end text-[10px] text-muted-foreground">{message.length}/4000</span>
              </div>

              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-foreground">
                  Attach an analysis <span className="text-muted-foreground">(optional)</span>
                </span>
                <button
                  type="button"
                  onClick={() => setPickerOpen(true)}
                  className="flex h-9 w-full items-center justify-between rounded-lg border border-border bg-background px-2.5 text-sm text-muted-foreground outline-none transition-colors hover:border-brand-600/50 hover:text-foreground"
                >
                  <span>{analysisIds.length ? `${analysisIds.length} analysis selected` : "Select analysis"}</span>
                  <span className="text-[10px]">Choose</span>
                </button>
                {analysisIds.length > 0 && (
                  <ul className="flex flex-col gap-1">
                    {analysisIds.map((id) => {
                      const a = analysisList.find((x) => x.id === id);
                      return (
                        <li
                          key={id}
                          className="flex items-center justify-between gap-2 rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs"
                        >
                          <span className="truncate text-foreground">
                            {a ? `${a.project_name ? `${a.project_name} · ` : ""}${a.name}` : "Analysis"}
                          </span>
                          <button
                            type="button"
                            aria-label="Remove analysis"
                            onClick={() => setAnalysisIds((prev) => prev.filter((x) => x !== id))}
                            className="rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                          >
                            <XClose className="size-3" />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
                <span className="text-[10px] text-muted-foreground">
                  A view-only link is added to your query so support can open the exact analysis.
                </span>
              </div>




              <div className="flex flex-col gap-1.5">
                <span className="text-xs font-medium text-foreground">Attachments</span>
                <label
                  htmlFor="support-files"
                  className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-muted/30 px-3 py-2.5 text-xs text-muted-foreground transition-colors hover:border-brand-600/50 hover:bg-brand-600/5 hover:text-foreground"
                >
                  <Paperclip className="size-3.5 text-brand-600" />
                  <span className="font-medium text-foreground">Attach files</span>
                  <span>· 5 files · 10MB</span>
                </label>
                <input
                  id="support-files"
                  type="file"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    addFiles(e.target.files);
                    e.target.value = "";
                  }}
                />
                {files.length > 0 && (
                  <ul className="flex flex-col gap-1">
                    {files.map((f, i) => (
                      <li
                        key={`${f.name}-${i}`}
                        className="flex items-center justify-between gap-2 rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs"
                      >
                        <span className="flex min-w-0 items-center gap-1.5">
                          <Paperclip className="size-3 shrink-0 text-muted-foreground" />
                          <span className="truncate text-foreground" title={f.name}>
                            {f.name}
                          </span>
                        </span>
                        <span className="flex shrink-0 items-center gap-1.5 text-[10px] text-muted-foreground">
                          {formatSize(f.size)}
                          <button
                            type="button"
                            aria-label={`Remove ${f.name}`}
                            onClick={() => setFiles(files.filter((_, idx) => idx !== i))}
                            className="rounded p-0.5 transition-colors hover:bg-muted hover:text-foreground"
                          >
                            <XClose className="size-3" />
                          </button>
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            <div className="mt-auto flex shrink-0 flex-col gap-2 border-t border-border pt-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-[10px] text-muted-foreground">
                Sent to <span className="font-medium text-foreground">support@avaloka.ai</span>{" "}
                with your account details.
              </p>
              <Button
                type="submit"
                disabled={sending}
                className="h-9 gap-1.5 rounded-lg bg-brand-600 px-4 text-sm text-white shadow-sm transition-colors hover:bg-brand-700"
              >
                <Send01 className="size-3.5" />
                {sending ? "Sending…" : "Send query"}
              </Button>
            </div>
          </form>

          {/* Right: previous queries */}
          <section className="flex min-h-[320px] flex-col rounded-2xl border border-border bg-card shadow-sm lg:min-h-0">
            <div className="shrink-0 border-b border-border px-4 py-3">
              <div className="flex items-center gap-2">
                <MessageChatCircle className="size-3.5 text-brand-600" />
                <h2 className="text-sm font-semibold tracking-tight text-foreground">
                  Your support queries
                </h2>
              </div>
              <p className="text-[10px] text-muted-foreground">Open a query to reply.</p>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto">
              {historyLoading && (
                <p className="px-4 py-4 text-xs text-muted-foreground">Loading your queries…</p>
              )}
              {!historyLoading && tickets.length === 0 && (
                <p className="px-4 py-4 text-xs text-muted-foreground">
                  You haven't submitted any support queries yet.
                </p>
              )}
              <ul className="divide-y divide-border">
                {tickets.map((t) => (
                  <li key={t.id}>
                    <button
                      type="button"
                      onClick={() => setOpenTicketId(t.id)}
                      className="flex w-full items-start justify-between gap-2 px-4 py-2.5 text-left transition-colors hover:bg-muted/50"
                    >
                      <span className="min-w-0">
                        <span className="block truncate text-xs font-medium text-foreground">
                          {t.subject}
                        </span>
                        <span className="mt-0.5 block truncate text-[10px] text-muted-foreground">
                          {t.category} · {new Date(t.created_at).toLocaleString()}
                        </span>
                      </span>
                      <SupportStatusPill status={t.status} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        </div>
      </main>


      {openTicketId && (
        <SupportThreadPanel ticketId={openTicketId} onClose={() => setOpenTicketId(null)} />
      )}

      {pickerOpen && (
        <AnalysisPickerModal
          analyses={analysisList}
          projects={pickerData?.projects ?? []}
          selected={analysisIds}
          onClose={() => setPickerOpen(false)}
          onConfirm={(ids) => {
            setAnalysisIds(ids);
            setPickerOpen(false);
          }}
        />
      )}
    </div>
  );
}



