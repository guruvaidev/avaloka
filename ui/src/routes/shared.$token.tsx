// Read-only viewer for shared analyses via /shared/:token.
// Reuses existing AnalysisOutputTable + DynamicChart components.

import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, useMemo } from "react";
import { supabase } from "@/integrations/supabase/client";
import { AnalysisOutputTable } from "@/components/dashboard/AnalysisOutputTable";
import { DynamicChart, normalizeVizConfig } from "@/components/dashboard/dynamicChart";
import { deriveVizFromRows } from "@/lib/derive-viz";
import { Button } from "@/components/base/buttons/button";
import { toast } from "sonner";

export const Route = createFileRoute("/shared/$token")({
  head: () => ({
    meta: [
      { title: "Shared analysis · Avaloka" },
      { name: "robots", content: "noindex" },
    ],
  }),
  component: SharedAnalysisPage,
});

type SharedPayload = {
  access_level: "view" | "edit";
  analysis: {
    id: string;
    name: string;
    filename: string | null;
    samples: unknown;
    viz_config: unknown;
    created_at: string;
  };
  messages: Array<{
    id: string;
    role: "user" | "assistant";
    content: string | null;
    output: unknown;
    created_at: string;
  }>;
};

function SharedAnalysisPage() {
  const { token } = Route.useParams();
  const [status, setStatus] = useState<"loading" | "unauth" | "ok" | "error">("loading");
  const [payload, setPayload] = useState<SharedPayload | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [email, setEmail] = useState("");
  const [sending, setSending] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const consumeHashSession = async () => {
      if (typeof window === "undefined") return;
      const hash = window.location.hash;
      if (hash && hash.includes("access_token=")) {
        const params = new URLSearchParams(hash.replace(/^#/, ""));
        const access_token = params.get("access_token");
        const refresh_token = params.get("refresh_token");
        if (access_token && refresh_token) {
          await supabase.auth.setSession({ access_token, refresh_token });
          history.replaceState(null, "", window.location.pathname + window.location.search);
        }
      }
    };

    const load = async () => {
      await consumeHashSession();
      const { data: sess } = await supabase.auth.getSession();
      const accessToken = sess.session?.access_token;
      try {
        const res = await fetch(`/api/shared/${token}`, {
          headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
        });
        if (res.status === 401 || res.status === 403) {
          if (!cancelled) setStatus("unauth");
          return;
        }
        if (res.status === 404) {
          if (!cancelled) {
            setStatus("error");
            setErrorMsg("This link is no longer available.");
          }
          return;
        }
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const data = (await res.json()) as SharedPayload;
        if (!cancelled) {
          setPayload(data);
          setStatus("ok");
        }
      } catch (err) {
        if (!cancelled) {
          setStatus("error");
          setErrorMsg(err instanceof Error ? err.message : "Failed to load shared analysis");
        }
      }
    };


    void load();

    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => {
      if (session?.access_token) void load();
    });

    return () => {
      cancelled = true;
      sub.subscription.unsubscribe();
    };
  }, [token]);

  const sendMagicLink = async () => {
    if (!email.trim()) return;
    setSending(true);
    try {
      const { error } = await supabase.auth.signInWithOtp({
        email: email.trim().toLowerCase(),
        options: { emailRedirectTo: window.location.href },
      });
      if (error) throw error;
      toast.success("Check your email for the sign-in link");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to send link");
    } finally {
      setSending(false);
    }
  };

  const samples = useMemo(() => {
    const s = payload?.analysis.samples;
    return Array.isArray(s) ? (s as any[]) : [];
  }, [payload]);

  // User Insights: viz_config from assistant messages (not the analysis-level auto viz_config).
  const userSlides = useMemo(() => {
    if (!payload) return [];
    const out: any[] = [];
    const seen = new Set<string>();
    for (const m of payload.messages) {
      if (m.role !== "assistant") continue;
      const o = m.output as { viz_config?: unknown; output_json?: unknown[] } | null;
      if (!o) continue;
      let viz = o.viz_config;
      if (!viz && Array.isArray(o.output_json) && o.output_json.length > 0) {
        viz = deriveVizFromRows(o.output_json);
      }
      if (!viz) continue;
      const rowSamples = Array.isArray(o.output_json) ? (o.output_json as any[]) : samples;
      try {
        const slides = normalizeVizConfig(viz as any, rowSamples);
        for (const s of slides) {
          const key = JSON.stringify({ t: s.title, x: (s as any).xKey, ty: (s as any).type });
          if (seen.has(key)) continue;
          seen.add(key);
          out.push(s);
        }
      } catch {
        /* ignore */
      }
    }
    return out;
  }, [payload, samples]);

  // Analysis output rows: the latest assistant message's output_json (the query result table).
  const outputRows = useMemo(() => {
    if (!payload) return [];
    for (let i = payload.messages.length - 1; i >= 0; i--) {
      const m = payload.messages[i]!;
      if (m.role !== "assistant") continue;
      const o = m.output as { output_json?: unknown[] } | null;
      const rows = o?.output_json;
      if (Array.isArray(rows) && rows.length > 0) {
        return rows.filter((r) => r && typeof r === "object") as Record<string, unknown>[];
      }
    }
    return [];
  }, [payload]);

  if (status === "loading") {
    return <div className="grid min-h-screen place-items-center bg-primary text-sm text-tertiary">Loading…</div>;
  }

  if (status === "unauth") {
    return (
      <div className="grid min-h-screen place-items-center bg-primary px-6">
        <div className="w-full max-w-sm rounded-2xl border border-secondary bg-primary p-6 shadow-sm">
          <h1 className="text-lg font-semibold text-primary">Sign in to view shared analysis</h1>
          <p className="mt-1 text-sm text-tertiary">
            Anyone with this link can sign in and view these results.
          </p>
          <form
            className="mt-4 space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              void sendMagicLink();
            }}
          >
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary shadow-xs placeholder:text-tertiary focus:border-[#1565ef] focus:outline-none"
            />
            <Button type="submit" color="primary" size="md" isDisabled={sending} isLoading={sending}>
              {sending ? "Sending…" : "Send magic link"}
            </Button>
          </form>
        </div>
      </div>
    );
  }

  if (status === "error" || !payload) {
    return (
      <div className="grid min-h-screen place-items-center bg-primary px-6">
        <div className="max-w-md text-center">
          <h1 className="text-lg font-semibold text-primary">Unavailable</h1>
          <p className="mt-2 text-sm text-tertiary">{errorMsg}</p>
        </div>
      </div>
    );
  }




  return (
    <div className="min-h-screen bg-primary text-primary">
      <header className="border-b border-secondary bg-primary px-6 py-4">
        <div className="mx-auto max-w-6xl">
          <p className="text-xs uppercase tracking-wide text-tertiary">Shared analysis · Read-only</p>
          <h1 className="mt-1 text-xl font-semibold text-primary">{payload.analysis.name}</h1>
          {payload.analysis.filename ? (
            <p className="mt-0.5 text-sm text-tertiary">{payload.analysis.filename}</p>
          ) : null}
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-6 py-6">
        {userSlides.length > 0 ? (
          <section className="rounded-xl border border-secondary bg-primary p-4">
            <h2 className="mb-3 text-sm font-semibold text-primary">Chart</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              {userSlides.map((slide: any, i: number) => (
                <div key={i} className="rounded-lg border border-secondary bg-primary p-3">
                  <p className="mb-2 text-sm font-medium text-primary">{slide.title}</p>
                  <DynamicChart slide={slide} height={240} />
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {outputRows.length > 0 ? (
          <section className="rounded-xl border border-secondary bg-primary p-4">
            <h2 className="mb-3 text-sm font-semibold text-primary">Data table</h2>
            <AnalysisOutputTable rows={outputRows} title="Result" pageSize={10} />
          </section>
        ) : null}

        <section className="space-y-3">
          <h2 className="text-sm font-semibold text-primary">Conversation</h2>
          {payload.messages.length === 0 ? (
            <p className="text-sm text-tertiary">No messages yet.</p>
          ) : (
            payload.messages.map((m) => {
              const outputRows = Array.isArray(m.output)
                ? (m.output as any[]).filter((r) => r && typeof r === "object")
                : null;
              return (
                <div
                  key={m.id}
                  className={
                    m.role === "user"
                      ? "rounded-lg border border-secondary bg-secondary/30 p-3"
                      : "rounded-lg border border-secondary bg-primary p-3"
                  }
                >
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-tertiary">
                    {m.role === "user" ? "You" : "Assistant"}
                  </p>
                  {m.content ? (
                    <p className="whitespace-pre-wrap text-sm text-primary">{m.content}</p>
                  ) : null}
                  {outputRows && outputRows.length > 0 ? (
                    <div className="mt-3">
                      <AnalysisOutputTable rows={outputRows} title="Result" pageSize={8} />
                    </div>
                  ) : null}
                </div>
              );
            })
          )}
        </section>
      </main>
    </div>
  );
}
