// GET /api/shared-report/:token
// Returns a report payload for a valid share-link token. If the link is
// marked public, allows anonymous reads; otherwise requires a Bearer token.

import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/api/public/shared-report/$token")({
  server: {
    handlers: {
      GET: async ({ request, params }) => {
        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");

        const { data: link, error: linkErr } = await supabaseAdmin
          .from("resource_share_links")
          .select("resource_type, resource_id, link_access, is_public")
          .eq("share_token", params.token)
          .maybeSingle();
        if (linkErr || !link || (link as any).resource_type !== "report") {
          return new Response(JSON.stringify({ error: "Not found" }), { status: 404 });
        }
        const isPublic = Boolean((link as any).is_public);

        if (!isPublic) {
          const auth = request.headers.get("authorization") ?? "";
          const accessToken = auth.replace(/^Bearer\s+/i, "");
          if (!accessToken) {
            return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
          }
          const { data: userData, error: userErr } = await supabaseAdmin.auth.getUser(accessToken);
          if (userErr || !userData.user) {
            return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
          }
        }

        const reportId = (link as any).resource_id as string;

        const { data: report, error: reportErr } = await supabaseAdmin
          .from("reports")
          .select("id, title, key_insights, analysis_id, project_id, created_at")
          .eq("id", reportId)
          .maybeSingle();
        if (reportErr) {
          return new Response(JSON.stringify({ error: reportErr.message }), { status: 500 });
        }
        if (!report) return new Response(JSON.stringify({ error: "Not found" }), { status: 404 });

        let analysis: any = null;
        if ((report as any).analysis_id) {
          const { data } = await supabaseAdmin
            .from("analyses")
            .select("id, name, viz_config, samples, filename")
            .eq("id", (report as any).analysis_id)
            .maybeSingle();
          analysis = data;
        }

        const { data: commentRows } = await supabaseAdmin
          .from("report_comments")
          .select(
            "id, body, created_at, parent_id, attachments, author_id, author:profiles!author_id(id, full_name, avatar_url)",
          )
          .eq("report_id", reportId)
          .order("created_at", { ascending: true });

        const rawRows = (commentRows ?? []) as any[];
        const idSet = new Set(rawRows.map((r) => r.id));
        const childrenBy = new Map<string, any[]>();
        const roots: any[] = [];
        for (const r of rawRows) {
          if (r.parent_id && idSet.has(r.parent_id)) {
            const arr = childrenBy.get(r.parent_id) ?? [];
            arr.push(r);
            childrenBy.set(r.parent_id, arr);
          } else {
            roots.push(r);
          }
        }
        const orderedRows: any[] = [];
        for (const root of roots) {
          orderedRows.push(root);
          for (const k of childrenBy.get(root.id) ?? []) orderedRows.push(k);
        }

        const comments = orderedRows.map((r: any) => ({
          id: r.id,
          text: r.body,
          created_at: r.created_at,
          parent_id: r.parent_id && idSet.has(r.parent_id) ? r.parent_id : null,
          attachments: Array.isArray(r.attachments) ? r.attachments : [],
          author: Array.isArray(r.author) ? r.author[0] ?? null : r.author,
        }));

        return Response.json({
          access_level: (link as any).link_access as string,
          is_public: isPublic,
          report,
          analysis,
          comments,
        });
      },
    },
  },
});
