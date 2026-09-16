// GET /api/shared/:token
// Returns the analysis payload for a valid share-link token. Requires the
// caller to be authenticated (via Bearer token), then bypasses RLS to load
// the shared analysis + messages regardless of collaborator membership.

import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/api/shared/$token")({
  server: {
    handlers: {
      GET: async ({ request, params }) => {
        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");

        const { data: link, error: linkErr } = await supabaseAdmin
          .from("resource_share_links")
          .select("resource_type, resource_id, link_access, is_public")
          .eq("share_token", params.token)
          .maybeSingle();
        if (linkErr || !link || (link as any).resource_type !== "analysis") {
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

        const analysisId = (link as any).resource_id as string;
        const accessLevel = (link as any).link_access as string;


        const { data: analysis } = await supabaseAdmin
          .from("analyses")
          .select("id, name, project_id, dataset_id, filename, schema, samples, viz_config, created_at")
          .eq("id", analysisId)
          .maybeSingle();
        if (!analysis) return new Response(JSON.stringify({ error: "Not found" }), { status: 404 });

        const { data: messages } = await supabaseAdmin
          .from("analysis_messages")
          .select("id, role, content, output, author_id, created_at")
          .eq("analysis_id", analysisId)
          .order("created_at", { ascending: true });

        const authorIds = Array.from(
          new Set((messages ?? []).map((m: any) => m.author_id).filter(Boolean)),
        ) as string[];
        let authors: Array<{ id: string; full_name: string | null; avatar_url: string | null }> = [];
        if (authorIds.length) {
          const { data } = await supabaseAdmin
            .from("profiles")
            .select("id, full_name, avatar_url")
            .in("id", authorIds);
          authors = (data ?? []) as any;
        }

        return Response.json({
          access_level: accessLevel,
          analysis,
          messages: messages ?? [],
          authors,
        });
      },
    },
  },
});
