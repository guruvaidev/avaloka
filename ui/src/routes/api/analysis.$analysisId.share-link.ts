// POST /api/analysis/:analysisId/share-link
// Creates (or returns) a share link for an analysis. Auth via Bearer token.

import { createFileRoute } from "@tanstack/react-router";

async function verifyOwner(analysisId: string, accessToken: string) {
  const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
  const { data: userData, error: userErr } = await supabaseAdmin.auth.getUser(accessToken);
  if (userErr || !userData.user) return { error: "Unauthorized", status: 401 as const };
  const authUserId = userData.user.id;

  const { data: profile } = await supabaseAdmin
    .from("profiles")
    .select("id")
    .eq("user_id", authUserId)
    .maybeSingle();
  const profileId = (profile as any)?.id as string | undefined;

  const { data: analysis, error: aErr } = await supabaseAdmin
    .from("analyses")
    .select("id, owner_id")
    .eq("id", analysisId)
    .maybeSingle();
  if (aErr || !analysis) return { error: "Analysis not found", status: 404 as const };
  const ownerId = (analysis as any).owner_id as string | null;
  if (ownerId && profileId && ownerId !== profileId && ownerId !== authUserId) {
    return { error: "Forbidden", status: 403 as const };
  }
  return { supabaseAdmin };
}

export const Route = createFileRoute("/api/analysis/$analysisId/share-link")({
  server: {
    handlers: {
      POST: async ({ request, params }) => {
        const auth = request.headers.get("authorization") ?? "";
        const token = auth.replace(/^Bearer\s+/i, "");
        if (!token) return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });

        let body: { access_level?: string } = {};
        try {
          body = await request.json();
        } catch {}
        const accessLevel = body.access_level === "edit" ? "edit" : "view";

        const check = await verifyOwner(params.analysisId, token);
        if ("error" in check) {
          return new Response(JSON.stringify({ error: check.error }), { status: check.status });
        }
        const { supabaseAdmin } = check;

        const { data: existing } = await supabaseAdmin
          .from("resource_share_links")
          .select("share_token, link_access")
          .eq("resource_type", "analysis")
          .eq("resource_id", params.analysisId)
          .maybeSingle();

        let row: { share_token: string; link_access: string } | null =
          (existing as any) ?? null;

        if (row) {
          if (row.link_access !== accessLevel) {
            const { data: updated, error: uErr } = await supabaseAdmin
              .from("resource_share_links")
              .update({ link_access: accessLevel, updated_at: new Date().toISOString() })
              .eq("resource_type", "analysis")
              .eq("resource_id", params.analysisId)
              .select("share_token, link_access")
              .single();
            if (uErr) return new Response(JSON.stringify({ error: uErr.message }), { status: 500 });
            row = updated as any;
          }
        } else {
          const { data: inserted, error: iErr } = await supabaseAdmin
            .from("resource_share_links")
            .insert({
              resource_type: "analysis",
              resource_id: params.analysisId,
              link_access: accessLevel,
            } as any)
            .select("share_token, link_access")
            .single();
          if (iErr) return new Response(JSON.stringify({ error: iErr.message }), { status: 500 });
          row = inserted as any;
        }

        const origin = new URL(request.url).origin;
        const url = `${origin}/shared/${row!.share_token}`;
        return Response.json({ url, token: row!.share_token, access_level: row!.link_access });
      },
    },
  },
});
