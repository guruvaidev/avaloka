// DELETE /api/analysis/:analysisId/share-link/:token
// Revokes a share link. Auth via Bearer token.

import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/api/analysis/$analysisId/share-link/$token")({
  server: {
    handlers: {
      DELETE: async ({ request, params }) => {
        const auth = request.headers.get("authorization") ?? "";
        const accessToken = auth.replace(/^Bearer\s+/i, "");
        if (!accessToken) return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });

        const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
        const { data: userData, error: userErr } = await supabaseAdmin.auth.getUser(accessToken);
        if (userErr || !userData.user) {
          return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
        }
        const authUserId = userData.user.id;

        const { data: profile } = await supabaseAdmin
          .from("profiles")
          .select("id")
          .eq("user_id", authUserId)
          .maybeSingle();
        const profileId = (profile as any)?.id as string | undefined;

        const { data: analysis } = await supabaseAdmin
          .from("analyses")
          .select("id, owner_id")
          .eq("id", params.analysisId)
          .maybeSingle();
        if (!analysis) return new Response(JSON.stringify({ error: "Not found" }), { status: 404 });
        const ownerId = (analysis as any).owner_id as string | null;
        if (ownerId && profileId && ownerId !== profileId && ownerId !== authUserId) {
          return new Response(JSON.stringify({ error: "Forbidden" }), { status: 403 });
        }

        const { error: delErr } = await supabaseAdmin
          .from("resource_share_links")
          .delete()
          .eq("resource_type", "analysis")
          .eq("resource_id", params.analysisId)
          .eq("share_token", params.token);
        if (delErr) return new Response(JSON.stringify({ error: delErr.message }), { status: 500 });

        return Response.json({ ok: true });
      },
    },
  },
});
