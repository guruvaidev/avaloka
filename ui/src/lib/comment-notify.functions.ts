// Server-side fan-out for analysis comment notifications.
// Uses the admin client to bypass RLS quirks (e.g. collaborators whose
// profiles/app_users rows would otherwise be invisible to the commenter).

import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

export const fanOutCommentNotifications = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator(
    z.object({
      commentId: z.string().uuid(),
      mentionedProfileIds: z.array(z.string().uuid()).optional(),
    }),
  )
  .handler(async ({ data, context }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");

    // Load the comment (admin, bypasses RLS).
    const { data: comment, error: commentErr } = await supabaseAdmin
      .from("analysis_comments" as never)
      .select("id, analysis_id, author_id, parent_id, content")
      .eq("id", data.commentId)
      .maybeSingle();
    if (commentErr || !comment) return { notified: 0 };

    const c = comment as unknown as {
      analysis_id: string;
      author_id: string;
      parent_id: string | null;
      content: string;
    };

    // Authorize: caller must be the author (identified via profiles.id or profiles.user_id).
    const authUserId = context.userId;
    const { data: profs } = await supabaseAdmin
      .from("profiles")
      .select("id,user_id")
      .or(`id.eq.${authUserId},user_id.eq.${authUserId}`);
    const callerProfileIds = new Set(((profs ?? []) as { id: string }[]).map((p) => p.id));
    if (!callerProfileIds.has(c.author_id)) {
      throw new Error("Forbidden");
    }

    // Author name.
    const { data: authorProf } = await supabaseAdmin
      .from("profiles")
      .select("full_name")
      .eq("id", c.author_id)
      .maybeSingle();
    const authorName =
      (authorProf as { full_name?: string | null } | null)?.full_name?.trim() || "Someone";

    // Analysis info.
    const { data: analysis } = await supabaseAdmin
      .from("analyses")
      .select("owner_id,name,project_id")
      .eq("id", c.analysis_id)
      .maybeSingle();
    const a = (analysis ?? null) as {
      owner_id: string | null;
      name: string | null;
      project_id: string | null;
    } | null;

    const recipients = new Set<string>();
    if (a?.owner_id) recipients.add(a.owner_id);

    // Helper: resolve resource_collaborators app_user_ids -> profiles.id list.
    async function collectCollabProfileIds(
      resourceType: "analysis" | "project",
      resourceId: string,
    ): Promise<string[]> {
      const { data: rows } = await supabaseAdmin
        .from("resource_collaborators")
        .select("app_user_id")
        .eq("resource_type", resourceType)
        .eq("resource_id", resourceId);
      const appUserIds = ((rows ?? []) as { app_user_id: string }[])
        .map((r) => r.app_user_id)
        .filter(Boolean);
      if (!appUserIds.length) return [];

      const { data: appUsers } = await supabaseAdmin
        .from("app_users")
        .select("auth_user_id")
        .in("id", appUserIds);
      const authIds = ((appUsers ?? []) as { auth_user_id: string | null }[])
        .map((u) => u.auth_user_id)
        .filter((v): v is string => !!v);
      if (!authIds.length) return [];

      const [{ data: linked }, { data: legacy }] = await Promise.all([
        supabaseAdmin.from("profiles").select("id").in("user_id", authIds),
        supabaseAdmin.from("profiles").select("id").in("id", authIds),
      ]);
      return [
        ...((linked ?? []) as { id: string }[]),
        ...((legacy ?? []) as { id: string }[]),
      ].map((p) => p.id);
    }

    for (const id of await collectCollabProfileIds("analysis", c.analysis_id)) {
      recipients.add(id);
    }

    if (a?.project_id) {
      const { data: project } = await supabaseAdmin
        .from("projects")
        .select("owner_id")
        .eq("id", a.project_id)
        .maybeSingle();
      const p = project as { owner_id?: string | null } | null;
      if (p?.owner_id) recipients.add(p.owner_id);
      for (const id of await collectCollabProfileIds("project", a.project_id)) {
        recipients.add(id);
      }
    }

    const preview =
      c.content.length > 120 ? c.content.slice(0, 117) + "…" : c.content;
    const analysisName = a?.name ?? "your analysis";
    const link = `/analysis?id=${c.analysis_id}`;

    // @mentions → dedicated notification, gated by the "Mentions & Tags" pref.
    const mentioned = (data.mentionedProfileIds ?? []).filter((id) => id !== c.author_id);
    const { filterRecipientsByPreference } = await import("@/lib/notify-prefs.server");
    const mentionTargets = await filterRecipientsByPreference(supabaseAdmin, mentioned, "mention");
    if (mentionTargets.length) {
      await supabaseAdmin.from("notifications").insert(
        mentionTargets.map((rid) => ({
          recipient_id: rid,
          actor_id: c.author_id,
          type: "comment.mention",
          title: `${authorName} mentioned you in "${analysisName}"`,
          body: preview,
          resource_type: "analysis",
          resource_id: c.analysis_id,
          link,
        })) as never,
      );
    }
    for (const id of mentioned) recipients.delete(id);

    // Reply → notify parent's author specifically.
    if (c.parent_id) {
      const { data: parent } = await supabaseAdmin
        .from("analysis_comments" as never)
        .select("author_id")
        .eq("id", c.parent_id)
        .maybeSingle();
      const parentAuthorId =
        (parent as { author_id?: string } | null)?.author_id ?? null;
      if (parentAuthorId && parentAuthorId !== c.author_id) {
        await supabaseAdmin.from("notifications").insert({
          recipient_id: parentAuthorId,
          actor_id: c.author_id,
          type: "comment.reply",
          title: `${authorName} replied to your comment`,
          body: preview,
          resource_type: "analysis",
          resource_id: c.analysis_id,
          link,
        } as never);
        recipients.delete(parentAuthorId);
      }
    }

    recipients.delete(c.author_id);
    const list = await filterRecipientsByPreference(supabaseAdmin, [...recipients], "comment");
    if (!list.length) return { notified: 0 };

    const rows = list.map((rid) => ({
      recipient_id: rid,
      actor_id: c.author_id,
      type: c.parent_id ? "comment.reply" : "comment.created",
      title: c.parent_id
        ? `${authorName} replied on "${analysisName}"`
        : `${authorName} commented on "${analysisName}"`,
      body: preview,
      resource_type: "analysis",
      resource_id: c.analysis_id,
      link,
    }));
    const { error: insertErr } = await supabaseAdmin
      .from("notifications")
      .insert(rows as never);
    if (insertErr) throw insertErr;
    return { notified: rows.length };
  });
