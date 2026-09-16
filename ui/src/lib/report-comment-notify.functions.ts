// Server-side fan-out for report comment notifications.
// Notifies report creator + all collaborators (report/analysis/project) and
// the project owner, excluding the comment author. Replies also ping the
// parent comment's author.

import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

export const fanOutReportCommentNotifications = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator(z.object({ commentId: z.string().uuid(), mentionedProfileIds: z.array(z.string().uuid()).optional() }))
  .handler(async ({ data, context }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");

    const { data: comment, error: commentErr } = await supabaseAdmin
      .from("report_comments" as never)
      .select("id, report_id, author_id, parent_id, body")
      .eq("id", data.commentId)
      .maybeSingle();
    if (commentErr || !comment) return { notified: 0 };
    const c = comment as unknown as {
      report_id: string;
      author_id: string;
      parent_id: string | null;
      body: string;
    };

    // Authorize: caller must be the author.
    const authUserId = context.userId;
    const { data: profs } = await supabaseAdmin
      .from("profiles")
      .select("id,user_id")
      .or(`id.eq.${authUserId},user_id.eq.${authUserId}`);
    const callerProfileIds = new Set(((profs ?? []) as { id: string }[]).map((p) => p.id));
    if (!callerProfileIds.has(c.author_id)) throw new Error("Forbidden");

    const { data: authorProf } = await supabaseAdmin
      .from("profiles")
      .select("full_name")
      .eq("id", c.author_id)
      .maybeSingle();
    const authorName =
      (authorProf as { full_name?: string | null } | null)?.full_name?.trim() || "Someone";

    // Load report.
    const { data: report } = await supabaseAdmin
      .from("reports")
      .select("id,title,analysis_id,project_id,created_by")
      .eq("id", c.report_id)
      .maybeSingle();
    const r = (report ?? null) as {
      id: string;
      title: string | null;
      analysis_id: string | null;
      project_id: string | null;
      created_by: string | null;
    } | null;

    const recipients = new Set<string>();

    // Report creator (created_by may be auth uid OR profile id — try both).
    if (r?.created_by) {
      const { data: creatorProfiles } = await supabaseAdmin
        .from("profiles")
        .select("id")
        .or(`id.eq.${r.created_by},user_id.eq.${r.created_by}`);
      for (const p of ((creatorProfiles ?? []) as { id: string }[])) recipients.add(p.id);
    }

    async function collectCollabProfileIds(
      resourceType: "analysis" | "project" | "report",
      resourceId: string,
    ): Promise<string[]> {
      const { data: rows } = await supabaseAdmin
        .from("resource_collaborators")
        .select("app_user_id")
        .eq("resource_type", resourceType)
        .eq("resource_id", resourceId);
      const appUserIds = ((rows ?? []) as { app_user_id: string }[])
        .map((x) => x.app_user_id)
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

    for (const id of await collectCollabProfileIds("report", c.report_id)) recipients.add(id);

    if (r?.analysis_id) {
      const { data: analysis } = await supabaseAdmin
        .from("analyses")
        .select("owner_id")
        .eq("id", r.analysis_id)
        .maybeSingle();
      const aOwner = (analysis as { owner_id?: string | null } | null)?.owner_id;
      if (aOwner) recipients.add(aOwner);
      for (const id of await collectCollabProfileIds("analysis", r.analysis_id)) recipients.add(id);
    }

    if (r?.project_id) {
      const { data: project } = await supabaseAdmin
        .from("projects")
        .select("owner_id")
        .eq("id", r.project_id)
        .maybeSingle();
      const pOwner = (project as { owner_id?: string | null } | null)?.owner_id;
      if (pOwner) recipients.add(pOwner);
      for (const id of await collectCollabProfileIds("project", r.project_id)) recipients.add(id);
    }

    const preview = c.body.length > 120 ? c.body.slice(0, 117) + "…" : c.body;
    const reportName = r?.title ?? "your report";
    const link = `/reports/${c.report_id}`;

    const mentioned = (data.mentionedProfileIds ?? []).filter((id) => id !== c.author_id);
    const { filterRecipientsByPreference } = await import("@/lib/notify-prefs.server");
    const mentionTargets = await filterRecipientsByPreference(supabaseAdmin, mentioned, "mention");
    if (mentionTargets.length) {
      await supabaseAdmin.from("notifications").insert(
        mentionTargets.map((rid) => ({
          recipient_id: rid,
          actor_id: c.author_id,
          type: "comment.mention",
          title: `${authorName} mentioned you in "${reportName}"`,
          body: preview,
          resource_type: "report",
          resource_id: c.report_id,
          link,
        })) as never,
      );
    }
    for (const id of mentioned) recipients.delete(id);

    if (c.parent_id) {
      const { data: parent } = await supabaseAdmin
        .from("report_comments" as never)
        .select("author_id")
        .eq("id", c.parent_id)
        .maybeSingle();
      const parentAuthorId = (parent as { author_id?: string } | null)?.author_id ?? null;
      if (parentAuthorId && parentAuthorId !== c.author_id) {
        await supabaseAdmin.from("notifications").insert({
          recipient_id: parentAuthorId,
          actor_id: c.author_id,
          type: "comment.reply",
          title: `${authorName} replied to your comment`,
          body: preview,
          resource_type: "report",
          resource_id: c.report_id,
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
        ? `${authorName} replied on "${reportName}"`
        : `${authorName} commented on "${reportName}"`,
      body: preview,
      resource_type: "report",
      resource_id: c.report_id,
      link,
    }));
    const { error: insertErr } = await supabaseAdmin
      .from("notifications")
      .insert(rows as never);
    if (insertErr) throw insertErr;
    return { notified: rows.length };
  });
