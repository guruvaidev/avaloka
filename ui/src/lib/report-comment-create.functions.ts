import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/primary-auth-middleware";

export const createReportComment = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator(
    z.object({
      reportId: z.string().uuid(),
      parentId: z.string().uuid().nullable(),
      content: z.string().trim().min(1).max(10_000),
      attachments: z.array(
        z.object({
          name: z.string(),
          path: z.string(),
          mime_type: z.string(),
          size: z.number().nonnegative(),
        }),
      ),
    }),
  )
  .handler(async ({ data, context }) => {
    const { data: report, error: reportError } = await context.supabase
      .from("reports")
      .select("id")
      .eq("id", data.reportId)
      .maybeSingle();

    if (reportError) throw reportError;
    if (!report) throw new Error("You do not have access to this report");

    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const { data: profiles, error: profileError } = await supabaseAdmin
      .from("profiles")
      .select("id")
      .or(`user_id.eq.${context.userId},id.eq.${context.userId}`)
      .limit(1);

    if (profileError) throw profileError;
    const authorId = profiles?.[0]?.id;
    if (!authorId) throw new Error("Profile not found for current user");

    if (data.parentId) {
      const { data: parent, error: parentError } = await supabaseAdmin
        .from("report_comments" as never)
        .select("id")
        .eq("id", data.parentId)
        .eq("report_id", data.reportId)
        .maybeSingle();
      if (parentError) throw parentError;
      if (!parent) throw new Error("The parent comment does not belong to this report");
    }

    const { data: comment, error: insertError } = await supabaseAdmin
      .from("report_comments" as never)
      .insert({
        report_id: data.reportId,
        author_id: authorId,
        parent_id: data.parentId,
        body: data.content,
        attachments: data.attachments,
      } as never)
      .select("id,body,created_at,parent_id,attachments,report_id,author_id,author:profiles!author_id(id,full_name,avatar_url)")
      .single();

    if (insertError) throw insertError;
    return comment;
  });