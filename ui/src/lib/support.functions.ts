import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/primary-auth-middleware";

const SUPPORT_INBOX = "support@avaloka.ai";

function esc(v: unknown): string {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export const sendSupportQueryEmail = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    subject: string;
    category: string;
    message: string;
    attachments?: { filename: string; contentType: string; base64: string }[];
  }) => {
    const subject = String(input?.subject ?? "").trim().slice(0, 150);
    const category = String(input?.category ?? "general").trim().slice(0, 50);
    const message = String(input?.message ?? "").trim().slice(0, 4000);
    if (!subject || !message) throw new Error("Subject and message are required");
    const attachments = (input?.attachments ?? []).slice(0, 5).map((a) => ({
      filename: String(a?.filename ?? "attachment").slice(0, 120),
      contentType: String(a?.contentType || "application/octet-stream").slice(0, 100),
      base64: String(a?.base64 ?? ""),
    })).filter((a) => a.base64.length > 0);
    const total = attachments.reduce((n, a) => n + a.base64.length, 0);
    if (total > 14_000_000) throw new Error("Attachments are too large (max ~10MB total)");
    return { subject, category, message, attachments };
  })

  .handler(async ({ data, context }) => {
    const { supabase, userId } = context as { supabase: any; userId: string };

    let profile: Record<string, any> = {};
    try {
      const { data: row } = await supabase.from("profiles").select("*").eq("id", userId).maybeSingle();
      profile = row ?? {};
    } catch {
      profile = {};
    }

    const name = profile["full_name"] ?? profile["name"] ?? profile["display_name"] ?? "—";
    const email = profile["email"] ?? (context as any).claims?.email ?? "—";
    const org = profile["organization_id"] ?? "—";

    const html = `
      <div style="font-family:Inter,Arial,sans-serif;color:#0f172a;line-height:1.6">
        <h2 style="margin:0 0 4px 0">New support query</h2>
        <p style="margin:0 0 16px 0;color:#475569;font-size:13px">Submitted from the Avaloka support page.</p>
        <table style="border-collapse:collapse;font-size:14px;margin-bottom:16px">
          <tr><td style="padding:4px 12px 4px 0;color:#64748b">Name</td><td><b>${esc(name)}</b></td></tr>
          <tr><td style="padding:4px 12px 4px 0;color:#64748b">Email</td><td><b>${esc(email)}</b></td></tr>
          <tr><td style="padding:4px 12px 4px 0;color:#64748b">User ID</td><td>${esc(userId)}</td></tr>
          <tr><td style="padding:4px 12px 4px 0;color:#64748b">Organization</td><td>${esc(org)}</td></tr>
          <tr><td style="padding:4px 12px 4px 0;color:#64748b">Category</td><td>${esc(data.category)}</td></tr>
          <tr><td style="padding:4px 12px 4px 0;color:#64748b">Subject</td><td><b>${esc(data.subject)}</b></td></tr>
        </table>
        <div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px;white-space:pre-wrap;font-size:14px">${esc(data.message)}</div>
      </div>
    `;

    const supabaseUrl = process.env["SUPABASE_URL"];
    const authToken =
      process.env["INTERNAL_MAIL_TOKEN"] || process.env["SUPABASE_SERVICE_ROLE_KEY"];
    if (!supabaseUrl || !authToken) return { ok: false, error: "Email service is not configured." };



    const res = await fetch(`${supabaseUrl}/functions/v1/send-smtp-email`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken}` },
      body: JSON.stringify({
        to: SUPPORT_INBOX,
        subject: `[Support · ${data.category}] ${data.subject}`,
        html,
        attachments: data.attachments,
      }),

    });
    if (!res.ok) {
      const text = await res.text();
      console.error("support email failed", res.status, text.slice(0, 500));
      return { ok: false, error: `${res.status}: ${text.slice(0, 300)}` };
    }
    return { ok: true, error: null };

  });
