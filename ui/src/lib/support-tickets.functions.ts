import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/primary-auth-middleware";

import { SUPPORT_ADMIN_EMAIL } from "./support-admin";

export type SupportTicket = {
  id: string;
  subject: string;
  category: string;
  message: string;
  status: string;
  created_at: string;
  updated_at: string;
  user_id: string;
  requester_email?: string | null;
  requester_name?: string | null;
  reply_count?: number;
};

export type SupportMessage = {
  id: string;
  ticket_id: string;
  author_id: string;
  author_role: string;
  body: string;
  created_at: string;
};

/** Messages now live inline on support_tickets.support_ticket_messages (jsonb array). */
function parseMessages(raw: unknown, ticketId: string): SupportMessage[] {
  let value = raw;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(value)) return [];
  return value
    .filter((m) => m && typeof m === "object")
    .map((m: any, i: number) => ({
      id: String(m.id ?? `${ticketId}-${i}`),
      ticket_id: ticketId,
      author_id: String(m.author_id ?? ""),
      author_role: String(m.author_role ?? "user"),
      body: String(m.body ?? ""),
      created_at: String(m.created_at ?? new Date(0).toISOString()),
    }))
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
}

function claimsEmail(context: unknown): string {
  return String((context as any)?.claims?.email ?? "").trim().toLowerCase();
}

function assertAdmin(context: unknown) {
  if (claimsEmail(context) !== SUPPORT_ADMIN_EMAIL) {
    throw new Error("Forbidden: support admin access only");
  }
}

/** Tells the client whether the signed-in account is the support admin. */
export const amISupportAdmin = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => ({
    isAdmin: claimsEmail(context) === SUPPORT_ADMIN_EMAIL,
  }));

/** Tickets created by the signed-in user (RLS scoped). */
export const listMySupportTickets = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabase, userId } = context as { supabase: any; userId: string };
    const { data, error } = await supabase
      .from("support_tickets")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    const tickets = ((data ?? []) as any[]).map((t) => ({
      ...(t as SupportTicket),
      reply_count: parseMessages(t.support_ticket_messages, t.id).length,
    }));
    return { tickets };
  });

/** Full conversation for one ticket. Users only get their own (RLS). */
export const getSupportThread = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { ticketId: string }) => ({
    ticketId: String(input?.ticketId ?? ""),
  }))
  .handler(async ({ data, context }) => {
    const isAdmin = claimsEmail(context) === SUPPORT_ADMIN_EMAIL;
    // Admin reads go through the privileged client (verified above); every
    // other caller stays on their RLS-scoped client, so users only ever see
    // their own queries.
    const db = isAdmin
      ? (await import("@/integrations/supabase/primary-client.server")).supabaseAdmin
      : (context as { supabase: any }).supabase;

    const { data: ticket, error } = await db
      .from("support_tickets")
      .select("*")
      .eq("id", data.ticketId)
      .maybeSingle();
    if (error) throw new Error(error.message);
    if (!ticket) throw new Error("Support query not found");

    const messages = parseMessages(
      (ticket as any).support_ticket_messages,
      data.ticketId,
    );

    let requester: { email: string | null; name: string | null } = {
      email: null,
      name: null,
    };
    if (isAdmin) requester = await resolveRequester(db, ticket.user_id);

    return {
      ticket: {
        ...(ticket as SupportTicket),
        requester_email: requester.email,
        requester_name: requester.name,
      },
      messages,
    };
  });

/** Post a reply. Users reply on their own tickets; the support admin on any. */
export const replyToSupportTicket = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { ticketId: string; body: string }) => {
    const ticketId = String(input?.ticketId ?? "");
    const body = String(input?.body ?? "").trim().slice(0, 4000);
    if (!ticketId || !body) throw new Error("A message is required");
    return { ticketId, body };
  })
  .handler(async ({ data, context }) => {
    const { userId } = context as { userId: string };
    const isAdmin = claimsEmail(context) === SUPPORT_ADMIN_EMAIL;
    const db = isAdmin
      ? (await import("@/integrations/supabase/primary-client.server")).supabaseAdmin
      : (context as { supabase: any }).supabase;

    const { data: ticket, error: readErr } = await db
      .from("support_tickets")
      .select("id, support_ticket_messages")
      .eq("id", data.ticketId)
      .maybeSingle();
    if (readErr) throw new Error(readErr.message);
    if (!ticket) throw new Error("Support query not found");

    const existing = parseMessages(
      (ticket as any).support_ticket_messages,
      data.ticketId,
    );
    const message: SupportMessage = {
      id:
        globalThis.crypto?.randomUUID?.() ??
        `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      ticket_id: data.ticketId,
      author_id: userId,
      author_role: isAdmin ? "support" : "user",
      body: data.body,
      created_at: new Date().toISOString(),
    };

    const { error } = await db
      .from("support_tickets")
      .update({
        support_ticket_messages: [...existing, message],
        status: isAdmin ? "answered" : "open",
        updated_at: new Date().toISOString(),
      })
      .eq("id", data.ticketId);
    if (error) throw new Error(error.message);

    return { message };
  });



async function resolveRequester(admin: any, userId: string) {
  let email: string | null = null;
  let name: string | null = null;
  try {
    const { data } = await admin.auth.admin.getUserById(userId);
    email = data?.user?.email ?? null;
  } catch {
    /* ignore */
  }
  try {
    const { data } = await admin
      .from("profiles")
      .select("full_name, first_name, last_name")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .limit(1)
      .maybeSingle();
    if (data) {
      name =
        (data.full_name && String(data.full_name).trim()) ||
        [data.first_name, data.last_name].filter(Boolean).join(" ").trim() ||
        null;
    }
  } catch {
    /* ignore */
  }
  return { email, name };
}

/** Every client ticket — support admin only. */
export const listAllSupportTickets = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    assertAdmin(context);
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );

    const { data, error } = await supabaseAdmin
      .from("support_tickets")
      .select("*")
      .order("updated_at", { ascending: false })
      .limit(500);
    if (error) throw new Error(error.message);

    const tickets = (data ?? []) as (SupportTicket & {
      support_ticket_messages?: unknown;
    })[];

    const uniqueUsers = [...new Set(tickets.map((t) => t.user_id))];
    const info = new Map<string, { email: string | null; name: string | null }>();
    await Promise.all(
      uniqueUsers.map(async (uid) => {
        info.set(uid, await resolveRequester(supabaseAdmin, uid));
      }),
    );

    return {
      tickets: tickets.map(({ support_ticket_messages, ...t }) => ({
        ...t,
        reply_count: parseMessages(support_ticket_messages, t.id).length,
        requester_email: info.get(t.user_id)?.email ?? null,
        requester_name: info.get(t.user_id)?.name ?? null,
      })) as SupportTicket[],
    };

  });

/** Change a ticket's status — support admin only. */
export const setSupportTicketStatus = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { ticketId: string; status: string }) => {
    const ticketId = String(input?.ticketId ?? "");
    const status = String(input?.status ?? "").toLowerCase();
    if (!ticketId) throw new Error("Missing ticket");
    if (!["open", "answered", "closed"].includes(status)) {
      throw new Error("Invalid status");
    }
    return { ticketId, status };
  })
  .handler(async ({ data, context }) => {
    assertAdmin(context);
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const { error } = await supabaseAdmin
      .from("support_tickets")
      .update({ status: data.status, updated_at: new Date().toISOString() })
      .eq("id", data.ticketId);
    if (error) throw new Error(error.message);
    return { ok: true };
  });
