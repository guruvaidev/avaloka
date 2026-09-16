import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/primary-auth-middleware";
import { safeRedirectOrigin } from "@/lib/auth-redirect";


// ---- Invitation delivery over SMTP -------------------------------------
// Auth's built-in mailer is bypassed: we mint the invite action link with the
// admin API and deliver it ourselves through the configured SMTP relay.
async function sendInviteEmailViaSmtp(opts: {
  admin: any;
  email: string;
  redirectTo?: string;
  data?: Record<string, unknown>;
  /** Existing platform accounts cannot be "invited" again — send a magic link. */
  existingUser?: boolean;
}): Promise<{ userId: string | null; error: string | null }> {
  const redirectTo = safeRedirectOrigin(opts.redirectTo);
  const buildLink = (type: "invite" | "magiclink") =>
    opts.admin.auth.admin.generateLink({
      type,
      email: opts.email,
      options: { redirectTo, data: opts.data },
    });


  let link = await buildLink(opts.existingUser ? "magiclink" : "invite");
  if (link.error && /already.*registered|already.*exists|exists/i.test(link.error.message ?? "")) {
    link = await buildLink("magiclink");
  }
  if (link.error) {
    return { userId: null, error: link.error.message || "Failed to create invitation" };
  }
  const actionLink: string =
    link.data?.properties?.action_link ?? link.data?.action_link ?? "";
  const userId: string | null = link.data?.user?.id ?? null;


  // The mail function is deployed on the Cloud-managed project; authenticate
  // with the shared internal token (falls back to that project's service key).
  const supabaseUrl = process.env["SUPABASE_URL"];
  const serviceKey =
    process.env["INTERNAL_MAIL_TOKEN"] || process.env["SUPABASE_SERVICE_ROLE_KEY"];
  if (!supabaseUrl || !serviceKey) {
    return { userId, error: "Email service is not configured." };
  }



  const html = `
    <div style="font-family:Inter,Arial,sans-serif;color:#0f172a;line-height:1.6">
      <h2 style="margin:0 0 8px 0">You've been invited to Avaloka</h2>
      <p style="margin:0 0 16px 0;color:#475569;font-size:14px">
        You have been invited to collaborate. Click the button below to accept the
        invitation and set up your account.
      </p>
      <p style="margin:24px 0">
        <a href="${actionLink}" style="background:#1565ef;color:#ffffff;text-decoration:none;padding:10px 18px;border-radius:8px;font-weight:600;font-size:14px">Accept invitation</a>
      </p>
      <p style="margin:16px 0;color:#64748b;font-size:12px;word-break:break-all">
        Or paste this link into your browser:<br/>${actionLink}
      </p>
    </div>
  `;

  const res = await fetch(`${supabaseUrl}/functions/v1/send-smtp-email`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${serviceKey}` },
    body: JSON.stringify({
      to: opts.email,
      subject: "You've been invited to Avaloka",
      html,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    return { userId, error: `Failed to send invitation email: ${text}` };
  }
  return { userId, error: null };
}


// Fetch the signed-in user's organization_id via SECURITY DEFINER RPC.
async function getOrgId(supabase: any, userId: string): Promise<string> {
  const { data: rpcOrg, error: rpcErr } = await supabase.rpc("current_org_id");
  if (!rpcErr && rpcOrg) return rpcOrg as string;

  // Fallback: read directly (tries both possible PK columns).
  const { data } = await supabase
    .from("profiles")
    .select("organization_id")
    .or(`id.eq.${userId},user_id.eq.${userId}`)
    .maybeSingle();
  if (data?.organization_id) return data.organization_id as string;

  throw new Error("No organization found for current user. Please sign out and back in.");
}

// Same as getOrgId but returns null instead of throwing (used by list reads).
async function getOrgIdSoft(supabase: any, userId: string): Promise<string | null> {
  try {
    return await getOrgId(supabase, userId);
  } catch {
    return null;
  }
}


async function getProfileIdForAuthUser(supabase: any, userId: string): Promise<string | null> {
  const { data: byUserId, error: byUserIdError } = await supabase
    .from("profiles")
    .select("id")
    .eq("user_id", userId)
    .maybeSingle();
  if (!byUserIdError && byUserId?.id) return byUserId.id as string;

  const { data: byId, error: byIdError } = await supabase
    .from("profiles")
    .select("id")
    .eq("id", userId)
    .maybeSingle();
  if (!byIdError && byId?.id) return byId.id as string;

  return null;
}

// Resolve profiles.id for an auth user, creating a minimal profile row if
// missing. Handles both schema variants (profiles.id = auth uid, or
// profiles.user_id = auth uid with a separate profiles.id).
async function ensureProfileForAuthUser(
  supabase: any,
  authUserId: string,
  opts: { organizationId?: string | null; fullName?: string | null } = {},
): Promise<string> {
  const existing = await getProfileIdForAuthUser(supabase, authUserId);
  if (existing) {
    if (opts.organizationId || opts.fullName) {
      const patch: any = { updated_at: new Date().toISOString() };
      if (opts.organizationId) patch.organization_id = opts.organizationId;
      if (opts.fullName) patch.full_name = opts.fullName;
      await supabase.from("profiles").update(patch).eq("id", existing);
    }
    return existing;
  }

  // Try inserting with user_id (modern schema); fall back to id (legacy schema).
  const now = new Date().toISOString();
  const base: any = { updated_at: now };
  if (opts.organizationId) base.organization_id = opts.organizationId;
  if (opts.fullName) base.full_name = opts.fullName;

  const withUserId = await supabase
    .from("profiles")
    .insert({ ...base, user_id: authUserId })
    .select("id")
    .maybeSingle();
  if (!withUserId.error && withUserId.data?.id) return withUserId.data.id as string;

  const withId = await supabase
    .from("profiles")
    .insert({ ...base, id: authUserId })
    .select("id")
    .maybeSingle();
  if (!withId.error && withId.data?.id) return withId.data.id as string;

  throw new Error(
    `Failed to create profile for auth user: ${withUserId.error?.message ?? withId.error?.message ?? "unknown error"}`,
  );
}



// Build a map of lowercase email -> avatar_url. First tries to resolve via
// app_users.auth_user_id -> profiles.user_id. For rows where auth_user_id is
// missing (e.g. invited but not yet linked), falls back to looking up the
// auth user by email via the admin API, then joining to profiles. The admin
// client is required because profiles RLS restricts SELECT to the owner.
async function buildAvatarByEmail(supabase: any): Promise<Map<string, string>> {
  const map = new Map<string, string>();
  try {
    const { data: users } = await supabase
      .from("app_users")
      .select("email,auth_user_id");
    const rows = users ?? [];
    if (!rows.length) return map;

    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const admin = supabaseAdmin as any;

    const knownIds = rows.map((u: any) => u.auth_user_id).filter(Boolean);
    const emailToAuthId = new Map<string, string>();

    const missing = rows.filter((u: any) => !u.auth_user_id && u.email);
    for (const u of missing) {
      try {
        const { data } = await admin.auth.admin.listUsers({
          page: 1,
          perPage: 1,
          email: String(u.email).toLowerCase(),
        } as any);
        const found = data?.users?.[0];
        if (found?.id) emailToAuthId.set(String(u.email).toLowerCase(), found.id);
      } catch {
        /* ignore */
      }
    }
    const allIds = Array.from(
      new Set([...knownIds, ...Array.from(emailToAuthId.values())]),
    );
    if (!allIds.length) return map;

    const { data: profs } = await admin
      .from("profiles")
      .select("id,avatar_url")
      .in("id", allIds);
    const profMap = new Map<string, string>();
    for (const p of profs ?? []) {
      if (p.avatar_url) profMap.set(p.id, p.avatar_url);
    }

    for (const u of rows) {
      const email = u.email ? String(u.email).toLowerCase() : null;
      if (!email) continue;
      const authId = u.auth_user_id ?? emailToAuthId.get(email);
      const av = authId ? profMap.get(authId) : null;
      if (av) map.set(email, av);
    }
  } catch (e) {
    console.error("[buildAvatarByEmail] failed", e);
  }
  return map;
}

function pickAvatar(map: Map<string, string>, email?: string | null): string | null {
  if (!email) return null;
  return map.get(String(email).toLowerCase()) ?? null;
}

async function clearGeneratedOrgForStandaloneProfile(
  admin: any,
  userId: string,
  fullName?: string | null,
) {
  const now = new Date().toISOString();
  const byUserId = await admin
    .from("profiles")
    .select("id,user_id,organization_id,full_name")
    .eq("user_id", userId)
    .maybeSingle();

  const byId = byUserId.data
    ? { data: null, error: null }
    : await admin
        .from("profiles")
        .select("id,user_id,organization_id,full_name")
        .eq("id", userId)
        .maybeSingle();

  const profile = byUserId.data ?? byId.data ?? null;
  const error = byUserId.error ?? byId.error ?? null;

  if (error) {
    console.error("Failed to read signed-in profile", error);
    return;
  }

  const name = String(fullName ?? "").trim();
  if (!profile) {
    const payload: any = {
      id: userId,
      user_id: userId,
      organization_id: null,
      updated_at: now,
    };
    if (name) payload.full_name = name;
    const { error: insertErr } = await admin.from("profiles").insert(payload);
    if (insertErr) console.error("Failed to create standalone profile", insertErr);
    return;
  }

  // Enterprise access is now derived from an active subscription joined to plans.
  let hasEnterpriseOrg = false;
  if (profile.organization_id) {
    const { data: sub } = await admin
      .from("subscriptions")
      .select("status, plan:plans(plan_type)")
      .eq("organization_id", profile.organization_id)
      .in("status", ["trialing", "active", "past_due"])
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    hasEnterpriseOrg =
      Boolean(sub) && (sub as any)?.plan?.plan_type === "enterprise";
  }
  if (hasEnterpriseOrg) return;

  const payload: any = { organization_id: null, updated_at: now };
  if (!profile.user_id) payload.user_id = userId;
  if (!profile.full_name && name) payload.full_name = name;

  const shouldUpdate =
    Boolean(profile.organization_id) ||
    !profile.user_id ||
    (!profile.full_name && Boolean(name));
  if (!shouldUpdate) return;

  const { error: updateErr } = await admin
    .from("profiles")
    .update(payload)
    .eq("id", profile.id);
  if (updateErr) console.error("Failed to clear generated organization_id", updateErr);
}

// Translate a selected auth user's id into a profiles.id, creating the profile
// row when the user has not signed in yet. Shared by departments + branches.
async function resolveHeadProfileId(
  authUserId: string | null,
  orgId: string | null,
): Promise<string | null> {
  if (!authUserId) return null;
  try {
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const admin = supabaseAdmin as any;

    let profileRow: any = null;
    const byId = await admin
      .from("profiles")
      .select("id")
      .eq("id", authUserId)
      .maybeSingle();
    if (byId.data) profileRow = byId.data;
    if (!profileRow) {
      const byUserId = await admin
        .from("profiles")
        .select("id")
        .eq("user_id", authUserId)
        .maybeSingle();
      if (!byUserId.error && byUserId.data) profileRow = byUserId.data;
    }

    if (!profileRow) {
      const { data: u } = await admin.auth.admin.getUserById(authUserId);
      if (!u?.user?.id) {
        throw new Error("Selected head is not a registered user yet.");
      }
      const { data: inserted, error: profErr } = await admin
        .from("profiles")
        .upsert(
          { id: authUserId, user_id: authUserId, organization_id: orgId },
          { onConflict: "id" },
        )
        .select("id")
        .maybeSingle();
      if (profErr) throw new Error(profErr.message);
      profileRow = inserted;
    }
    return profileRow?.id ?? authUserId;
  } catch (e) {
    throw new Error(
      e instanceof Error ? e.message : "Failed to link head profile.",
    );
  }
}

// ---------- Departments ----------

// Enrich department rows with head_name/head_email/head_avatar derived from
// public.profiles (name + avatar) and auth.users (email), joined via
// departments.head_profile_id -> profiles.id (which equals auth.users.id).
async function enrichDepartmentHeads(rows: any[]): Promise<any[]> {
  if (!rows?.length) return rows ?? [];
  const ids = Array.from(
    new Set(rows.map((r) => r.head_profile_id).filter(Boolean)),
  ) as string[];
  if (!ids.length) {
    return rows.map((r) => ({
      ...r,
      head_name: null,
      head_email: null,
      head_avatar: null,
    }));
  }

  const headById = new Map<string, { name: string | null; email: string | null; avatar: string | null }>();
  try {
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const admin = supabaseAdmin as any;

    const { data: profs } = await admin
      .from("profiles")
      .select("id,full_name,avatar_url")
      .in("id", ids);
    for (const p of profs ?? []) {
      headById.set(p.id, {
        name: p.full_name ?? null,
        avatar: p.avatar_url ?? null,
        email: null,
      });
    }

    // Fetch emails from auth.users via admin API.
    for (const id of ids) {
      try {
        const { data: u } = await admin.auth.admin.getUserById(id);
        const email = u?.user?.email ?? null;
        const existing = headById.get(id) ?? { name: null, avatar: null, email: null };
        headById.set(id, { ...existing, email });
      } catch {
        /* ignore */
      }
    }
  } catch (e) {
    console.error("[enrichDepartmentHeads] failed", e);
  }

  return rows.map((r) => {
    const h = r.head_profile_id ? headById.get(r.head_profile_id) : null;
    return {
      ...r,
      head_name: h?.name ?? null,
      head_email: h?.email ?? null,
      head_avatar: h?.avatar ?? null,
    };
  });
}

/** Live headcount per department / branch, derived from app_users. */
async function countAppUsersBy(
  supabase: any,
  field: "department" | "branch",
): Promise<Map<string, number>> {
  const counts = new Map<string, number>();
  try {
    const { data, error } = await supabase.from("app_users").select(field);
    if (error) throw new Error(error.message);
    for (const row of data ?? []) {
      const key = String((row as any)[field] ?? "").trim().toLowerCase();
      if (!key) continue;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  } catch (e) {
    console.error("[countAppUsersBy] failed", e);
  }
  return counts;
}

export const listDepartments = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const orgId = await getOrgIdSoft(context.supabase, context.userId);
    if (!orgId) return [];
    const { data, error } = await context.supabase
      .from("departments")
      .select("*")
      .eq("organization_id", orgId)
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    const counts = await countAppUsersBy(context.supabase, "department");
    const rows = await enrichDepartmentHeads(data ?? []);
    return rows.map((r: any) => ({
      ...r,
      employees: counts.get(String(r.department ?? "").trim().toLowerCase()) ?? 0,
    }));
  });



export const upsertDepartment = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    id?: string;
    department: string;
    code: string;
    head_profile_id?: string | null;
    employees?: number;
  }) => input)
  .handler(async ({ data, context }) => {
    const { id, ...rest } = data;
    const orgId = await getOrgId(context.supabase, context.userId);

    const headProfileId = await resolveHeadProfileId(rest.head_profile_id ?? null, orgId);

    const payload = {
      ...rest,
      head_profile_id: headProfileId,
      organization_id: orgId,
    } as any;

    const q = id
      ? context.supabase.from("departments").update(payload).eq("id", id).select().single()
      : context.supabase.from("departments").insert(payload).select().single();
    const { data: row, error } = await q;
    if (error) throw new Error(error.message);
    return row;
  });

export const deleteDepartment = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { id: string }) => input)
  .handler(async ({ data, context }) => {
    const { error } = await context.supabase.from("departments").delete().eq("id", data.id);
    if (error) throw new Error(error.message);
    return { ok: true };
  });

export const bulkUpsertDepartments = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    rows: Array<{
      department: string;
      code: string;
      head_email?: string | null;
      status?: "Active" | "Inactive";
    }>;
  }) => input)
  .handler(async ({ data, context }) => {
    if (!data.rows?.length) return { inserted: 0 };
    const orgId = await getOrgId(context.supabase, context.userId);

    // Resolve head_email -> head_profile_id via auth.users admin API
    // (profiles.id == auth.users.id).
    const emails = Array.from(
      new Set(
        data.rows
          .map((r) => (r.head_email ? String(r.head_email).toLowerCase() : null))
          .filter(Boolean) as string[],
      ),
    );
    const profileByEmail = new Map<string, string>();
    if (emails.length) {
      try {
        const { supabaseAdmin } = await import(
          "@/integrations/supabase/primary-client.server"
        );
        const admin = supabaseAdmin as any;
        for (const email of emails) {
          try {
            const { data: ld } = await admin.auth.admin.listUsers({
              page: 1,
              perPage: 1,
              email,
            } as any);
            const found = ld?.users?.[0];
            if (found?.id) profileByEmail.set(email, found.id);
          } catch {
            /* ignore */
          }
        }
      } catch (e) {
        console.error("[bulkUpsertDepartments] email lookup failed", e);
      }
    }

    const payload = data.rows.map((r) => ({
      department: r.department,
      code: r.code,
      head_profile_id: r.head_email
        ? profileByEmail.get(String(r.head_email).toLowerCase()) ?? null
        : null,
      organization_id: orgId,
    })) as any;
    const { data: rows, error } = await context.supabase
      .from("departments")
      .insert(payload)
      .select("id");
    if (error) throw new Error(error.message);
    return { inserted: rows?.length ?? 0 };
  });


// ---------- Roles (config_roles) ----------
export const listConfigRoles = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const orgId = await getOrgIdSoft(context.supabase, context.userId);
    if (!orgId) return [];
    const { data, error } = await context.supabase
      .from("config_roles")
      .select("*")
      .eq("organization_id", orgId)
      .order("created_at", { ascending: false });

    if (error) throw new Error(error.message);
    return data ?? [];
  });

export const upsertConfigRole = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    id?: string;
    role: string;
    department: string;
    permissions: string[];
  }) => input)
  .handler(async ({ data, context }) => {
    const { id, ...rest } = data;
    const orgId = await getOrgId(context.supabase, context.userId);
    const payload = { ...rest, organization_id: orgId };
    const q = id
      ? context.supabase.from("config_roles").update(payload).eq("id", id).select().single()
      : context.supabase.from("config_roles").insert(payload).select().single();
    const { data: row, error } = await q;
    if (error) throw new Error(error.message);
    return row;
  });

export const deleteConfigRole = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { id: string }) => input)
  .handler(async ({ data, context }) => {
    const { error } = await context.supabase.from("config_roles").delete().eq("id", data.id);
    if (error) throw new Error(error.message);
    return { ok: true };
  });

export const bulkUpsertConfigRoles = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    rows: Array<{
      role: string;
      department: string;
      permissions?: string[];
    }>;
  }) => input)
  .handler(async ({ data, context }) => {
    if (!data.rows?.length) return { inserted: 0 };
    const orgId = await getOrgId(context.supabase, context.userId);
    const payload = data.rows.map((r) => ({
      role: r.role,
      department: r.department,
      permissions: r.permissions ?? [],
      organization_id: orgId,
    }));
    const { data: rows, error } = await context.supabase
      .from("config_roles")
      .insert(payload)
      .select("id");
    if (error) throw new Error(error.message);
    return { inserted: rows?.length ?? 0 };
  });

// ---------- Branches ----------
// branches.head_profile_id -> profiles.id (name + avatar). No email stored.
async function enrichBranchHeads(rows: any[]): Promise<any[]> {
  if (!rows?.length) return rows ?? [];
  const ids = Array.from(
    new Set(rows.map((r) => r.head_profile_id).filter(Boolean)),
  ) as string[];
  const byId = new Map<string, { name: string | null; avatar: string | null }>();
  if (ids.length) {
    try {
      const { supabaseAdmin } = await import(
        "@/integrations/supabase/primary-client.server"
      );
      const { data: profs } = await (supabaseAdmin as any)
        .from("profiles")
        .select("id,full_name,avatar_url")
        .in("id", ids);
      for (const p of profs ?? []) {
        byId.set(p.id, { name: p.full_name ?? null, avatar: p.avatar_url ?? null });
      }
    } catch (e) {
      console.error("[enrichBranchHeads] failed", e);
    }
  }
  return rows.map((r) => {
    const h = r.head_profile_id ? byId.get(r.head_profile_id) : null;
    return { ...r, head_name: h?.name ?? null, head_avatar: h?.avatar ?? null };
  });
}

export const listBranches = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { data, error } = await context.supabase
      .from("branches")
      .select("*")
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    const counts = await countAppUsersBy(context.supabase, "branch");
    const rows = await enrichBranchHeads(data ?? []);
    return rows.map((r: any) => ({
      ...r,
      employees: counts.get(String(r.name ?? "").trim().toLowerCase()) ?? 0,
    }));
  });

export const upsertBranch = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    id?: string;
    name: string;
    location: string;
    head_profile_id?: string | null;
    employees?: number;
  }) => input)
  .handler(async ({ data, context }) => {
    const { id, head_profile_id, ...rest } = data;
    const orgId = await getOrgId(context.supabase, context.userId);
    const headProfileId = await resolveHeadProfileId(head_profile_id ?? null, orgId);
    const payload = { ...rest, head_profile_id: headProfileId, organization_id: orgId } as any;
    const q = id
      ? context.supabase.from("branches").update(payload).eq("id", id).select().single()
      : context.supabase.from("branches").insert(payload).select().single();
    const { data: row, error } = await q;
    if (error) throw new Error(error.message);
    return row;
  });

export const deleteBranch = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { id: string }) => input)
  .handler(async ({ data, context }) => {
    const { error } = await context.supabase.from("branches").delete().eq("id", data.id);
    if (error) throw new Error(error.message);
    return { ok: true };
  });

export const bulkUpsertBranches = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    rows: Array<{
      name: string;
      location: string;
      head_email?: string | null;
      employees?: number;
    }>;
  }) => input)
  .handler(async ({ data, context }) => {
    if (!data.rows?.length) return { inserted: 0 };
    const orgId = await getOrgId(context.supabase, context.userId);

    // Optional head_email in the CSV is resolved to a profile id.
    const emails = Array.from(
      new Set(
        data.rows
          .map((r) => (r.head_email ? String(r.head_email).toLowerCase() : null))
          .filter(Boolean) as string[],
      ),
    );
    const profileByEmail = new Map<string, string>();
    if (emails.length) {
      try {
        const { supabaseAdmin } = await import(
          "@/integrations/supabase/primary-client.server"
        );
        const admin = supabaseAdmin as any;
        for (const email of emails) {
          try {
            const { data: ld } = await admin.auth.admin.listUsers({
              page: 1,
              perPage: 1,
              email,
            } as any);
            const found = ld?.users?.[0];
            if (found?.id) profileByEmail.set(email, found.id);
          } catch {
            /* ignore */
          }
        }
      } catch (e) {
        console.error("[bulkUpsertBranches] email lookup failed", e);
      }
    }

    const payload = data.rows.map((r) => ({
      name: r.name,
      location: r.location,
      head_profile_id: r.head_email
        ? profileByEmail.get(String(r.head_email).toLowerCase()) ?? null
        : null,
      employees: r.employees ?? 0,
      organization_id: orgId,
    })) as any;
    const { data: rows, error } = await context.supabase
      .from("branches")
      .insert(payload)
      .select("id");
    if (error) throw new Error(error.message);
    return { inserted: rows?.length ?? 0 };
  });


// ---------- App Users ----------
export const listAppUsers = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const orgId = await getOrgIdSoft(context.supabase, context.userId);
    if (!orgId) return [];
    const { data, error } = await context.supabase
      .from("app_users")
      .select("*")
      .eq("organization_id", orgId)
      .order("created_at", { ascending: false });

    if (error) throw new Error(error.message);
    const rows = data ?? [];

    // Enrich with full_name/avatar_url/phone/country from profiles (via admin
    // to bypass RLS). Prefer profile_id; fall back to auth_user_id.
    const profileById = new Map<
      string,
      { full_name?: string | null; avatar_url?: string | null; phone?: string | null; country?: string | null }
    >();
    try {
      const { supabaseAdmin } = await import(
        "@/integrations/supabase/primary-client.server"
      );
      const admin = supabaseAdmin as any;
      const ids = Array.from(
        new Set(
          rows
            .map((u: any) => u.profile_id ?? u.auth_user_id)
            .filter(Boolean),
        ),
      ) as string[];
      if (ids.length) {
        const { data: profs } = await admin
          .from("profiles")
          .select("id,full_name,avatar_url,phone,country")
          .in("id", ids);
        for (const p of profs ?? []) profileById.set(p.id, p);
      }
    } catch (e) {
      console.error("[listAppUsers] profile enrich failed", e);
    }

    return rows.map((u: any) => {
      const pid = u.profile_id ?? u.auth_user_id ?? null;
      const p = pid ? profileById.get(pid) : null;
      return {
        ...u,
        name: p?.full_name ?? (u.email ? String(u.email).split("@")[0] : ""),
        avatar: p?.avatar_url ?? null,
        profile_id: pid,
        phone: p?.phone ?? null,
        country: p?.country ?? null,
      };
    });
  });


export const upsertAppUser = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    id?: string;
    name?: string;
    email: string;
    avatar?: string | null;
    user_id_code: string;
    role: string;
    department: string;
    branch: string;
    status?: "Active" | "Inactive" | "Invited";
    permissions?: Record<string, Record<string, boolean>>;
  }) => input)
  .handler(async ({ data, context }) => {
    const { id, name: _name, avatar: _avatar, ...rest } = data;
    const orgId = await getOrgId(context.supabase, context.userId);
    const payload = { ...rest, organization_id: orgId };
    const q = id
      ? context.supabase.from("app_users").update(payload).eq("id", id).select().single()
      : context.supabase.from("app_users").insert(payload).select().single();
    const { data: row, error } = await q;
    if (error) throw new Error(error.message);
    return row;
  });


// Generate a unique user_id_code from a department name. Format: `${DEPT_CODE}-NNN`.
// DEPT_CODE comes from `departments.code` for the org; falls back to a slug of the name.
async function generateUserIdCode(
  supabase: any,
  orgId: string,
  departmentName: string,
): Promise<string> {
  const { data: dept } = await supabase
    .from("departments")
    .select("code")
    .eq("organization_id", orgId)
    .eq("department", departmentName)
    .maybeSingle();
  const raw = (dept?.code || departmentName || "USR").toString();
  const code =
    raw.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 6) || "USR";
  const prefix = `${code}-`;
  const { data: existing } = await supabase
    .from("app_users")
    .select("user_id_code")
    .eq("organization_id", orgId)
    .ilike("user_id_code", `${prefix}%`);
  let max = 0;
  for (const r of existing ?? []) {
    const m = String(r?.user_id_code ?? "").match(/-(\d+)$/);
    if (m) {
      const n = parseInt(m[1], 10);
      if (Number.isFinite(n) && n > max) max = n;
    }
  }
  return `${prefix}${String(max + 1).padStart(3, "0")}`;
}

// Invite a new user via Supabase Admin API — sends a magic-link email and
// inserts the row into app_users tagged with the inviter's organization.
export const inviteAppUser = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    name: string;
    email: string;
    user_id_code?: string;
    role: string;
    department: string;
    branch: string;
    permissions?: Record<string, Record<string, boolean>>;
    redirectTo?: string;
  }) => input)
  .handler(async ({ data, context }) => {
    const email = data.email.trim().toLowerCase();
    if (!email) throw new Error("Email is required");

    const orgId = await getOrgId(context.supabase, context.userId);

    // Guard: already in this org?
    const { data: existingInOrg } = await context.supabase
      .from("app_users")
      .select("id")
      .eq("organization_id", orgId)
      .ilike("email", email)
      .maybeSingle();
    if (existingInOrg) {
      return { ok: false as const, reason: "already_member" as const, message: "This user is already in your organization." };
    }

    // Auto-generate the user_id_code from the selected department.
    const generatedUserIdCode = await generateUserIdCode(
      context.supabase,
      orgId,
      data.department,
    );

    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const admin = supabaseAdmin as any;

    // Check if an auth user already exists for this email.
    // GoTrue's listUsers does not reliably filter by email, so paginate.
    const findAuthUserByEmail = async (): Promise<{ id: string } | null> => {
      try {
        for (let page = 1; page <= 20; page++) {
          const { data: lookup } = await admin.auth.admin.listUsers({
            page,
            perPage: 200,
          } as any);
          const users = lookup?.users ?? [];
          const match = users.find(
            (u: any) => String(u?.email ?? "").toLowerCase() === email,
          );
          if (match?.id) return { id: match.id };
          if (users.length < 200) break;
        }
      } catch {
        /* ignore lookup failure — fall through to invite path */
      }
      return null;
    };
    let existingAuthUser: { id: string } | null = await findAuthUserByEmail();

    const now = new Date().toISOString();

    // Resolve the inviter's profiles.id (FK target). profiles.id may equal
    // the auth uid, or the row may be keyed differently with user_id = auth uid.
    let inviterProfileId: string | null = null;
    {
      const byId = await admin
        .from("profiles")
        .select("id")
        .eq("id", context.userId)
        .maybeSingle();
      if (byId.data?.id) inviterProfileId = byId.data.id;
      if (!inviterProfileId) {
        const byUserId = await admin
          .from("profiles")
          .select("id")
          .eq("user_id", context.userId)
          .maybeSingle();
        if (byUserId.data?.id) inviterProfileId = byUserId.data.id;
      }
    }


    // ---- Existing platform user → email an invitation, stay "Invited" ----
    // The user is NOT added to the organization until they accept the link
    // (acceptAppUserInvite flips the row to Active and sets their org).
    if (existingAuthUser) {
      // Resolve their profile row by user_id first, then id (schema-flexible).
      const { data: prof } = await admin
        .from("profiles")
        .select("id,organization_id,full_name")
        .or(`user_id.eq.${existingAuthUser.id},id.eq.${existingAuthUser.id}`)
        .maybeSingle();

      // Block (and send no email) when they already belong to another
      // organization — either via their profile link or any app_users row.
      const { data: rowElsewhere } = await admin
        .from("app_users")
        .select("id,organization_id,status")
        .or(`auth_user_id.eq.${existingAuthUser.id},email.ilike.${email}`)
        .neq("organization_id", orgId)
        .limit(1)
        .maybeSingle();
      if (rowElsewhere?.id || (prof?.organization_id && prof.organization_id !== orgId)) {
        throw new Error("This user already belongs to another organization.");
      }



      const sentExisting = await sendInviteEmailViaSmtp({
        admin,
        email,
        redirectTo: data.redirectTo,
        existingUser: true,
        data: {
          organization_id: orgId,
          full_name: data.name,
          role: data.role,
          department: data.department,
          branch: data.branch,
        },
      });
      if (sentExisting.error) throw new Error(sentExisting.error);

      const { data: row, error } = await context.supabase
        .from("app_users")
        .insert({
          email,
          user_id_code: generatedUserIdCode,
          role: data.role,
          department: data.department,
          branch: data.branch,
          status: "Invited",
          permissions: data.permissions ?? {},
          organization_id: orgId,
          auth_user_id: existingAuthUser.id,
          invited_by: inviterProfileId,
          invited_at: now,
        } as any)
        .select()
        .single();
      if (error) throw new Error(error.message);
      return { ...row, linked: false };
    }



    // ---- New user → send signup invitation ----
    const sent = await sendInviteEmailViaSmtp({
      admin,
      email,
      redirectTo: data.redirectTo,
      data: {
        organization_id: orgId,
        full_name: data.name,
        role: data.role,
        department: data.department,
        branch: data.branch,
      },
    });
    const invite = {
      error: sent.error ? { message: sent.error } : null,
      data: { user: sent.userId ? { id: sent.userId } : null },
    };

    if (invite.error) {
      const msg = invite.error.message || "Failed to send invitation";
      // Auth user exists but our lookup missed them — retry lookup and link.
      if (/already.*registered|already.*exists|exists/i.test(msg)) {
        const retryUser = await findAuthUserByEmail();
        if (retryUser) {
          const { data: prof } = await admin
            .from("profiles")
            .select("id,organization_id,full_name")
            .or(`user_id.eq.${retryUser.id},id.eq.${retryUser.id}`)
            .maybeSingle();
          if (prof?.organization_id && prof.organization_id !== orgId) {
            throw new Error("This user already belongs to another organization.");
          }
          const retrySent = await sendInviteEmailViaSmtp({
            admin,
            email,
            redirectTo: data.redirectTo,
            existingUser: true,
            data: {
              organization_id: orgId,
              full_name: data.name,
              role: data.role,
              department: data.department,
              branch: data.branch,
            },
          });
          if (retrySent.error) throw new Error(retrySent.error);
          const { data: row, error } = await context.supabase
            .from("app_users")
            .insert({
              email,
              user_id_code: generatedUserIdCode,
              role: data.role,
              department: data.department,
              branch: data.branch,
              status: "Invited",
              permissions: data.permissions ?? {},
              organization_id: orgId,
              auth_user_id: retryUser.id,
              invited_by: inviterProfileId,
              invited_at: now,
            } as any)
            .select()
            .single();
          if (error) throw new Error(error.message);
          return { ...row, linked: false };
        }
        throw new Error("This email has already been invited or registered.");
      }

      throw new Error(msg);
    }

    const authUserId = invite.data.user?.id ?? null;

    const { data: row, error } = await context.supabase
      .from("app_users")
      .insert({
        email,
        user_id_code: generatedUserIdCode,
        role: data.role,
        department: data.department,
        branch: data.branch,
        status: "Invited",
        permissions: data.permissions ?? {},
        organization_id: orgId,
        auth_user_id: authUserId,
        invited_by: inviterProfileId,
        invited_at: now,
      } as any)
      .select()
      .single();

    if (error) throw new Error(error.message);
    return { ...row, linked: false };
  });


// Re-send the invitation email for an app_users row still in "Invited" state.
export const resendAppUserInvite = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { id: string; redirectTo?: string }) => input)
  .handler(async ({ data, context }) => {
    const orgId = await getOrgId(context.supabase, context.userId);

    const { data: row, error: rowErr } = await context.supabase
      .from("app_users")
      .select("id, email, status, role, department, branch, organization_id")
      .eq("id", data.id)
      .maybeSingle();
    if (rowErr) throw new Error(rowErr.message);
    if (!row) throw new Error("User not found.");
    if ((row as any).organization_id !== orgId) {
      throw new Error("This user is not part of your organization.");
    }
    if ((row as any).status !== "Invited") {
      return { ok: false as const, reason: "not_pending" as const, message: "This user has already accepted the invitation." };
    }

    const email = String((row as any).email ?? "").trim().toLowerCase();
    if (!email) throw new Error("This user has no email address.");

    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const admin = supabaseAdmin as any;

    // Does an auth account already exist for this email? If so the resend has
    // to be a magic link — an "invite" link is rejected for existing accounts.
    let existingUserId: string | null = null;
    try {
      for (let page = 1; page <= 20; page++) {
        const { data: lookup } = await admin.auth.admin.listUsers({ page, perPage: 200 } as any);
        const users = lookup?.users ?? [];
        const match = users.find(
          (u: any) => String(u?.email ?? "").toLowerCase() === email,
        );
        if (match) {
          existingUserId = match.id;
          break;
        }
        if (users.length < 200) break;
      }
    } catch {
      /* ignore lookup failures — fall through to re-invite */
    }

    const now = new Date().toISOString();

    // Status stays "Invited" until the invitee actually accepts the link.
    const resent = await sendInviteEmailViaSmtp({
      admin,
      email,
      redirectTo: data.redirectTo,
      existingUser: Boolean(existingUserId),
      data: {
        organization_id: orgId,
        role: (row as any).role,
        department: (row as any).department,
        branch: (row as any).branch,
      },
    });

    if (resent.error) {
      throw new Error(resent.error || "Failed to resend invitation");
    }
    const invite = { data: { user: resent.userId ? { id: resent.userId } : null } };


    await context.supabase
      .from("app_users")
      .update({
        invited_at: now,
        auth_user_id: invite.data?.user?.id ?? (row as any).auth_user_id ?? null,
      } as any)
      .eq("id", data.id);

    return { ok: true as const, activated: false as const, email };
  });



// Called after an invitee accepts the magic link. The auth trigger in the
// external database is not guaranteed to exist, so this makes the app_users
// row active from the authenticated app session as a reliable fallback.
export const acceptAppUserInvite = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );

    let email = String((context.claims as any)?.email ?? "").trim().toLowerCase();
    if (!email) {
      const { data } = await supabaseAdmin.auth.admin.getUserById(context.userId);
      email = data.user?.email?.trim().toLowerCase() ?? "";
    }

    const selectColumns = "id,email,organization_id,status";
    let appUser: any = null;

    const byAuthId = await (supabaseAdmin as any)
      .from("app_users")
      .select(selectColumns)
      .eq("auth_user_id", context.userId)
      .maybeSingle();
    if (!byAuthId.error && byAuthId.data) appUser = byAuthId.data;

    if (!appUser && email) {
      const byEmail = await (supabaseAdmin as any)
        .from("app_users")
        .select(selectColumns)
        .ilike("email", email)
        .maybeSingle();
      if (byEmail.error) throw new Error(byEmail.error.message);
      appUser = byEmail.data;
    }

    if (!appUser) {
      await clearGeneratedOrgForStandaloneProfile(
        supabaseAdmin as any,
        context.userId,
      );
      return { ok: true, updated: false };
    }

    const now = new Date().toISOString();

    // Read the invitee's name from their profile (auth metadata fallback).
    let inviteeName: string | null = null;
    try {
        const { data: prof } = await (supabaseAdmin as any)
          .from("profiles")
          .select("full_name")
          .or(`user_id.eq.${context.userId},id.eq.${context.userId}`)
          .maybeSingle();
      inviteeName = prof?.full_name ?? null;
      if (!inviteeName) {
        const { data: u } = await supabaseAdmin.auth.admin.getUserById(context.userId);
        inviteeName =
          (u?.user?.user_metadata as any)?.full_name ??
          (u?.user?.user_metadata as any)?.name ??
          null;
      }
    } catch { /* ignore */ }

    if (appUser.organization_id) {
      const profilePayload: any = {
        user_id: context.userId,
        organization_id: appUser.organization_id,
        updated_at: now,
      };
      if (inviteeName) profilePayload.full_name = inviteeName;

      const profileByUserId = await (supabaseAdmin as any)
        .from("profiles")
        .upsert(profilePayload, { onConflict: "user_id" });

      if (profileByUserId.error) {
        console.error("Failed to sync invited user profile", profileByUserId.error);
      }
    } else {
      await clearGeneratedOrgForStandaloneProfile(
        supabaseAdmin as any,
        context.userId,
        inviteeName,
      );
    }

    const profileId = await getProfileIdForAuthUser(supabaseAdmin as any, context.userId);

    const fullPayload: Record<string, any> = {
      status: "Active",
      auth_user_id: context.userId,
      profile_id: profileId,
      accepted_at: now,
      updated_at: now,
    };

    let { error } = await (supabaseAdmin as any)
      .from("app_users")
      .update(fullPayload)
      .eq("id", appUser.id);

    // If the external DB is missing newer columns (profile_id / accepted_at),
    // retry with only the fields that are guaranteed to exist so status still
    // flips to Active. Surface the schema issue in the logs.
    if (error && /column .* does not exist|schema cache|Could not find/i.test(error.message)) {
      console.error("acceptAppUserInvite: schema mismatch, retrying with minimal payload:", error.message);
      const minimal = {
        status: "Active",
        auth_user_id: context.userId,
        updated_at: now,
      };
      const retry = await (supabaseAdmin as any)
        .from("app_users")
        .update(minimal)
        .eq("id", appUser.id);
      error = retry.error;
    }

    if (error) throw new Error(error.message);
    return { ok: true, updated: true };
  });

export const deleteAppUser = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { id: string }) => input)
  .handler(async ({ data, context }) => {
    const orgId = await getOrgId(context.supabase, context.userId);

    const { supabaseAdmin } = await import(
      "@/integrations/supabase/client.server"
    );

    const { data: row, error: loadErr } = await (supabaseAdmin as any)
      .from("app_users")
      .select("id, email, role, organization_id, profile_id, auth_user_id")
      .eq("id", data.id)
      .maybeSingle();
    if (loadErr) throw new Error(loadErr.message);
    if (!row) throw new Error("User not found.");
    if (row.organization_id !== orgId) {
      throw new Error("This user does not belong to your organization.");
    }
    if (String(row.role ?? "").toLowerCase() === "owner") {
      throw new Error("The organization owner cannot be removed.");
    }
    if (row.auth_user_id && row.auth_user_id === context.userId) {
      throw new Error("You cannot remove yourself from the organization.");
    }

    // Also guard against removing the org owner via organizations.owner_profile_id.
    const { data: org } = await (supabaseAdmin as any)
      .from("organizations")
      .select("owner_profile_id")
      .eq("id", orgId)
      .maybeSingle();
    if (
      org?.owner_profile_id &&
      (org.owner_profile_id === row.profile_id ||
        org.owner_profile_id === row.auth_user_id)
    ) {
      throw new Error("The organization owner cannot be removed.");
    }

    const { error: delErr } = await (supabaseAdmin as any)
      .from("app_users")
      .delete()
      .eq("id", row.id);
    if (delErr) throw new Error(delErr.message);

    // Detach the person's profile from the organization → falls back to Free.
    const now = new Date().toISOString();
    const targets = [row.profile_id, row.auth_user_id].filter(Boolean) as string[];
    for (const t of targets) {
      const { error: profErr } = await (supabaseAdmin as any)
        .from("profiles")
        .update({ organization_id: null, updated_at: now })
        .or(`id.eq.${t},user_id.eq.${t}`)
        .eq("organization_id", orgId);
      if (profErr) {
        console.error("[deleteAppUser] failed to clear profile org", profErr.message);
      }
    }

    return { ok: true, email: row.email ?? null };
  });


// ---------- Teams ----------
// Normalized model: organization_teams(lead_user_id -> app_users.id),
// app_users(team_id -> organization_teams.id). Lead and member info are
// fetched via joins to app_users and profiles — never stored on the team row.
async function enrichAppUsers(
  supabase: any,
  users: any[],
): Promise<Map<string, { id: string; email: string | null; name: string; avatar: string | null; role: string | null }>> {
  const map = new Map<string, any>();
  if (!users?.length) return map;
  const profileIds = Array.from(
    new Set(users.map((u: any) => u.profile_id ?? u.auth_user_id).filter(Boolean)),
  ) as string[];
  const profById = new Map<string, { full_name: string | null; avatar_url: string | null }>();
  if (profileIds.length) {
    try {
      const { supabaseAdmin } = await import(
        "@/integrations/supabase/primary-client.server"
      );
      const { data: profs } = await (supabaseAdmin as any)
        .from("profiles")
        .select("id,full_name,avatar_url")
        .in("id", profileIds);
      for (const p of profs ?? []) {
        profById.set(p.id, { full_name: p.full_name, avatar_url: p.avatar_url });
      }
    } catch (e) {
      console.error("[enrichAppUsers] profile enrich failed", e);
    }
  }
  for (const u of users) {
    const pid = u.profile_id ?? u.auth_user_id ?? null;
    const p = pid ? profById.get(pid) : null;
    map.set(u.id, {
      id: u.id,
      email: u.email ?? null,
      name: p?.full_name ?? (u.email ? String(u.email).split("@")[0] : ""),
      avatar: p?.avatar_url ?? null,
      role: u.role ?? null,
    });
  }
  return map;
}

export const listTeams = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { data: teams, error } = await (context.supabase as any)
      .from("organization_teams")
      .select("id,name,description,branch,status,lead_user_id,organization_id,created_at,updated_at")
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    const rows = teams ?? [];
    if (!rows.length) return [];

    const { data: users } = await (context.supabase as any)
      .from("app_users")
      .select("id,email,profile_id,auth_user_id,team_id,role");
    const allUsers = users ?? [];
    const userMap = await enrichAppUsers(context.supabase, allUsers);

    const membersByTeam = new Map<string, any[]>();
    for (const u of allUsers) {
      if (!u.team_id) continue;
      const enriched = userMap.get(u.id);
      if (!enriched) continue;
      const list = membersByTeam.get(u.team_id) ?? [];
      list.push(enriched);
      membersByTeam.set(u.team_id, list);
    }

    return rows.map((t: any) => {
      const lead = t.lead_user_id ? userMap.get(t.lead_user_id) ?? null : null;
      const members = membersByTeam.get(t.id) ?? [];
      return {
        ...t,
        lead,
        members,
        // Derived (never stored) fields kept for UI compatibility:
        lead_name: lead?.name ?? null,
        lead_email: lead?.email ?? null,
        lead_avatar: lead?.avatar ?? null,
        member_ids: members.map((m) => m.id),
        member_emails: members.map((m) => m.email).filter((e): e is string => !!e),
        user_avatars: members.map((m) => m.avatar).filter((a): a is string => !!a),
        user_count: members.length,
      };
    });
  });

export const upsertTeam = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    id?: string;
    name: string;
    description?: string | null;
    lead_user_id?: string | null;
    member_ids?: string[];
    branch: string;
    status?: "Active" | "Inactive";
  }) => input)
  .handler(async ({ data, context }) => {
    const { id, member_ids, ...rest } = data;
    const orgId = await getOrgId(context.supabase, context.userId);
    const payload = { ...rest, organization_id: orgId };
    const q = id
      ? (context.supabase as any).from("organization_teams").update(payload).eq("id", id).select().single()
      : (context.supabase as any).from("organization_teams").insert(payload).select().single();

    const { data: row, error } = await q;
    if (error) throw new Error(error.message);

    // Sync team membership via app_users.team_id when provided.
    if (Array.isArray(member_ids) && row?.id) {
      const teamId = row.id;
      const { data: current } = await (context.supabase as any)
        .from("app_users")
        .select("id")
        .eq("team_id", teamId);
      const currentIds = new Set<string>((current ?? []).map((u: any) => u.id));
      const nextIds = new Set<string>(member_ids);
      const toRemove = Array.from(currentIds).filter((x) => !nextIds.has(x));
      const toAdd = Array.from(nextIds).filter((x) => !currentIds.has(x));
      if (toRemove.length) {
        await (context.supabase as any)
          .from("app_users")
          .update({ team_id: null })
          .in("id", toRemove);
      }
      if (toAdd.length) {
        await (context.supabase as any)
          .from("app_users")
          .update({ team_id: teamId })
          .in("id", toAdd);
      }
    }
    return row;
  });

export const deleteTeam = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { id: string }) => input)
  .handler(async ({ data, context }) => {
    const { error } = await (context.supabase as any).from("organization_teams").delete().eq("id", data.id);
    if (error) throw new Error(error.message);
    return { ok: true };
  });

// ---------- Models ----------
export const listModels = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { data, error } = await context.supabase
      .from("models")
      .select("*")
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    return data ?? [];
  });

export const upsertModel = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: {
    id?: string;
    name: string;
    task_type: string;
    tags: string[];
    version: string;
    updated_label?: string;
  }) => input)
  .handler(async ({ data, context }) => {
    const { id, ...rest } = data;
    const orgId = await getOrgId(context.supabase, context.userId);
    const payload = { ...rest, organization_id: orgId };
    const q = id
      ? context.supabase.from("models").update(payload).eq("id", id).select().single()
      : context.supabase.from("models").insert(payload).select().single();
    const { data: row, error } = await q;
    if (error) throw new Error(error.message);
    return row;
  });

export const deleteModel = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { id: string }) => input)
  .handler(async ({ data, context }) => {
    const { error } = await context.supabase.from("models").delete().eq("id", data.id);
    if (error) throw new Error(error.message);
    return { ok: true };
  });

// ---------- Admin check (for UI gating) ----------
export const checkIsAdmin = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { data, error } = await context.supabase.rpc("has_role", {
      _user_id: context.userId,
      _role: "admin",
    });
    if (error) return { isAdmin: false };
    return { isAdmin: !!data };
  });

// Inline email availability check used by the New User modal. Returns whether
// the email is already an ACTIVE member of a different organization (or of the
// caller's own organization), so the form can show an error under the field.
export const checkInviteEmail = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { email: string }) => input)
  .handler(async ({ data, context }) => {
    const email = data.email.trim().toLowerCase();
    if (!email || !email.includes("@")) {
      return { available: true as const, reason: null, message: null };
    }

    const orgId = await getOrgId(context.supabase, context.userId);

    const { data: inOrg } = await context.supabase
      .from("app_users")
      .select("id")
      .eq("organization_id", orgId)
      .ilike("email", email)
      .maybeSingle();
    if (inOrg) {
      return {
        available: false as const,
        reason: "already_member" as const,
        message: "This user is already in your organization.",
      };
    }

    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );
    const admin = supabaseAdmin as any;

    const { data: elsewhere } = await admin
      .from("app_users")
      .select("id,organization_id,status")
      .ilike("email", email)
      .neq("organization_id", orgId)
      .limit(1)
      .maybeSingle();

    const { data: profElsewhere } = await admin
      .from("profiles")
      .select("id,organization_id")
      .ilike("company_email", email)
      .neq("organization_id", orgId)
      .limit(1)
      .maybeSingle();

    if (elsewhere?.id || profElsewhere?.organization_id) {
      return {
        available: false as const,
        reason: "other_org" as const,

        message: "This user already exists in another organization.",
      };
    }

    return { available: true as const, reason: null, message: null };
  });
