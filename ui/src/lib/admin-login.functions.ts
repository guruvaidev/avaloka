import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { normalizeRedirectOrigin } from "@/lib/auth-redirect";


const ADMIN_EMAIL = "kishan.kumar@guruvaisciences.org";

export const DEACTIVATED_MESSAGE =
  "Your account has been deactivated. Please contact your organization administrator.";

async function clearStandaloneGeneratedOrg(admin: any, email: string) {
  try {
    const normalized = email.trim().toLowerCase();
    const { data: list } = await admin.auth.admin.listUsers({
      page: 1,
      perPage: 1,
      email: normalized,
    } as any);
    const user = list?.users?.find(
      (u: any) => String(u.email ?? "").toLowerCase() === normalized,
    );
    if (!user?.id) return;

    const byAuthId = await admin
      .from("app_users")
      .select("id,organization_id")
      .eq("auth_user_id", user.id)
      .maybeSingle();

    let appUser = byAuthId.data ?? null;
    if (!appUser) {
      const byEmail = await admin
        .from("app_users")
        .select("id,organization_id")
        .ilike("email", normalized)
        .maybeSingle();
      appUser = byEmail.data ?? null;
    }

    if (appUser?.organization_id) return;

    const { data: profile } = await admin
      .from("profiles")
      .select("id,organization_id")
      .or(`id.eq.${user.id},user_id.eq.${user.id}`)
      .maybeSingle();

    if (!profile?.organization_id) return;

    // Enterprise access is derived from an active subscription on the org.
    const { data: sub } = await admin
      .from("subscriptions")
      .select("status, plan:plans(plan_type)")
      .eq("organization_id", profile.organization_id)
      .in("status", ["trialing", "active", "past_due"])
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (sub && (sub as any)?.plan?.plan_type === "enterprise") return;

    const { error } = await admin
      .from("profiles")
      .update({ organization_id: null, updated_at: new Date().toISOString() })
      .eq("id", profile.id);
    if (error) console.error("Failed to clear generated organization_id", error);
  } catch (error) {
    console.error("Failed to normalize signup organization", error);
  }
}

export const sendLoginMagicLink = createServerFn({ method: "POST" })
  .inputValidator((data: { email: string; redirectTo: string }) => {
    const email = String(data?.email ?? "").trim().toLowerCase();
    if (!/^\S+@\S+\.\S+$/.test(email)) throw new Error("Enter a valid email address");
    // Environment-aware: the caller sends its own window.location.origin,
    // validated against the shared allow-list (no open redirect).
    const redirectTo = normalizeRedirectOrigin(String(data?.redirectTo ?? ""));
    return { email, redirectTo };
  })

  .handler(async ({ data }) => {
    const { createClient } = await import("@supabase/supabase-js");
    const { SUPABASE_ANON_KEY, SUPABASE_INCLUSTER_URL } = await import(
      "@/integrations/supabase/config"
    );
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );

    // Deactivated accounts (and members of orgs no longer on Enterprise)
    // must not receive a login link at all.
    const { loginBlockReason } = await import("@/lib/org-access.server");
    const blocked = await loginBlockReason(supabaseAdmin as any, {
      email: data.email,
    });
    if (blocked) throw new Error(blocked);

    const authClient = createClient(SUPABASE_INCLUSTER_URL, SUPABASE_ANON_KEY, {
      auth: {
        storage: undefined,
        persistSession: false,
        autoRefreshToken: false,
      },
    });

    const { error } = await authClient.auth.signInWithOtp({
      email: data.email,
      options: { emailRedirectTo: data.redirectTo },
    });
    if (error) throw new Error(error.message);

    // Single normalization pass — the retry/sleep loop added seconds of latency
    // to the "send login link" request without changing the outcome.
    await clearStandaloneGeneratedOrg(supabaseAdmin as any, data.email);

    return { ok: true };
  });

export const generateAdminMagicLink = createServerFn({ method: "POST" }).handler(
  async () => {
    const { supabaseAdmin } = await import("@/integrations/supabase/primary-client.server");

    // Find the admin user, paginating through auth.users (listUsers() only
    // returns the first page — with many users the admin can be far down).
    const findAdminId = async (): Promise<string | undefined> => {
      for (let page = 1; page <= 40; page++) {
        const { data, error } = await supabaseAdmin.auth.admin.listUsers({
          page,
          perPage: 200,
        });
        if (error) throw new Error(error.message);
        const hit = data?.users.find(
          (u) => String(u.email ?? "").toLowerCase() === ADMIN_EMAIL.toLowerCase(),
        );
        if (hit) return hit.id;
        if (!data?.users?.length || data.users.length < 200) return undefined;
      }
      return undefined;
    };

    let userId = await findAdminId();
    if (!userId) {
      const { data: created, error: createErr } =
        await supabaseAdmin.auth.admin.createUser({
          email: ADMIN_EMAIL,
          email_confirm: true,
        });
      if (createErr) {
        // Race / stale lookup: the user actually exists — re-resolve it.
        userId = await findAdminId();
        if (!userId) throw new Error(createErr.message);
      } else {
        userId = created.user?.id;
      }
    }


    // Ensure admin role
    if (userId) {
      await supabaseAdmin
        .from("user_roles")
        .upsert(
          { user_id: userId, role: "admin" },
          { onConflict: "user_id,role", ignoreDuplicates: true },
        );
    }

    // Generate a magic-link token we can verify directly on the client
    const { data, error } = await supabaseAdmin.auth.admin.generateLink({
      type: "recovery",
      email: ADMIN_EMAIL,
    });

    const tokenHash = data?.properties?.hashed_token;
    if (error || !tokenHash) {
      throw new Error(error?.message ?? "Failed to generate admin login token");
    }

    return { token_hash: tokenHash, email: ADMIN_EMAIL };
  },
);

/** One-click sign-in for the support desk account (support@avaloka.ai). */
export const generateSupportAdminMagicLink = createServerFn({ method: "POST" }).handler(
  async () => {
    const { SUPPORT_ADMIN_EMAIL } = await import("./support-admin");
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );

    const findUserId = async (): Promise<string | undefined> => {
      for (let page = 1; page <= 40; page++) {
        const { data, error } = await supabaseAdmin.auth.admin.listUsers({
          page,
          perPage: 200,
        });
        if (error) throw new Error(error.message);
        const hit = data?.users.find(
          (u) => String(u.email ?? "").toLowerCase() === SUPPORT_ADMIN_EMAIL,
        );
        if (hit) return hit.id;
        if (!data?.users?.length || data.users.length < 200) return undefined;
      }
      return undefined;
    };

    let userId = await findUserId();
    if (!userId) {
      const { data: created, error: createErr } =
        await supabaseAdmin.auth.admin.createUser({
          email: SUPPORT_ADMIN_EMAIL,
          email_confirm: true,
        });
      if (createErr) {
        userId = await findUserId();
        if (!userId) throw new Error(createErr.message);
      } else {
        userId = created.user?.id;
      }
    }

    const { data, error } = await supabaseAdmin.auth.admin.generateLink({
      type: "recovery",
      email: SUPPORT_ADMIN_EMAIL,
    });
    const tokenHash = data?.properties?.hashed_token;
    if (error || !tokenHash) {
      throw new Error(error?.message ?? "Failed to generate support login token");
    }

    return { token_hash: tokenHash, email: SUPPORT_ADMIN_EMAIL };
  },
);


// Returns whether the signed-in caller's app_users record is still active.
// Users with no app_users record (org owners signing up, admin login) are
// treated as active.
export const checkMyAccountActive = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/primary-client.server"
    );

    const email = String((context.claims as any)?.email ?? "").trim().toLowerCase();

    const { loginBlockReason } = await import("@/lib/org-access.server");
    const blocked = await loginBlockReason(supabaseAdmin as any, {
      authUserId: context.userId,
      email,
    });
    const active = !blocked;
    return { active, message: blocked };
  });
