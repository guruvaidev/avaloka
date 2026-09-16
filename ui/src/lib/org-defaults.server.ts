// Server-side organization initialization: default division + default roles.
// Implemented in the backend (no SQL triggers / RPC functions).

const MODULES = [
  "User Management",
  "Analysis",
  "Projects",
  "Dashboard",
  "Data Configuration",
] as const;

const ACTIONS = ["view", "create", "edit", "approve"] as const;

const ADMIN_PERMISSIONS = MODULES.flatMap((m) => ACTIONS.map((a) => `${m}:${a}`));
const ANALYST_PERMISSIONS = MODULES.map((m) => `${m}:view`);

// Full permission matrix, lowercase-keyed (same shape the UI stores).
const OWNER_PERMISSIONS: Record<string, Record<string, boolean>> = MODULES.reduce(
  (acc, m) => {
    acc[m.toLowerCase()] = ACTIONS.reduce(
      (row, a) => {
        row[a] = true;
        return row;
      },
      {} as Record<string, boolean>,
    );
    return acc;
  },
  {} as Record<string, Record<string, boolean>>,
);

/**
 * Ensures the organization owner exists in `app_users` with role "Owner" and
 * every permission enabled. Backend-only (no SQL triggers / RPC). Idempotent.
 */
export async function seedOrganizationOwnerAppUser(
  supabase: any,
  organizationId: string,
) {
  if (!organizationId) return;

    const { data: org, error: orgError } = await supabase
      .from("organizations")
      .select("id, owner_profile_id")
      .eq("id", organizationId)
      .maybeSingle();

    if (orgError) {
      throw new Error(`organization read failed: ${orgError.message}`);
    }

    const ownerProfileId = org?.owner_profile_id as string | undefined;
    if (!ownerProfileId) {
      throw new Error(`organization ${organizationId} has no owner_profile_id`);
    }

    const { data: profile, error: profileError } = await supabase
      .from("profiles")
      .select("id, user_id, full_name, organization_id")
      .eq("id", ownerProfileId)
      .maybeSingle();

    if (profileError) {
      throw new Error(`owner profile read failed: ${profileError.message}`);
    }
    if (!profile?.id) {
      throw new Error(`owner profile ${ownerProfileId} was not found`);
    }

    const authUserId = (profile?.user_id ?? profile?.id ?? ownerProfileId) as string;

    // The owner's profile must point at the organization, otherwise every
    // org-scoped read (divisions, roles, users) comes back empty.
    if (profile.organization_id !== organizationId) {
      const { error: linkError } = await supabase
        .from("profiles")
        .update({ organization_id: organizationId, updated_at: new Date().toISOString() })
        .eq("id", ownerProfileId);
      if (linkError) {
        console.error("[org-defaults] owner profile org link failed", linkError);
      }
    }

    // email is NOT NULL — resolve it, with a last-resort placeholder.
    let email: string | null = null;
    try {
      const { supabaseAdmin } = await import(
        "@/integrations/supabase/primary-client.server"
      );
      const { data: authUser } = await (supabaseAdmin as any).auth.admin.getUserById(
        authUserId,
      );
      email = authUser?.user?.email ?? null;
    } catch (e) {
      console.error("[org-defaults] owner email lookup failed", e);
    }
    if (!email) email = `owner+${authUserId}@unknown.local`;


    const payload: Record<string, unknown> = {
      role: "Owner",
      permissions: OWNER_PERMISSIONS,
      status: "Active",
      organization_id: organizationId,
      profile_id: ownerProfileId,
      auth_user_id: authUserId,
      accepted_at: new Date().toISOString(),
    };

    // Existing row lookup — separate queries (auth_user_id is UNIQUE table-wide).
    const findExistingId = async (): Promise<string | null> => {
      const byAuth = await supabase
        .from("app_users")
        .select("id")
        .eq("auth_user_id", authUserId)
        .limit(1);
      if (byAuth.error) {
        throw new Error(`owner lookup by auth user failed: ${byAuth.error.message}`);
      }
      if (byAuth.data?.[0]?.id) return byAuth.data[0].id as string;

      const byProfile = await supabase
        .from("app_users")
        .select("id")
        .eq("organization_id", organizationId)
        .eq("profile_id", ownerProfileId)
        .limit(1);
      if (byProfile.error) {
        throw new Error(`owner lookup by profile failed: ${byProfile.error.message}`);
      }
      if (byProfile.data?.[0]?.id) return byProfile.data[0].id as string;

      const byEmail = await supabase
        .from("app_users")
        .select("id")
        .eq("organization_id", organizationId)
        .ilike("email", email as string)
        .limit(1);
      if (byEmail.error) {
        throw new Error(`owner lookup by email failed: ${byEmail.error.message}`);
      }
      if (byEmail.data?.[0]?.id) return byEmail.data[0].id as string;

      return null;
    };

    const existingId = await findExistingId();

    if (existingId) {
      const { error } = await supabase
        .from("app_users")
        .update(payload)
        .eq("id", existingId);
      if (error) throw new Error(`owner update failed: ${error.message}`);
      return;
    }

    const { error: insertError } = await supabase.from("app_users").insert({
      ...payload,
      email,
      user_id_code: `OWN-${organizationId.replace(/-/g, "").slice(0, 8).toUpperCase()}`,
      department: "Management",
      branch: "HQ",
    });

    if (insertError) {
      // Most likely the app_users_auth_user_id_key unique constraint.
      const retryId = await findExistingId();
      if (retryId) {
        const { error } = await supabase
          .from("app_users")
          .update(payload)
          .eq("id", retryId);
        if (error) throw new Error(`owner conflict update failed: ${error.message}`);
        return;
      }
      throw new Error(`owner insert failed: ${insertError.message}`);
    }
}


/**
 * Seeds the default "Management" division and the Admin / Analyst roles for a
 * newly created organization. Idempotent: existing rows are left untouched.
 */
export async function seedOrganizationDefaults(supabase: any, organizationId: string) {
  if (!organizationId) return;

  // Owner first: it also links the owner profile to the organization, which is
  // what every org-scoped read depends on. Never let one step block the other.
  const ownerResult = await seedOrganizationOwnerAppUser(supabase, organizationId).then(
    () => null,
    (e: unknown) => e,
  );

  try {


    const { data: existingDepts, error: departmentsReadError } = await supabase
      .from("departments")
      .select("id")
      .eq("organization_id", organizationId)
      .ilike("department", "Management")
      .limit(1);

    if (departmentsReadError) {
      throw new Error(`departments read failed: ${departmentsReadError.message}`);
    }

    if (!existingDepts?.[0]?.id) {
      const { error } = await supabase.from("departments").insert({
        organization_id: organizationId,
        department: "Management",
        code: "MGMT",
        employees: 0,
      });
      if (error) throw new Error(`division insert failed: ${error.message}`);
    }


    const { data: existingRoles, error: rolesReadError } = await supabase
      .from("config_roles")
      .select("role")
      .eq("organization_id", organizationId);

    if (rolesReadError) {
      throw new Error(`roles read failed: ${rolesReadError.message}`);
    }

    const have = new Set(
      (existingRoles ?? []).map((r: { role: string }) => String(r.role).toLowerCase()),
    );

    const toInsert: Record<string, unknown>[] = [];
    if (!have.has("admin")) {
      toInsert.push({
        organization_id: organizationId,
        role: "Admin",
        department: "Management",
        permissions: ADMIN_PERMISSIONS,
      });
    }
    if (!have.has("analyst")) {
      toInsert.push({
        organization_id: organizationId,
        role: "Analyst",
        department: "Management",
        permissions: ANALYST_PERMISSIONS,
      });
    }

    if (toInsert.length) {
      const { error } = await supabase.from("config_roles").insert(toInsert);
      if (error) throw new Error(`roles insert failed: ${error.message}`);
    }
  } catch (e) {
    if (ownerResult) console.error("[org-defaults] owner seeding failed", ownerResult);
    throw e;
  }

  if (ownerResult) throw ownerResult;
}

