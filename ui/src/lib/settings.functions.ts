import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

export type UpdateOrganizationProfileInput = {
  orgId: string;
  payload: {
    name?: string;
    slug?: string | null;
    tagline?: string | null;
    logo_url?: string | null;
    brand_reports?: boolean;
    brand_emails?: boolean;
  };
};

export const updateOrganizationProfile = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: UpdateOrganizationProfileInput) => {
    if (!input || typeof input !== "object") throw new Error("Missing organization settings");
    const orgId = String(input.orgId ?? "").trim();
    if (!orgId) throw new Error("No organization associated with this account.");

    const source = input.payload ?? {};
    const logoRaw = source.logo_url;
    const logoNormalized =
      logoRaw === undefined
        ? undefined
        : logoRaw === null
        ? null
        : String(logoRaw).trim().slice(0, 2048) || null;

    const payload: UpdateOrganizationProfileInput["payload"] = {
      tagline: source.tagline ? String(source.tagline) : null,
      brand_reports: Boolean(source.brand_reports),
      brand_emails: Boolean(source.brand_emails),
      slug: source.slug ? String(source.slug) : null,
    };

    if (logoNormalized !== undefined) {
      payload.logo_url = logoNormalized;
    }

    if (source.name && String(source.name).trim()) {
      payload.name = String(source.name).trim();
    }

    return { orgId, payload };
  })
  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;

    const { data: profile, error: profileError } = await supabase
      .from("profiles")
      .select("id,user_id,organization_id")
      .or(`user_id.eq.${context.userId},id.eq.${context.userId}`)
      .maybeSingle();

    if (profileError) throw new Error(`Profile lookup failed: ${profileError.message}`);
    if (!profile?.id) throw new Error("Profile not found for the signed-in user.");
    if (profile.organization_id !== data.orgId) {
      throw new Error("You can only update the organization linked to your profile.");
    }

    const { data: organization, error: orgError } = await supabase
      .from("organizations")
      .select("id,owner_profile_id")
      .eq("id", data.orgId)
      .maybeSingle();

    if (orgError) throw new Error(`Organization lookup failed: ${orgError.message}`);
    if (!organization?.id) throw new Error("Organization record was not found.");
    if (organization.owner_profile_id !== profile.id) {
      const { data: roleRow, error: roleError } = await supabase
        .from("user_roles")
        .select("id")
        .eq("user_id", context.userId)
        .eq("role", "admin")
        .maybeSingle();

      if (roleError) throw new Error(`Role lookup failed: ${roleError.message}`);
      if (!roleRow?.id) {
        throw new Error("Only the organization owner or an admin can update profile branding.");
      }
    }

    const { data: updated, error: updateError } = await supabase
      .from("organizations")
      .update(data.payload)
      .eq("id", data.orgId)
      .select("id,name,slug,tagline,logo_url,brand_reports,brand_emails")
      .maybeSingle();

    if (updateError) throw new Error(`Organization update failed: ${updateError.message}`);
    if (!updated?.id) throw new Error("Organization settings were not updated.");

    return updated as {
      id: string;
      name: string;
      slug: string | null;
      tagline: string | null;
      logo_url: string | null;
      brand_reports: boolean;
      brand_emails: boolean;
      owner_profile_id: string | null;
    };
  });


export const getOrganizationProfile = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { orgId: string }) => {
    const orgId = String(input?.orgId ?? "").trim();
    if (!orgId) throw new Error("orgId required");
    return { orgId };
  })
  .handler(async ({ context, data }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const supabase = supabaseAdmin as any;

    const { data: profile } = await supabase
      .from("profiles")
      .select("id,organization_id")
      .or(`user_id.eq.${context.userId},id.eq.${context.userId}`)
      .maybeSingle();

    if (!profile?.id || profile.organization_id !== data.orgId) {
      throw new Error("Not authorized to read this organization.");
    }

    const { data: org, error } = await supabase
      .from("organizations")
      .select("id,name,slug,tagline,logo_url,brand_reports,brand_emails,owner_profile_id")
      .eq("id", data.orgId)
      .maybeSingle();

    if (error) throw new Error(error.message);
    return org as {
      id: string;
      name: string;
      slug: string | null;
      tagline: string | null;
      logo_url: string | null;
      brand_reports: boolean;
      brand_emails: boolean;
      owner_profile_id: string | null;
    } | null;
  });