import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

/**
 * Resolve the caller's organization id with service-role access so RLS on
 * profiles / organizations / app_users cannot hide the membership row.
 * Used as a fallback when the client-side resolution returns null (typical for
 * invited collaborators and admins whose profile row has no organization_id).
 */
export const resolveMyOrgId = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const { resolveOrgIdForAuthUser } = await import("./current-org.server");
    return await resolveOrgIdForAuthUser(supabaseAdmin as any, context.userId as string);
  });
