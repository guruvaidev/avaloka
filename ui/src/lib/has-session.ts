import { supabase } from "@/integrations/supabase/client";

/**
 * True when the browser currently holds a Supabase session.
 * Server functions guarded by `requireSupabaseAuth` throw
 * "Unauthorized: No authorization header provided" without one, so callers
 * should skip them instead of firing a request that is certain to fail.
 */
export async function hasSupabaseSession(): Promise<boolean> {
  try {
    const { data } = await supabase.auth.getSession();
    return !!data.session?.access_token;
  } catch {
    return false;
  }
}
