// Resolves the current user's profiles.id from their auth user id.
//
// After the schema change, `projects.owner_id` and `analyses.owner_id`
// reference `public.profiles(id)`, NOT `auth.users(id)`. Use this helper
// anywhere you would previously have used `auth.uid()` as an owner id.

import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";

let cache: { authUserId: string; profileId: string } | null = null;
let inflight: Promise<string | null> | null = null;

export function clearCurrentProfileCache() {
  cache = null;
  inflight = null;
}

// Reset the cache on auth changes so a different signed-in user does not
// inherit the previous profile id.
supabase.auth.onAuthStateChange((event) => {
  if (event === "SIGNED_OUT" || event === "SIGNED_IN" || event === "USER_UPDATED") {
    clearCurrentProfileCache();
  }
});

export async function getCurrentProfileId(): Promise<string | null> {
  const data = { user: await getCachedAuthUser() };
  const authUserId = data.user?.id ?? null;
  if (!authUserId) {
    clearCurrentProfileCache();
    return null;
  }

  if (cache && cache.authUserId === authUserId) return cache.profileId;
  if (inflight) return inflight;

  inflight = (async () => {
    // Preferred: profiles.user_id links to auth.users.id
    const byUser = await supabase
      .from("profiles")
      .select("id")
      .eq("user_id", authUserId)
      .maybeSingle();
    let profileId = byUser.data?.id ?? null;

    // Legacy fallback: profiles.id equals auth uid (handle_new_user default).
    if (!profileId) {
      const byId = await supabase
        .from("profiles")
        .select("id")
        .eq("id", authUserId)
        .maybeSingle();
      profileId = byId.data?.id ?? null;
    }

    if (profileId) cache = { authUserId, profileId };
    return profileId;
  })();

  try {
    return await inflight;
  } finally {
    inflight = null;
  }
}

export async function requireCurrentProfileId(): Promise<string> {
  const id = await getCurrentProfileId();
  if (!id) throw new Error("Profile not found for current user");
  return id;
}
