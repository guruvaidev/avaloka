// Cached auth-user resolver.
//
// `supabase.auth.getUser()` performs a network round-trip to the auth server on
// every call. The app calls it on nearly every route change / query, which made
// navigation feel slow. This wrapper serves the locally-stored session (already
// validated + auto-refreshed by the supabase client) and only falls back to a
// network `getUser()` when no session is cached. The cache is cleared on any
// auth state change, so a sign-out / sign-in still resolves correctly.

import { supabase } from "@/integrations/supabase/client";
import type { User } from "@supabase/supabase-js";

let cached: { user: User | null; at: number } | null = null;
let inflight: Promise<User | null> | null = null;

const TTL_MS = 30_000;

export function clearCachedAuthUser() {
  cached = null;
  inflight = null;
}

supabase.auth.onAuthStateChange((_event, session) => {
  cached = { user: session?.user ?? null, at: Date.now() };
});

export async function getCachedAuthUser(): Promise<User | null> {
  if (cached && Date.now() - cached.at < TTL_MS) return cached.user;
  if (inflight) return inflight;

  inflight = (async () => {
    const { data: sessionRes } = await supabase.auth.getSession();
    let user = sessionRes.session?.user ?? null;
    if (!user) {
      const { data } = await supabase.auth.getUser();
      user = data.user ?? null;
    }
    cached = { user, at: Date.now() };
    return user;
  })();

  try {
    return await inflight;
  } finally {
    inflight = null;
  }
}

export async function getCachedAuthUserId(): Promise<string | null> {
  return (await getCachedAuthUser())?.id ?? null;
}
