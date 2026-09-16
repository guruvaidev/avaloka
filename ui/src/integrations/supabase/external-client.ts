// Browser Supabase client — pinned to the external project via config.ts.
import { createClient } from "@supabase/supabase-js";
import {
  SUPABASE_ANON_KEY,
  SUPABASE_STORAGE_KEY,
  SUPABASE_URL,
  isSupabaseConfigured,
  missingSupabaseConfig,
} from "./config";

// Backwards-compatible re-exports for any callers still using the old names.
export const EXTERNAL_SUPABASE_URL = SUPABASE_URL;
export const EXTERNAL_SUPABASE_ANON_KEY = SUPABASE_ANON_KEY;

// A misconfigured deployment used to fail as an opaque network error against a
// project the operator had never heard of. Now it says what is unset, once, at
// the point the client is built — without throwing, so the app still renders and
// can show a setup page rather than a white screen.
if (!isSupabaseConfigured() && typeof console !== "undefined") {
  console.error(
    `[avaloka] Supabase is not configured: ${missingSupabaseConfig().join(", ")} unset. ` +
      `Sign-in will not work. See README.md → "Setting up your own credentials".`,
  );
}

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: {
    storage: typeof window !== "undefined" ? window.localStorage : undefined,
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    flowType: "implicit",
    storageKey: SUPABASE_STORAGE_KEY,
  },
});
