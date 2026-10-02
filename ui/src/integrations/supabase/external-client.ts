// Browser Supabase client — pinned to the external project via config.ts.
import { createClient } from "@supabase/supabase-js";
import {
  SUPABASE_ANON_KEY,
  SUPABASE_STORAGE_KEY,
  SUPABASE_URL,
} from "./config";

// Backwards-compatible re-exports for any callers still using the old names.
export const EXTERNAL_SUPABASE_URL = SUPABASE_URL;
export const EXTERNAL_SUPABASE_ANON_KEY = SUPABASE_ANON_KEY;

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
