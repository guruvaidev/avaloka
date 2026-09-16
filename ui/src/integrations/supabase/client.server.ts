// Server-side ADMIN client for the external project. Bypasses RLS.
// Service role key comes from the secret named by SUPABASE_ADMIN_KEY_ENV.
import { createClient } from "@supabase/supabase-js";
import type { Database } from "./types";
import { SUPABASE_ADMIN_KEY_ENV, SUPABASE_URL, SUPABASE_INCLUSTER_URL } from "./config";

function createSupabaseAdminClient() {
  const key = process.env[SUPABASE_ADMIN_KEY_ENV];
  if (!key) {
    throw new Error(
      `Missing ${SUPABASE_ADMIN_KEY_ENV} secret. Add it in Lovable secrets.`,
    );
  }
  return createClient<Database>(SUPABASE_INCLUSTER_URL || SUPABASE_URL, key, {
    auth: {
      storage: undefined,
      persistSession: false,
      autoRefreshToken: false,
    },
  });
}

let _supabaseAdmin: ReturnType<typeof createSupabaseAdminClient> | undefined;

export const supabaseAdmin = new Proxy(
  {} as ReturnType<typeof createSupabaseAdminClient>,
  {
    get(_, prop, receiver) {
      if (!_supabaseAdmin) _supabaseAdmin = createSupabaseAdminClient();
      return Reflect.get(_supabaseAdmin, prop, receiver);
    },
  },
);
