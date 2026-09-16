// Server-side admin client — duplicate "primary" variant kept for back-compat.
// Same external project, same service-role secret.
import { createClient } from "@supabase/supabase-js";
import type { Database } from "./types";
import { SUPABASE_ADMIN_KEY_ENV, SUPABASE_URL, SUPABASE_INCLUSTER_URL } from "./config";

function createPrimaryAdmin() {
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

let _client: ReturnType<typeof createPrimaryAdmin> | undefined;

export const supabaseAdmin = new Proxy(
  {} as ReturnType<typeof createPrimaryAdmin>,
  {
    get(_, prop, receiver) {
      if (!_client) _client = createPrimaryAdmin();
      return Reflect.get(_client, prop, receiver);
    },
  },
);
