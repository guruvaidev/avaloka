import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";

export const getMyFirstName = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { supabaseAdmin } = await import(
      "@/integrations/supabase/client.server"
    );
    const supabase = supabaseAdmin as any;
    const userId = context.userId as string;

    const { data: profile } = await supabase
      .from("profiles")
      .select("full_name")
      .or(`id.eq.${userId},user_id.eq.${userId}`)
      .maybeSingle();

    const fullName =
      ((profile?.full_name as string | null | undefined)?.trim() ||
        (context.claims?.user_metadata as any)?.full_name?.trim()) ||
      "";
    const firstName = fullName
      ? fullName.split(/\s+/)[0]
      : ((context.claims?.email as string | null | undefined)?.split("@")[0] ??
          "");

    return { firstName };

  });
