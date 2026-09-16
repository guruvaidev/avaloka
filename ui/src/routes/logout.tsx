import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/logout")({
  component: LogoutPage,
});

function LogoutPage() {
  const navigate = useNavigate();

  useEffect(() => {
    (async () => {
      try {
        await supabase.auth.signOut();
      } catch {
        // ignore
      }
      localStorage.clear();
      sessionStorage.clear();
      navigate({ to: "/", replace: true });
    })();
  }, [navigate]);

  return (
    <div className="grid min-h-screen place-items-center text-sm text-muted-foreground">
      Signing out…
    </div>
  );
}
