import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

// Keep visibility handling local to Ray jobs; do not replace the app's focus manager.
export function useRayJobsVisible(refreshOnFocus = true) {
  const [visible, setVisible] = useState(true);
  const client = useQueryClient();
  useEffect(() => {
    const visibility = () => setVisible(document.visibilityState !== "hidden");
    const focus = () => {
      if (document.visibilityState === "hidden") return;
      void client.refetchQueries({ queryKey: ["ray-jobs"], type: "active", predicate: (query) =>
        ["list", "status", "detail", "logs"].includes(String(query.queryKey[1])) }, { cancelRefetch: false });
    };
    visibility();
    document.addEventListener("visibilitychange", visibility);
    if (refreshOnFocus) window.addEventListener("focus", focus);
    return () => {
      document.removeEventListener("visibilitychange", visibility);
      window.removeEventListener("focus", focus);
    };
  }, [client, refreshOnFocus]);
  return visible;
}