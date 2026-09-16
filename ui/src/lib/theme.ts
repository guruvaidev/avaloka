import { useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";

export type ThemeMode = "system" | "light" | "dark";

const STORAGE_KEY = "avaloka:theme";

function resolveEffective(mode: ThemeMode): "light" | "dark" {
  if (mode === "system") {
    if (typeof window === "undefined") return "light";
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return mode;
}

export function applyTheme(mode: ThemeMode) {
  if (typeof document === "undefined") return;
  const eff = resolveEffective(mode);
  const root = document.documentElement;
  root.classList.toggle("dark-mode", eff === "dark");
  root.classList.toggle("dark", eff === "dark");
  root.style.colorScheme = eff;
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {}
}

export function setTheme(mode: ThemeMode) {
  applyTheme(mode);
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("app:theme-change", { detail: mode }));
  }
}

async function loadThemeFromProfile(): Promise<ThemeMode> {
  try {
    const userRes = { user: await getCachedAuthUser() };
    const uid = userRes.user?.id;
    if (!uid) return (localStorage.getItem(STORAGE_KEY) as ThemeMode) || "system";
    const { data } = await supabase
      .from("user_settings" as any)
      .select("theme")
      .eq("user_id", uid)
      .maybeSingle();
    const t = (data as any)?.theme as ThemeMode | undefined;
    if (t) return t;
  } catch {}
  try {
    return (localStorage.getItem(STORAGE_KEY) as ThemeMode) || "system";
  } catch {
    return "system";
  }
}

export function useThemeApplier() {
  useEffect(() => {
    // Apply cached theme immediately to avoid flash
    try {
      const cached = (localStorage.getItem(STORAGE_KEY) as ThemeMode) || "system";
      applyTheme(cached);
    } catch {}

    let cancelled = false;
    (async () => {
      const t = await loadThemeFromProfile();
      if (!cancelled) applyTheme(t);
    })();

    const onChange = (e: Event) => {
      const mode = (e as CustomEvent).detail as ThemeMode;
      if (mode) applyTheme(mode);
    };
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onMedia = () => {
      const cur = (localStorage.getItem(STORAGE_KEY) as ThemeMode) || "system";
      if (cur === "system") applyTheme("system");
    };

    window.addEventListener("app:theme-change", onChange);
    mql.addEventListener?.("change", onMedia);

    const { data: sub } = supabase.auth.onAuthStateChange(async () => {
      const t = await loadThemeFromProfile();
      applyTheme(t);
    });

    return () => {
      cancelled = true;
      window.removeEventListener("app:theme-change", onChange);
      mql.removeEventListener?.("change", onMedia);
      sub.subscription.unsubscribe();
    };
  }, []);
}
