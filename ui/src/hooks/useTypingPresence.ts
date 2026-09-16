import { useCallback, useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { getCurrentProfileId } from "@/lib/current-profile";

export type TypingUser = {
  profileId: string;
  name: string;
  avatarUrl: string | null;
};

type PresencePayload = TypingUser & { typing: boolean; at: number };

const BROADCAST_THROTTLE_MS = 1500;
const IDLE_STOP_MS = 2000;
const EXPIRY_MS = 3000;

/**
 * Ephemeral "X is typing…" presence for a shared analysis chat.
 * Nothing is persisted — state lives only on a Supabase Realtime channel.
 */
export function useTypingPresence(analysisId: string | null | undefined) {
  const [typingUsers, setTypingUsers] = useState<TypingUser[]>([]);
  const channelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);
  const meRef = useRef<TypingUser | null>(null);
  const lastSentRef = useRef(0);
  const isTypingRef = useRef(false);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sweepTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const seenRef = useRef<Map<string, { user: TypingUser; at: number }>>(new Map());

  const flush = useCallback(() => {
    const now = Date.now();
    const next: TypingUser[] = [];
    seenRef.current.forEach((entry, key) => {
      if (now - entry.at > EXPIRY_MS) seenRef.current.delete(key);
      else next.push(entry.user);
    });
    setTypingUsers((prev) => {
      const sameLength = prev.length === next.length;
      if (sameLength && prev.every((p, i) => p.profileId === next[i].profileId)) return prev;
      return next;
    });
  }, []);

  useEffect(() => {
    if (!analysisId) {
      setTypingUsers([]);
      return;
    }
    let cancelled = false;

    (async () => {
      const profileId = await getCurrentProfileId();
      if (cancelled || !profileId) return;

      const { data: profile } = await supabase
        .from("profiles")
        .select("full_name, avatar_url")
        .eq("id", profileId)
        .maybeSingle();

      let name = profile?.full_name?.trim() ?? "";

      if (!name) {
        const { data: appUser } = await supabase
          .from("app_users")
          .select("email")
          .or(`profile_id.eq.${profileId},auth_user_id.eq.${profileId}`)
          .limit(1)
          .maybeSingle();
        name = appUser?.email?.trim() ?? "";
      }

      if (!name) {
        const { data: authData } = await supabase.auth.getUser();
        name = authData.user?.email?.trim() ?? "";
      }

      meRef.current = { profileId, name: name || "Someone", avatarUrl: profile?.avatar_url ?? null };

      const channel = supabase.channel(`analysis-typing:${analysisId}`, {
        config: { broadcast: { self: false } },
      });

      channel
        .on("broadcast", { event: "typing" }, ({ payload }) => {
          const p = payload as PresencePayload;
          if (!p?.profileId || p.profileId === meRef.current?.profileId) return;
          if (p.typing) {
            seenRef.current.set(p.profileId, {
              user: { profileId: p.profileId, name: p.name, avatarUrl: p.avatarUrl ?? null },
              at: Date.now(),
            });
          } else {
            seenRef.current.delete(p.profileId);
          }
          flush();
        })
        .subscribe();

      channelRef.current = channel;
      sweepTimerRef.current = setInterval(flush, 1000);
    })();

    return () => {
      cancelled = true;
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
      if (sweepTimerRef.current) clearInterval(sweepTimerRef.current);
      const ch = channelRef.current;
      if (ch) {
        const me = meRef.current;
        if (me && isTypingRef.current) {
          void ch.send({ type: "broadcast", event: "typing", payload: { ...me, typing: false, at: Date.now() } });
        }
        supabase.removeChannel(ch);
      }
      channelRef.current = null;
      isTypingRef.current = false;
      seenRef.current.clear();
      setTypingUsers([]);
    };
  }, [analysisId, flush]);

  const emit = useCallback((typing: boolean) => {
    const ch = channelRef.current;
    const me = meRef.current;
    if (!ch || !me) return;
    isTypingRef.current = typing;
    lastSentRef.current = Date.now();
    void ch.send({ type: "broadcast", event: "typing", payload: { ...me, typing, at: Date.now() } });
  }, []);

  /** Call on every keystroke in the composer. */
  const notifyTyping = useCallback(() => {
    const now = Date.now();
    if (!isTypingRef.current || now - lastSentRef.current > BROADCAST_THROTTLE_MS) emit(true);
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(() => emit(false), IDLE_STOP_MS);
  }, [emit]);

  /** Call on submit / blur / clear. */
  const stopTyping = useCallback(() => {
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    if (isTypingRef.current) emit(false);
  }, [emit]);

  return { typingUsers, notifyTyping, stopTyping };
}
