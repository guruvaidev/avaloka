import { useCallback, useEffect, useRef, useState } from "react";
import { speakText, stopSpeaking } from "@/hooks/useVoiceInput";
import {
  AUTO_INSIGHTS_BOT_FINISH_EVENT,
  AUTO_INSIGHTS_BOT_START_EVENT,
  isAutoInsightsBotActive,
} from "@/lib/auto-insights-bot";
import { cx } from "@/lib/utils/cx";

const IDLE_AVATAR = "/assets/dashboard/avatar_bot.png";
const SPEAKING_AVATAR = "/assets/dashboard/avatar_tts.mp4";
const MESSAGE =
  "I'm running Auto Insights on your uploaded documents — pulling out the key metrics and patterns. This'll just take a moment.";

export function AutoInsightsBotOverlay() {
  const [visible, setVisible] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearHideTimer = useCallback(() => {
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    hideTimerRef.current = null;
  }, []);

  const clearDismissTimer = useCallback(() => {
    if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    dismissTimerRef.current = null;
  }, []);

  const begin = useCallback(() => {
    clearHideTimer();
    clearDismissTimer();
    setVisible(true);
    setFinishing(false);
    setSpeaking(false);
    speakText(MESSAGE, {
      onStart: () => setSpeaking(true),
      onEnd: () => setSpeaking(false),
    });
    dismissTimerRef.current = setTimeout(() => {
      stopSpeaking();
      setSpeaking(false);
      setVisible(false);
      setFinishing(false);
    }, 10000);
  }, [clearHideTimer, clearDismissTimer]);

  const finish = useCallback(() => {
    clearDismissTimer();
    stopSpeaking();
    setSpeaking(false);
    setFinishing(true);
    clearHideTimer();
    hideTimerRef.current = setTimeout(() => {
      setVisible(false);
      setFinishing(false);
    }, 900);
  }, [clearHideTimer, clearDismissTimer]);

  useEffect(() => {
    const onStart = () => begin();
    const onFinish = () => finish();
    window.addEventListener(AUTO_INSIGHTS_BOT_START_EVENT, onStart);
    window.addEventListener(AUTO_INSIGHTS_BOT_FINISH_EVENT, onFinish);
    if (isAutoInsightsBotActive()) begin();
    return () => {
      window.removeEventListener(AUTO_INSIGHTS_BOT_START_EVENT, onStart);
      window.removeEventListener(AUTO_INSIGHTS_BOT_FINISH_EVENT, onFinish);
      clearHideTimer();
      clearDismissTimer();
      stopSpeaking();
    };
  }, [begin, clearHideTimer, clearDismissTimer, finish]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (speaking) {
      video.play().catch(() => {
        /* The idle image remains visible if browser autoplay is unavailable. */
      });
      return;
    }
    video.pause();
    try {
      video.currentTime = 0;
    } catch {
      /* Ignore browsers that do not allow seeking before metadata loads. */
    }
  }, [speaking]);

  if (!visible) return null;

  return (
    <aside
      aria-live="polite"
      aria-label="Auto Insights generation status"
      className={cx(
        "pointer-events-none fixed bottom-5 right-5 z-[100] flex w-[min(22rem,calc(100vw-2rem))] items-center gap-3 rounded-lg border border-secondary bg-primary p-3 shadow-xl transition-all duration-500 motion-reduce:transition-none",
        finishing ? "translate-y-2 opacity-0" : "translate-y-0 opacity-100",
      )}
    >
      <div className="relative size-20 shrink-0 overflow-hidden rounded-full">
        <span className="pointer-events-none absolute inset-2 rounded-full bg-brand-600/20 animate-[voice-thinking-pulse_1.5s_ease-in-out_infinite] motion-reduce:animate-none" />
        <img
          src={IDLE_AVATAR}
          alt="Avaloka AI"
          className={cx(
            "absolute inset-0 size-full object-cover transition-opacity duration-200",
            speaking ? "opacity-0" : "opacity-100",
          )}
        />
        <video
          ref={videoRef}
          src={SPEAKING_AVATAR}
          loop
          muted
          playsInline
          preload="auto"
          aria-hidden
          className={cx(
            "absolute left-[4%] top-[-14%] h-auto w-[92%] object-cover transition-opacity duration-200",
            speaking ? "opacity-100" : "opacity-0",
          )}
        />
      </div>
      <div className="min-w-0">
        <p className="text-sm font-semibold text-primary">Running Auto Insights</p>
        <p className="mt-1 text-xs leading-5 text-tertiary">Pulling out key metrics and patterns…</p>
      </div>
    </aside>
  );
}