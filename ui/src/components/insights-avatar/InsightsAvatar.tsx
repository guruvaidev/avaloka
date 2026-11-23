import { useCallback, useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { Microphone01, Send01, Stars02, StopSquare, VolumeMax, VolumeX, X } from "@untitledui/icons";
import type { Slide } from "@/components/dashboard/dynamicChart";
import { getRenderedPoints } from "@/lib/chart-points";
import { EXPLAIN_MESSAGES, explainInsights, type ExplainChart } from "@/lib/insightsExplainApi";
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition";
import { useSpeechSynthesis } from "@/hooks/useSpeechSynthesis";
import { supabase } from "@/integrations/supabase/client";
import { getCurrentProfileId } from "@/lib/current-profile";
import { AvatarFace, type AvatarState } from "./AvatarFace";
import { useInsightsAvatar } from "./InsightsAvatarContext";
import { CommentaryText, commentaryPlainText } from "@/components/chat/CommentaryText";
import type { ReplySegment } from "@/lib/reply-segments";

type Msg = { id: number; role: "user" | "assistant"; text: string; segments?: ReplySegment[] | null };

export type AvatarChart = { id: string; slide: Slide };

const SUGGESTIONS = ["Summarize all charts", "What's the most important insight?", "Which chart looks unreliable?"];
const STATUS: Record<AvatarState, string> = {
  idle: "Tap the mic and ask a question",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};
const FOCUS_RING = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2";

function chartType(slide: Slide): ExplainChart["type"] {
  if (slide.chartKind === "histogram") return "histogram";
  if (slide.type === "donut" || slide.type === "pie") return "pie";
  if (slide.type === "area" || slide.type === "line") return "line";
  return slide.type;
}

function toPayloadCharts(charts: AvatarChart[]): ExplainChart[] {
  return charts.slice(0, 12).map(({ id, slide }) => ({
    id,
    title: slide.title,
    type: chartType(slide),
    x_field: slide.xField ?? null,
    y_field: slide.yField ?? null,
    aggregate: slide.aggregate ?? null,
    reason: slide.reason ?? null,
    points: getRenderedPoints(slide),
  }));
}

/** Explain button placed next to a chart title. Renders a span so it can live inside card buttons. */
export function ExplainChartButton({ chartId }: { chartId: string }) {
  const { requestExplain } = useInsightsAvatar();
  const trigger = (e: { preventDefault: () => void; stopPropagation: () => void }) => {
    e.preventDefault();
    e.stopPropagation();
    requestExplain(chartId);
  };
  return (
    <span
      role="button"
      tabIndex={0}
      title="Explain this chart"
      aria-label="Explain this chart"
      onClick={trigger}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") trigger(e);
      }}
      className={`ml-auto inline-grid size-6 shrink-0 cursor-pointer place-items-center rounded-md text-[#1a5fe8] transition hover:bg-[#eff4ff] ${FOCUS_RING}`}
    >
      <Stars02 className="size-3.5" />
    </span>
  );
}

export function InsightsAvatar({
  aid,
  datasetId,
  datasetName,
  charts,
  open,
  onOpenChange,
}: {
  aid: string | null;
  datasetId: string | null;
  datasetName: string | null;
  charts: AvatarChart[];
  /** Controlled by Auto Insights so each tab applies its own default. */
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const ctx = useInsightsAvatar();
  const setOpen = onOpenChange;
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [portraitPulse, setPortraitPulse] = useState(0);
  const [firstName, setFirstName] = useState("");

  const messagesRef = useRef<Msg[]>([]);
  messagesRef.current = messages;
  const chartsRef = useRef(charts);
  chartsRef.current = charts;
  const abortRef = useRef<AbortController | null>(null);
  const highlightTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const boundaryRef = useRef(false);
  const logRef = useRef<HTMLDivElement | null>(null);
  const idRef = useRef(0);
  const sendRef = useRef<(q: string, focus?: string | null) => void>(() => {});

  const tts = useSpeechSynthesis();
  const stt = useSpeechRecognition({ onFinal: (t) => sendRef.current(t) });
  const { setHighlightedIds } = ctx;

  const state: AvatarState = pending ? "thinking" : stt.listening ? "listening" : tts.speaking ? "speaking" : "idle";

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const profileId = await getCurrentProfileId();
        if (!profileId) return;
        const { data } = await supabase.from("profiles").select("full_name").eq("id", profileId).maybeSingle();
        if (!cancelled) setFirstName((data?.full_name ?? "").trim().split(/\s+/)[0] ?? "");
      } catch {
        // Keep the generic greeting when the profile is unavailable.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const clearHighlight = useCallback(() => {
    if (highlightTimer.current) clearTimeout(highlightTimer.current);
    highlightTimer.current = null;
    setHighlightedIds([]);
  }, [setHighlightedIds]);

  const push = (role: Msg["role"], text: string, segments?: ReplySegment[] | null) => {
    idRef.current += 1;
    const msg = { id: idRef.current, role, text, segments };
    setMessages((m) => [...m, msg]);
  };

  const respond = (text: string, focusIds: string[], segments?: ReplySegment[] | null) => {
    push("assistant", text, segments);
    const ids = focusIds.filter(Boolean);
    if (ids.length) {
      setHighlightedIds(ids);
      try {
        const el = document.querySelector(`[data-insight-chart-id="${CSS.escape(ids[0])}"]`);
        el?.scrollIntoView({ behavior: "smooth", block: "center" });
      } catch {
        /* ignore */
      }
    }
    if (tts.muted || !tts.supported) {
      if (ids.length) highlightTimer.current = setTimeout(() => setHighlightedIds([]), 6000);
      return;
    }
    // Spoken without the "commentary — not checked" label; it is shown on screen.
    tts.speak(commentaryPlainText(text, segments), {
      onBoundary: () => {
        boundaryRef.current = true;
         setPortraitPulse((p) => p + 1);
      },
      onEnd: () => {
        clearHighlight();
      },
    });
  };

  const send = async (question: string, focus: string | null = ctx.focusedChartId) => {
    const q = question.trim().slice(0, 500);
    if (!q || pending) return;
    abortRef.current?.abort();
    tts.cancel();
    clearHighlight();
    const history = messagesRef.current.slice(-6).map((m) => ({ role: m.role, text: m.text }));
    push("user", q);
    setInput("");
    if (!aid) {
      respond(EXPLAIN_MESSAGES.unsaved, []);
      return;
    }
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setPending(true);
    try {
      const res = await explainInsights(
        aid,
        {
          question: q,
          dataset_id: datasetId,
          dataset_name: datasetName,
          focus_chart_id: focus ?? null,
          history,
          charts: toPayloadCharts(chartsRef.current),
        },
        ctrl.signal,
      );
      if (abortRef.current !== ctrl) return;
      respond(res.answer, res.focus_chart_ids, res.answer_segments);
    } catch (err: any) {
      if (err?.name === "AbortError" || abortRef.current !== ctrl) return;
      respond(err?.message || EXPLAIN_MESSAGES.network, []);
    } finally {
      if (abortRef.current === ctrl) {
        abortRef.current = null;
        setPending(false);
      }
    }
  };
  sendRef.current = (q, focus) => void send(q, focus);

  const closePanel = useCallback(() => {
    setOpen(false);
    abortRef.current?.abort();
    abortRef.current = null;
    setPending(false);
    tts.cancel();
    stt.abort();
    clearHighlight();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearHighlight, tts.cancel, stt.abort, setOpen]);

  // Full cleanup when the modal (and this component) unmounts.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (highlightTimer.current) clearTimeout(highlightTimer.current);
      setHighlightedIds([]);
    };
  }, [setHighlightedIds]);

  // Per-chart "Explain" buttons.
  useEffect(() => {
    const req = ctx.explainRequest;
    if (!req) return;
    setOpen(true);
    sendRef.current("Explain this chart", req.chartId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ctx.explainRequest?.nonce]);

  // Esc closes the panel before the modal's own Esc handler runs.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopImmediatePropagation();
      closePanel();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, closePanel]);

  // Visual pulse fallback when the voice never fires boundary events.
  useEffect(() => {
    if (!tts.speaking) return;
    boundaryRef.current = false;
    const t = setInterval(() => {
      if (!boundaryRef.current) setPortraitPulse((p) => p + 1);
    }, 400);
    return () => clearInterval(t);
  }, [tts.speaking]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, pending]);

  const toggleMic = () => {
    if (pending) return;
    if (stt.listening) {
      stt.stop();
      return;
    }
    if (tts.speaking) {
      tts.cancel();
      clearHighlight();
    }
    stt.start();
  };

  const stopSpeaking = () => {
    tts.cancel();
    clearHighlight();
  };

  const toggleMute = () => {
    const next = !tts.muted;
    tts.setMuted(next);
    if (next) stopSpeaking();
  };

  // Keep arrow keys typed in the panel from flipping modal slides.
  const stopArrowKeys = (e: ReactKeyboardEvent) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") e.stopPropagation();
  };

  const shownInput = stt.listening && stt.interim ? stt.interim : input;

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          title="Ask about these charts"
          aria-label="Open Insights Assistant"
          className={`absolute bottom-6 right-10 z-30 grid size-16 place-items-center overflow-visible rounded-full border border-border bg-card shadow-[0_12px_28px_-8px_rgba(26,95,232,0.45)] transition hover:-translate-y-0.5 ${FOCUS_RING}`}
        >
          <AvatarFace state="idle" size={56} />
        </button>
      )}

      {open && (
        <div
          aria-hidden="true"
          data-testid="insights-assistant-backdrop"
          onClick={closePanel}
          className="absolute inset-0 z-30 bg-foreground/30 xl:hidden"
        />
      )}

      {open && (
        <aside
          aria-label="Insights Assistant"
          onKeyDown={stopArrowKeys}
          className="absolute inset-x-0 bottom-0 z-40 flex h-[75%] flex-col rounded-t-2xl border-t border-border bg-card text-foreground shadow-[0_-12px_40px_-12px_rgba(16,24,40,0.3)] animate-in slide-in-from-bottom-6 duration-200 md:inset-x-auto md:right-0 md:top-0 md:h-full md:w-[380px] md:rounded-none md:border-t-0 md:border-l md:slide-in-from-right-6 xl:shadow-none"
        >
          <div className="flex items-center gap-2 border-b border-border px-4 py-3">
            <h3 className="flex-1 text-[15px] font-semibold text-foreground">Insights Assistant</h3>
            <button
              type="button"
              onClick={toggleMute}
              aria-label={tts.muted ? "Unmute voice" : "Mute voice"}
              aria-pressed={tts.muted}
              title={tts.muted ? "Unmute" : "Mute"}
              className={`grid size-8 place-items-center rounded-lg border border-border text-muted-foreground hover:bg-muted hover:text-foreground ${FOCUS_RING}`}
            >
              {tts.muted ? <VolumeX className="size-4" /> : <VolumeMax className="size-4" />}
            </button>
            <button
              type="button"
              onClick={closePanel}
              aria-label="Close assistant"
              className={`grid size-8 place-items-center rounded-lg border border-border text-muted-foreground hover:bg-muted hover:text-foreground ${FOCUS_RING}`}
            >
              <X className="size-4" />
            </button>
          </div>

          <div className="flex flex-col items-center px-4 pt-3">
            <AvatarFace state={state} pulse={portraitPulse} size={120} />
            {messages.length === 0 && (
              <p className="text-center text-lg font-semibold text-foreground">
                Hi {firstName || "there"}, ask me about these charts.
              </p>
            )}
            <p aria-live="polite" className="mt-1 text-xs font-medium text-muted-foreground">
              {STATUS[state]}
            </p>
          </div>

          <div ref={logRef} role="log" aria-label="Conversation" className="mt-3 min-h-0 flex-1 space-y-2 overflow-y-auto px-4 pb-2">
            {messages.map((m) => (
              <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                <p
                  className={
                    "max-w-[85%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-[13px] leading-5 " +
                    (m.role === "user"
                      ? "rounded-br-md bg-primary text-primary-foreground"
                      : "rounded-bl-md border border-border bg-muted text-foreground")
                  }
                >
                  {m.role === "assistant" ? <CommentaryText text={m.text} segments={m.segments} tone="light" /> : m.text}
                </p>
              </div>
            ))}
            {messages.length === 0 && (
              <div className="flex flex-wrap justify-center gap-2 pt-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    disabled={pending}
                    onClick={() => void send(s)}
                    className={`rounded-full border border-border bg-background px-3 py-1.5 text-xs font-medium text-primary hover:bg-muted disabled:opacity-50 ${FOCUS_RING}`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="border-t border-border px-4 py-3">
            {stt.error && <p className="mb-2 text-[11px] text-destructive">{stt.error}</p>}
            {stt.supported === false && (
              <p className="mb-2 text-[11px] text-muted-foreground">Voice input works in Chrome and Edge — you can type instead.</p>
            )}
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (!tts.speaking) void send(input);
              }}
            >
              {stt.supported && (
                <button
                  type="button"
                  onClick={toggleMic}
                  disabled={pending}
                  aria-pressed={stt.listening}
                  aria-label={stt.listening ? "Stop listening" : "Start voice input"}
                  className={
                    `grid size-11 shrink-0 place-items-center rounded-full text-primary-foreground shadow-sm transition disabled:opacity-50 ${FOCUS_RING} ` +
                    (stt.listening ? "bg-destructive hover:bg-destructive/90" : "bg-primary hover:bg-primary/90")
                  }
                >
                  <Microphone01 className="size-5" />
                </button>
              )}
              <input
                value={shownInput}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about the charts…"
                maxLength={500}
                aria-label="Ask about the charts"
                className="h-10 min-w-0 flex-1 rounded-lg border border-input bg-background px-3 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/20"
              />
              {tts.speaking ? (
                <button
                  type="button"
                  onClick={stopSpeaking}
                  className={`inline-flex h-10 shrink-0 items-center gap-1.5 rounded-lg border border-input bg-background px-3 text-sm font-semibold text-foreground hover:bg-muted ${FOCUS_RING}`}
                >
                  <StopSquare className="size-4" /> Stop
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={pending || !input.trim()}
                  aria-label="Send question"
                  className={`grid size-10 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground ${FOCUS_RING}`}
                >
                  <Send01 className="size-4" />
                </button>
              )}
            </form>
          </div>
        </aside>
      )}
    </>
  );
}
