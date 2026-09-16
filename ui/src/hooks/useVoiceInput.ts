import { useCallback, useEffect, useRef, useState } from "react";

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
};

function getRecognitionCtor(): (new () => SpeechRecognitionLike) | null {
  if (typeof window === "undefined") return null;
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/**
 * Press-and-hold dictation built on the browser Web Speech API.
 * While held, `transcript` updates live (final + interim); on release the
 * caller receives the final text via `onFinal`.
 */
export function useVoiceInput(onFinal: (text: string) => void) {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const finalRef = useRef("");
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  useEffect(() => {
    setSupported(!!getRecognitionCtor());
    return () => {
      try {
        recognitionRef.current?.abort();
      } catch {
        /* ignore */
      }
    };
  }, []);

  const start = useCallback(() => {
    if (recognitionRef.current) return;
    const Ctor = getRecognitionCtor();
    if (!Ctor) return;

    const recognition = new Ctor();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    finalRef.current = "";
    setTranscript("");

    recognition.onresult = (event: any) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0]?.transcript ?? "";
        if (result.isFinal) finalRef.current = `${finalRef.current} ${text}`.trim();
        else interim += text;
      }
      setTranscript(`${finalRef.current} ${interim}`.trim());
    };
    recognition.onerror = () => {
      /* surfaced via empty transcript */
    };
    recognition.onend = () => {
      setListening(false);
    };

    try {
      recognition.start();
      recognitionRef.current = recognition;
      setListening(true);
    } catch {
      recognitionRef.current = null;
      setListening(false);
    }
  }, []);

  const stop = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    recognitionRef.current = null;
    try {
      recognition.stop();
    } catch {
      /* ignore */
    }
    setListening(false);
    const text = finalRef.current.trim() || transcript.trim();
    finalRef.current = "";
    setTranscript("");
    if (text) onFinalRef.current(text);
  }, [transcript]);

  return { supported, listening, transcript, start, stop };
}

const FEMALE_VOICE_HINTS = [
  "google uk english female",
  "google us english",
  "microsoft aria",
  "microsoft jenny",
  "microsoft zira",
  "samantha",
  "karen",
  "serena",
  "moira",
  "tessa",
  "victoria",
  "female",
];

/** Pick the most natural-sounding female voice available in this browser. */
function pickFemaleVoice(lang: string): SpeechSynthesisVoice | null {
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null;
  const base = lang.split("-")[0]?.toLowerCase() ?? "en";
  const matchesLang = (v: SpeechSynthesisVoice) => v.lang?.toLowerCase().startsWith(base);
  const candidates = voices.filter(matchesLang).length ? voices.filter(matchesLang) : voices;

  for (const hint of FEMALE_VOICE_HINTS) {
    const found = candidates.find((v) => v.name.toLowerCase().includes(hint));
    if (found) return found;
  }
  return candidates.find((v) => v.localService === false) ?? candidates[0] ?? null;
}

export type SpeakOptions = {
  /** Fired with the character index of the word currently being spoken. */
  onBoundary?: (charIndex: number) => void;
  onStart?: () => void;
  onEnd?: () => void;
};

/**
 * Speak text aloud with the browser speech synthesiser (natural female voice).
 * Returns the normalised text that is actually spoken so callers can align
 * word-highlighting with the utterance.
 */
export function speakText(text: string, options: SpeakOptions = {}): string {
  const clean = text.replace(/\s+/g, " ").trim().slice(0, 1000);
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return clean;
  if (!clean) return clean;
  try {
    window.speechSynthesis.cancel();
    const lang = navigator.language || "en-US";
    let spoken = false;

    // Word offsets used by the timer fallback (some voices never fire
    // `onboundary`, notably remote/network voices in Chrome).
    const wordStarts: number[] = [];
    const wordRegex = /\S+/g;
    let m: RegExpExecArray | null;
    while ((m = wordRegex.exec(clean))) wordStarts.push(m.index);

    const RATE = 0.85;
    const speak = () => {
      if (spoken) return;
      spoken = true;
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.lang = lang;
      const voice = pickFemaleVoice(lang);
      if (voice) utterance.voice = voice;
      utterance.rate = RATE;
      utterance.pitch = 1.1;

      let usedBoundary = false;
      let fallbackTimer: ReturnType<typeof setInterval> | null = null;
      let startedAt = 0;
      const stopFallback = () => {
        if (fallbackTimer) clearInterval(fallbackTimer);
        fallbackTimer = null;
      };

      utterance.onstart = () => {
        options.onStart?.();
        startedAt = Date.now();
        // ~165 wpm at rate 1 → per-word duration in ms.
        const perWord = (60_000 / 165) / RATE;
        fallbackTimer = setInterval(() => {
          if (usedBoundary) {
            stopFallback();
            return;
          }
          const index = Math.min(
            wordStarts.length - 1,
            Math.floor((Date.now() - startedAt) / perWord),
          );
          if (index >= 0) options.onBoundary?.(wordStarts[index]!);
        }, 80);
      };
      utterance.onboundary = (event: SpeechSynthesisEvent) => {
        if (event.name && event.name !== "word") return;
        usedBoundary = true;
        stopFallback();
        options.onBoundary?.(event.charIndex ?? 0);
      };
      utterance.onend = () => {
        stopFallback();
        options.onEnd?.();
      };
      utterance.onerror = () => {
        stopFallback();
        options.onEnd?.();
      };
      window.speechSynthesis.speak(utterance);
    };
    if (window.speechSynthesis.getVoices().length === 0) {
      window.speechSynthesis.addEventListener("voiceschanged", speak, { once: true });
      // Fallback in case the event never fires.
      setTimeout(speak, 250);
    } else {
      speak();
    }
  } catch {
    /* ignore */
  }

  return clean;
}


export function stopSpeaking() {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  try {
    window.speechSynthesis.cancel();
  } catch {
    /* ignore */
  }
}
