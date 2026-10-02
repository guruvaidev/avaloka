import { useCallback, useEffect, useRef, useState } from "react";

const MUTE_KEY = "insightsAvatarMuted";

function hasSynth() {
  return typeof window !== "undefined" && "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined";
}

const FEMALE_HINTS = [
  "female",
  "neerja",
  "heera",
  "swara",
  "aditi",
  "veena",
  "raveena",
  "lekha",
  "kalpana",
  "zira",
  "susan",
  "samantha",
  "victoria",
  "karen",
  "moira",
  "tessa",
  "fiona",
  "kate",
  "serena",
  "allison",
  "ava",
  "joelle",
  "nicky",
];

function isFemaleVoice(v: SpeechSynthesisVoice): boolean {
  const name = v.name?.toLowerCase() ?? "";
  return FEMALE_HINTS.some((hint) => name.includes(hint));
}

function pickVoice(): SpeechSynthesisVoice | null {
  const voices = window.speechSynthesis.getVoices();
  const enIn = voices.filter((v) => v.lang?.toLowerCase().startsWith("en-in"));
  const en = voices.filter((v) => v.lang?.toLowerCase().startsWith("en-"));
  return (
    enIn.find(isFemaleVoice) ??
    en.find(isFemaleVoice) ??
    voices.find(isFemaleVoice) ??
    enIn[0] ??
    en[0] ??
    voices.find((v) => v.default) ??
    null
  );
}

function splitSentences(text: string): string[] {
  const parts = text.replace(/\s+/g, " ").trim().match(/[^.!?]+[.!?]+["')\]]*|[^.!?]+$/g) ?? [];
  return parts.map((p) => p.trim()).filter(Boolean);
}

export type SpeakCallbacks = { onBoundary?: () => void; onEnd?: () => void };

export function useSpeechSynthesis() {
  const [supported, setSupported] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [muted, setMutedState] = useState(false);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const tokenRef = useRef(0);

  useEffect(() => {
    try {
      setMutedState(window.localStorage.getItem(MUTE_KEY) === "true");
    } catch {
      /* storage unavailable */
    }
    if (!hasSynth()) return;
    setSupported(true);
    const load = () => {
      voiceRef.current = pickVoice();
    };
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", load);
      tokenRef.current += 1;
      try {
        window.speechSynthesis.cancel();
      } catch {
        /* ignore */
      }
    };
  }, []);

  const cancel = useCallback(() => {
    tokenRef.current += 1;
    setSpeaking(false);
    if (!hasSynth()) return;
    try {
      window.speechSynthesis.cancel();
    } catch {
      /* ignore */
    }
  }, []);

  const speak = useCallback((text: string, cb: SpeakCallbacks = {}) => {
    if (!hasSynth()) {
      cb.onEnd?.();
      return;
    }
    const sentences = splitSentences(text);
    tokenRef.current += 1;
    const token = tokenRef.current;
    try {
      window.speechSynthesis.cancel();
    } catch {
      /* ignore */
    }
    if (!sentences.length) {
      cb.onEnd?.();
      return;
    }
    const finish = () => {
      if (token !== tokenRef.current) return;
      setSpeaking(false);
      cb.onEnd?.();
    };
    const next = (i: number) => {
      if (token !== tokenRef.current) return;
      if (i >= sentences.length) return finish();
      const u = new SpeechSynthesisUtterance(sentences[i]);
      if (!voiceRef.current) voiceRef.current = pickVoice();
      if (voiceRef.current) {
        u.voice = voiceRef.current;
        u.lang = voiceRef.current.lang;
      }
      u.rate = 1;
      u.pitch = 1;
      u.onboundary = () => {
        if (token === tokenRef.current) cb.onBoundary?.();
      };
      u.onend = () => next(i + 1);
      u.onerror = () => finish();
      window.speechSynthesis.speak(u);
    };
    setSpeaking(true);
    next(0);
  }, []);

  const setMuted = useCallback((value: boolean) => {
    setMutedState(value);
    try {
      window.localStorage.setItem(MUTE_KEY, String(value));
    } catch {
      /* ignore */
    }
  }, []);

  return { supported, speaking, muted, setMuted, speak, cancel };
}
