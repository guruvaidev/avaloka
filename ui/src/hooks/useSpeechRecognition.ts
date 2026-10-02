import { useCallback, useEffect, useRef, useState } from "react";

type Recognition = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((e: any) => void) | null;
  onerror: ((e: any) => void) | null;
  onend: (() => void) | null;
};

function getCtor(): (new () => Recognition) | null {
  if (typeof window === "undefined") return null;
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export const MIC_BLOCKED_MESSAGE =
  "Microphone access is blocked. Allow it in your browser settings or type your question.";

export function useSpeechRecognition({ onFinal }: { onFinal: (text: string) => void }) {
  /** null until checked on the client, so SSR renders nothing speech-specific. */
  const [supported, setSupported] = useState<boolean | null>(null);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<Recognition | null>(null);
  const finalRef = useRef("");
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  useEffect(() => {
    setSupported(!!getCtor());
    return () => {
      const r = recRef.current;
      recRef.current = null;
      if (r) {
        r.onend = null;
        try {
          r.abort();
        } catch {
          /* ignore */
        }
      }
    };
  }, []);

  const stop = useCallback(() => {
    const r = recRef.current;
    if (!r) return;
    try {
      r.stop();
    } catch {
      /* ignore */
    }
  }, []);

  const abort = useCallback(() => {
    const r = recRef.current;
    recRef.current = null;
    finalRef.current = "";
    setInterim("");
    setListening(false);
    if (!r) return;
    r.onend = null;
    try {
      r.abort();
    } catch {
      /* ignore */
    }
  }, []);

  const start = useCallback(() => {
    if (recRef.current) return;
    const Ctor = getCtor();
    if (!Ctor) return;
    const r = new Ctor();
    r.lang = "en-IN";
    r.interimResults = true;
    r.continuous = false;
    finalRef.current = "";
    setInterim("");
    setError(null);

    r.onresult = (e: any) => {
      let finalText = "";
      let interimText = "";
      for (let i = 0; i < e.results.length; i += 1) {
        const res = e.results[i];
        const t = res[0]?.transcript ?? "";
        if (res.isFinal) finalText += t;
        else interimText += t;
      }
      if (finalText) finalRef.current = finalText.trim();
      setInterim((finalText + interimText).trim());
    };
    r.onerror = (e: any) => {
      if (e?.error === "not-allowed" || e?.error === "service-not-allowed") setError(MIC_BLOCKED_MESSAGE);
    };
    r.onend = () => {
      recRef.current = null;
      setListening(false);
      setInterim("");
      const text = finalRef.current;
      finalRef.current = "";
      if (text) onFinalRef.current(text);
    };
    try {
      r.start();
      recRef.current = r;
      setListening(true);
    } catch {
      recRef.current = null;
      setListening(false);
    }
  }, []);

  return { supported, listening, interim, error, start, stop, abort };
}
