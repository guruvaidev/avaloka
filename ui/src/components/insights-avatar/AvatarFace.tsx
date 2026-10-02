export type AvatarState = "idle" | "listening" | "thinking" | "speaking";

import { avatarPortrait } from "./avatarPortrait";

const STYLES = `
@keyframes ia-breathe { 50% { transform: scale(1.02); } }
@keyframes ia-ring-expand { 0% { transform: scale(.9); opacity: .65; } 100% { transform: scale(1.19); opacity: 0; } }
@keyframes ia-dot { 0%, 80%, 100% { opacity: .35; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
@keyframes ia-wave { 0%, 100% { transform: scaleY(.35); } 50% { transform: scaleY(1); } }
@keyframes ia-glow { 50% { box-shadow: 0 0 18px 7px color-mix(in srgb, var(--color-brand-600) 30%, transparent); } }
@keyframes ia-speech-beat { 40% { transform: scale(1.03); box-shadow: 0 0 22px 9px color-mix(in srgb, var(--color-brand-600) 40%, transparent); } }
.ia-idle { animation: ia-breathe 4s ease-in-out infinite; }
.ia-ring { animation: ia-ring-expand 1.8s ease-out infinite; }
.ia-ring-second { animation-delay: .6s; }
.ia-spinner { animation: spin 1.4s linear infinite; }
.ia-dot { animation: ia-dot 1.2s ease-in-out infinite; }
.ia-wave { animation: ia-wave .6s ease-in-out infinite; }
.ia-speaking { animation: ia-glow .8s ease-in-out infinite; }
.ia-speech-beat { animation: ia-speech-beat .4s ease-out; }
.ia-speaking.ia-speech-beat { animation: ia-glow .8s ease-in-out infinite, ia-speech-beat .4s ease-out; }
@media (prefers-reduced-motion: reduce) {
  .ia-idle, .ia-ring, .ia-spinner, .ia-dot, .ia-wave, .ia-speaking, .ia-speech-beat { animation: none !important; transform: none !important; box-shadow: none !important; }
}
`;

export function AvatarFace({ state, pulse = 0, size = 120 }: { state: AvatarState; pulse?: number; size?: number }) {
  const compact = size <= 56;
  return (
    <div className={compact ? "relative flex size-14 shrink-0 items-center justify-center" : "relative flex h-[152px] w-[152px] shrink-0 items-start justify-center pt-4"}>
      <style>{STYLES}</style>
      <div className={compact ? "relative size-14" : "relative size-[120px]"}>
        {state === "listening" && !compact && (
          <>
            <span aria-hidden="true" className="ia-ring absolute inset-0 rounded-full border-2 border-brand-600 motion-reduce:hidden" />
            <span aria-hidden="true" className="ia-ring ia-ring-second absolute inset-0 rounded-full border-2 border-brand-600 motion-reduce:hidden" />
          </>
        )}
        {state === "thinking" && !compact && (
          <span aria-hidden="true" className="ia-spinner absolute -inset-1 rounded-full border-2 border-transparent border-t-brand-600 border-r-brand-600 motion-reduce:hidden" />
        )}
        <div
          key={state === "speaking" ? pulse : state}
          className={`relative size-full rounded-full ${state === "idle" ? "ia-idle" : state === "speaking" ? `ia-speaking ${pulse ? "ia-speech-beat" : ""}` : ""}`}
        >
          <img
            src={avatarPortrait}
            alt="Insights Assistant avatar"
            className={`size-full rounded-full border-2 object-cover object-top ${state === "idle" ? "border-border" : state === "thinking" ? "border-dashed border-brand-600" : "border-brand-600"}`}
          />
        </div>
      </div>
      {!compact && state === "thinking" && (
        <div aria-hidden="true" className="absolute bottom-1 flex gap-1.5">
          {[0, 1, 2].map((i) => <span key={i} className="ia-dot size-1.5 rounded-full bg-brand-600" style={{ animationDelay: `${i * .15}s` }} />)}
        </div>
      )}
      {!compact && state === "speaking" && (
        <div aria-hidden="true" className="absolute bottom-1 flex h-3 items-center gap-1">
          {[0, 1, 2, 3, 4].map((i) => <span key={i} className="ia-wave h-3 w-1 rounded-full bg-brand-600" style={{ animationDelay: `${i * .09}s` }} />)}
        </div>
      )}
    </div>
  );
}