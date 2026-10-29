import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { CHAT_WIDTH_MIN, CHAT_WIDTH_STEP, clampChatWidth, maxChatWidth } from "@/lib/chat-width";
import { cn } from "@/lib/utils";

/**
 * Vertical separator between the data area and the docked chat. `chatSide` is
 * where the chat sits, so dragging toward the data area always widens the chat.
 */
export function ChatResizeHandle({
  width,
  viewport,
  chatSide,
  onResize,
  onReset,
}: {
  width: number;
  viewport: number;
  chatSide: "left" | "right";
  onResize: (width: number) => void;
  onReset: () => void;
}) {
  const startRef = useRef<{ x: number; width: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const sign = chatSide === "right" ? -1 : 1;

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    startRef.current = { x: e.clientX, width };
    setDragging(true);
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* capture unsupported */
    }
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!startRef.current) return;
    const delta = (e.clientX - startRef.current.x) * sign;
    onResize(clampChatWidth(startRef.current.width + delta, viewport));
  };
  const end = () => {
    startRef.current = null;
    setDragging(false);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const direction = e.key === "ArrowLeft" ? -1 : 1;
    onResize(clampChatWidth(width + direction * sign * CHAT_WIDTH_STEP, viewport));
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize chat panel"
      aria-valuenow={Math.round(width)}
      aria-valuemin={CHAT_WIDTH_MIN}
      aria-valuemax={maxChatWidth(viewport)}
      tabIndex={0}
      data-dragging={dragging || undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
      className="group relative hidden w-1.5 shrink-0 cursor-col-resize touch-none select-none self-stretch outline-none lg:block"
    >
      <span
        aria-hidden
        className={cn(
          "absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border transition-colors group-hover:bg-primary group-focus-visible:bg-primary",
          dragging && "bg-primary",
        )}
      />
    </div>
  );
}
