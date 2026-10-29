import type { ReactNode } from "react";
import { ArrowLeft, ArrowRight, LinkExternal01 } from "@untitledui/icons";
import { cx } from "@/lib/utils/cx";
import type { ChatDockSide } from "@/lib/chat-dock";

/**
 * The chat header's three dock-mode buttons: dock left, pop out (floating
 * card on the same page), dock right. Purely presentational — it never opens
 * windows and never touches storage; the caller owns the mode.
 */
export function ChatDockControls({
  mode,
  onChange,
  className,
}: {
  mode: ChatDockSide;
  onChange: (mode: ChatDockSide) => void;
  className?: string;
}) {
  const dockButton = (target: ChatDockSide, label: string, children: ReactNode) => (
    <button
      type="button"
      onClick={() => onChange(target)}
      aria-label={label}
      title={label}
      className={cx(
        "grid size-7 cursor-pointer place-items-center rounded-md text-fg-quaternary hover:bg-secondary hover:text-fg-secondary",
        mode === target && "bg-[#eef4ff] text-[#1a5fe0]",
      )}
    >
      {children}
    </button>
  );

  return (
    <div className={cx("flex items-center gap-0.5", className)}>
      {dockButton("left", "Dock chat left", <ArrowLeft className="size-4" />)}
      {dockButton("float", "Pop out chat", <LinkExternal01 className="size-4" />)}
      {dockButton("right", "Dock chat right", <ArrowRight className="size-4" />)}
    </div>
  );
}
