import type { FC } from "react";
import {
  MessageChatCircle,
  PieChart03,
  Database01,
  ClockRewind,
  BarChart03,
} from "@untitledui/icons";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { Logomark } from "@/components/brand/Logomark";
import { cx } from "@/lib/utils/cx";
import { useNavigate } from "@tanstack/react-router";
import { ProfileAvatarMenu } from "./IconRail";

type RailItem = { id: string; icon: FC<{ className?: string }>; label: string };

const items: RailItem[] = [
  { id: "chat", icon: MessageChatCircle, label: "Analysis" },
  { id: "projects", icon: PieChart03, label: "Dashboard" },
  { id: "sources", icon: Database01, label: "Dataset" },
  { id: "restore", icon: ClockRewind, label: "Restore" },
  { id: "reports", icon: BarChart03, label: "Insights" },
];

export function AnalysisIconRail({
  active = "chat",
  onItemClick,
}: {
  active?: string;
  onItemClick?: (id: string) => void;
}) {
  const navigate = useNavigate();

  return (
    <aside className="flex h-full w-[64px] shrink-0 flex-col items-center border-r border-secondary bg-primary py-3">
      <button
        type="button"
        aria-label="Go to analysis"
        onClick={() => navigate({ to: "/proanalysis" })}
        className="mb-4 outline-none"
      >
        <Logomark size={36} />
      </button>

      <div className="flex flex-col items-center gap-1">
        {items.map((item) => {
          const isActive = active === item.id;
          return (
            <ButtonUtility
              key={item.id}
              size="sm"
              color="tertiary"
              icon={item.icon}
              tooltip={item.label}
              tooltipPlacement="right"
              onClick={() => onItemClick?.(item.id)}
              className={cx(
                "size-10 rounded-lg *:data-icon:size-[18px]",
                isActive && "bg-primary_hover text-fg-primary",
              )}
            />
          );
        })}
      </div>

      <div className="relative mt-auto">
        <ProfileAvatarMenu />
      </div>
    </aside>
  );
}
