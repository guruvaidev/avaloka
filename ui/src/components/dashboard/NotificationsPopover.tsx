import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { AlertCircle, Bell01, CheckCircle, Trash01 } from "@untitledui/icons";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cx } from "@/lib/utils/cx";
import { useNotifications } from "@/hooks/useNotifications";
import { supabase } from "@/integrations/supabase/client";
import type { NotificationRow } from "@/lib/notifications";


function relTime(iso: string): string {
  const diff = Math.max(0, Date.now() - new Date(iso).getTime());
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} hr ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d} day${d === 1 ? "" : "s"} ago`;
  return new Date(iso).toLocaleDateString();
}

const SUCCESS_TYPES = new Set(["analysis.completed", "training.completed", "batch_job.completed", "insights.generated"]);

function iconFor(type: string) {
  if (type.endsWith(".failed")) return { Icon: AlertCircle, cls: "bg-red-500/10 text-red-500" };
  if (SUCCESS_TYPES.has(type)) return { Icon: CheckCircle, cls: "bg-emerald-500/10 text-emerald-600" };
  if (type === "analysis.deleted") return { Icon: Trash01, cls: "bg-muted text-muted-foreground" };
  return { Icon: Bell01, cls: "bg-muted text-foreground" };
}

type NotificationMeta = { ref?: string; thread_id?: string; dataset_id?: string };

function metaOf(n: NotificationRow): NotificationMeta {
  const meta = (n as unknown as { metadata?: unknown }).metadata;
  return meta && typeof meta === "object" ? (meta as NotificationMeta) : {};
}

export function NotificationsPopover() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const { items, unreadCount, markOne, markAll } = useNotifications();

  const openTarget = async (n: NotificationRow) => {
    const link = n.link ?? "";
    const meta = metaOf(n);

    // Old form: /analysis?id=<id> -> /analysis?aid=<id>
    const idMatch = link.match(/^\/analysis\?id=([^&]+)/);
    if (idMatch) {
      navigate({ to: "/analysis", search: { aid: idMatch[1] } as never });
      return;
    }
    if (link.startsWith("/analysis?aid=")) {
      navigate({ to: link as never });
      return;
    }
    // Bare /analysis: resolve via metadata.thread_id when present.
    if (link === "/analysis" || link === "") {
      if (meta.thread_id) {
        const { data } = await supabase.from("analyses").select("id").eq("thread_id", meta.thread_id).limit(1);
        const aid = data?.[0]?.id;
        if (aid) {
          navigate({ to: "/analysis", search: { aid } as never });
          return;
        }
      }
      navigate({ to: "/analysis" });
      return;
    }
    // Any other internal link.
    if (link.startsWith("/")) {
      navigate({ to: link as never });
    } else if (/^https?:\/\//i.test(link)) {
      window.location.href = link;
    }
  };

  const handleClick = (n: NotificationRow) => {
    if (!n.read_at) void markOne(n.id); // don't wait before navigating
    setOpen(false);
    void openTarget(n);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="Notifications"
          className="relative inline-flex size-10 items-center justify-center rounded-lg text-fg-tertiary outline-none transition hover:bg-primary_hover hover:text-fg-primary focus-visible:outline-2 focus-visible:outline-focus-ring"
        >
          <Bell01 className="size-[18px]" />
          {unreadCount > 0 && (
            <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-red-500 px-1 text-[10px] font-semibold leading-none text-white">
              {unreadCount > 9 ? "9+" : unreadCount}
            </span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent side="right" align="end" sideOffset={12} className="flex max-h-[480px] w-[360px] flex-col p-0">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="text-sm font-semibold text-foreground">Notifications</div>
          <button
            type="button"
            onClick={() => void markAll()}
            disabled={unreadCount === 0}
            className="text-xs font-medium text-primary hover:underline disabled:cursor-not-allowed disabled:opacity-40 disabled:no-underline"
          >
            Mark all as read
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center">
              <Bell01 className="size-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">You're all caught up.</p>
            </div>
          ) : (
            <ul className="divide-y divide-border">
              {items.map((n) => {
                const unread = !n.read_at;
                const { Icon, cls } = iconFor(n.type);
                return (
                  <li key={n.id}>
                    <button
                      type="button"
                      onClick={() => handleClick(n)}
                      className={cx(
                        "flex w-full items-start gap-3 px-4 py-3 text-left transition hover:bg-accent/60",
                        unread && "bg-primary/5",
                      )}
                    >
                      <div
                        className={cx("mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full", cls)}
                      >
                        <Icon className="size-4" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <div className="truncate text-sm font-semibold text-foreground">{n.title}</div>
                          {unread && <span className="inline-block size-2 shrink-0 rounded-full bg-blue-500" />}
                        </div>
                        {n.body && <div className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">{n.body}</div>}
                        <div className="mt-1 text-[11px] text-muted-foreground">{relTime(n.created_at)}</div>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
