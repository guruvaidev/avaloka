import { useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { Bell01, X } from "@untitledui/icons";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cx } from "@/lib/utils/cx";
import { useNotifications } from "@/hooks/useNotifications";
import { supabase } from "@/integrations/supabase/client";
import type { NotificationRow } from "@/lib/notifications";
import { Logomark } from "@/components/brand/Logomark";

type ActorInfo = { full_name: string | null; avatar_url: string | null };

const USER_INTERACTION_TYPES = new Set(["comment.created", "comment.reply", "comment.mention", "collaborator.invited", "share.created"]);

async function resolveAvatarUrl(value: string | null): Promise<string | null> {
  if (!value) return null;
  const publicMarker = "/storage/v1/object/public/avatars/";
  if (/^https?:\/\//i.test(value) && !value.includes(publicMarker)) return value;
  const rawPath = value.includes(publicMarker) ? value.split(publicMarker)[1] : value.replace(/^avatars\//, "");
  const path = decodeURIComponent(rawPath.split("?")[0] ?? "");
  if (!path) return null;
  const { data, error } = await supabase.storage.from("avatars").createSignedUrl(path, 60 * 60);
  if (error) return null;
  return data.signedUrl;
}

function initialsOf(name: string | null): string {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() ?? "").join("") || "?";
}

function relTime(iso: string): string {
  const then = new Date(iso).getTime();
  const diff = Math.max(0, Date.now() - then);
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(iso).toLocaleDateString();
}

export function NotificationsPopover() {
  const { items, unreadCount, markOne, markAll, removeOne, clearAll } = useNotifications();
  const navigate = useNavigate();
  const [actors, setActors] = useState<Record<string, ActorInfo>>({});

  useEffect(() => {
    let cancelled = false;
    const ids = Array.from(
      new Set(items.filter((n) => n.actor_id && USER_INTERACTION_TYPES.has(n.type)).map((n) => n.actor_id as string)),
    ).filter((id) => !(id in actors));
    if (ids.length === 0) return;
    (async () => {
      const { data } = await supabase.from("profiles").select("id, full_name, avatar_url").in("id", ids);
      if (cancelled || !data) return;
      const next: Record<string, ActorInfo> = {};
      await Promise.all(
        (data as Array<{ id: string; full_name: string | null; avatar_url: string | null }>).map(async (row) => {
          next[row.id] = {
            full_name: row.full_name,
            avatar_url: await resolveAvatarUrl(row.avatar_url),
          };
        }),
      );
      if (!cancelled) setActors((prev) => ({ ...prev, ...next }));
    })();
    return () => {
      cancelled = true;
    };
  }, [items, actors]);

  const handleClick = async (n: NotificationRow) => {
    if (!n.read_at) await markOne(n.id);
    if (n.link) {
      try {
        if (/^https?:\/\//i.test(n.link)) window.location.href = n.link;
        else navigate({ to: n.link });
      } catch {
        window.location.href = n.link;
      }
    }
  };

  return (
    <Popover>
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
      <PopoverContent
        side="right"
        align="end"
        sideOffset={12}
        className="w-[360px] p-0"
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="text-sm font-semibold text-foreground">Notifications</div>
          <div className="flex items-center gap-3">
            {unreadCount > 0 && (
              <button
                type="button"
                onClick={markAll}
                className="text-xs font-medium text-primary hover:underline"
              >
                Mark all as read
              </button>
            )}
            {items.length > 0 && (
              <button
                type="button"
                onClick={clearAll}
                className="text-xs font-medium text-muted-foreground hover:text-foreground hover:underline"
              >
                Clear all
              </button>
            )}
          </div>
        </div>
        <div className="max-h-[440px] overflow-y-auto">
          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center">
              <Bell01 className="size-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">You're all caught up.</p>
            </div>
          ) : (
            <ul className="divide-y divide-border">
              {items.map((n) => {
                const unread = !n.read_at;
                const isUserEvent = !!n.actor_id && USER_INTERACTION_TYPES.has(n.type);
                const actor = isUserEvent && n.actor_id ? actors[n.actor_id] : undefined;
                return (
                  <li key={n.id} className="group relative">
                    <button
                      type="button"
                      onClick={() => handleClick(n)}
                      className={cx(
                        "flex w-full items-start gap-3 px-4 py-3 pr-9 text-left transition hover:bg-accent/60",
                        unread && "bg-primary/5",
                      )}
                    >
                      <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full bg-muted">
                        {isUserEvent ? (
                          actor?.avatar_url ? (
                            <img
                              src={actor.avatar_url}
                              alt={actor.full_name ?? "User"}
                              className="size-full object-cover"
                            />
                          ) : (
                            <span className="text-[11px] font-semibold text-foreground">
                              {initialsOf(actor?.full_name ?? null)}
                            </span>
                          )
                        ) : (
                          <Logomark size={32} className="size-8" />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <div className="text-sm font-medium text-foreground">{n.title}</div>
                          {unread && <span className="inline-block size-2 shrink-0 rounded-full bg-primary" />}
                        </div>
                        {n.body && (
                          <div className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
                            {n.body}
                          </div>
                        )}
                        <div className="mt-1 text-[11px] text-muted-foreground">
                          {relTime(n.created_at)}
                        </div>
                      </div>
                    </button>
                    <button
                      type="button"
                      aria-label="Delete notification"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeOne(n.id);
                      }}
                      className="absolute right-2 top-2 inline-flex size-6 items-center justify-center rounded-md text-muted-foreground opacity-0 transition hover:bg-accent hover:text-foreground group-hover:opacity-100 focus:opacity-100"
                    >
                      <X className="size-3.5" />
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
