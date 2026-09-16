import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { toast } from "sonner";
import { Send01, XClose } from "@untitledui/icons";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  getSupportThread,
  replyToSupportTicket,
  setSupportTicketStatus,
  type SupportMessage,
  type SupportTicket,
} from "@/lib/support-tickets.functions";

function formatWhen(value: string) {
  const d = new Date(value);
  return d.toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Clients only ever see open/closed; "answered" is a support-side state. */
export function visibleStatus(status: string, isAdmin = false) {
  const s = String(status || "open").toLowerCase();
  if (!isAdmin && s === "answered") return "open";
  return s;
}

export function SupportStatusPill({ status, isAdmin = false }: { status: string; isAdmin?: boolean }) {
  const s = visibleStatus(status, isAdmin);
  return (
    <span
      className={cn(
        "rounded-full px-2.5 py-1 text-xs font-medium capitalize",
        s === "closed" && "bg-muted text-muted-foreground",
        s === "answered" && "bg-brand-600/10 text-brand-600",
        s !== "closed" && s !== "answered" && "bg-amber-500/10 text-amber-600",
      )}
    >
      {s}
    </span>
  );
}


type Props = {
  ticketId: string;
  /** true when rendered for the support@avaloka.ai account */
  isAdmin?: boolean;
  onClose: () => void;
};

export function SupportThreadPanel({ ticketId, isAdmin = false, onClose }: Props) {
  const qc = useQueryClient();
  const [reply, setReply] = useState("");
  const [sending, setSending] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["support-thread", ticketId],
    queryFn: () => getSupportThread({ data: { ticketId } }),
    refetchInterval: 20_000,
  });

  const ticket = data?.ticket as SupportTicket | undefined;
  const messages = (data?.messages ?? []) as SupportMessage[];

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const send = async () => {
    const body = reply.trim();
    if (!body) return;
    setSending(true);
    try {
      await replyToSupportTicket({ data: { ticketId, body } });
      setReply("");
      await qc.invalidateQueries({ queryKey: ["support-thread", ticketId] });
      await qc.invalidateQueries({ queryKey: ["support-tickets"] });
    } catch (err) {
      console.error(err);
      toast.error("Could not send your reply. Please try again.");
    } finally {
      setSending(false);
    }
  };
  const changeStatus = async (status: string) => {
    try {
      await setSupportTicketStatus({ data: { ticketId, status } });
      await qc.invalidateQueries({ queryKey: ["support-thread", ticketId] });
      await qc.invalidateQueries({ queryKey: ["support-tickets"] });
    } catch (err) {
      console.error(err);
      toast.error("Could not update the status.");
    }
  };


  const modal = (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-4"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-3xl border border-border bg-card shadow-xl">
        <header className="flex items-start justify-between gap-4 border-b border-border px-6 py-5">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="truncate text-base font-semibold text-foreground">
                {ticket?.subject ?? "Support query"}
              </h2>
              {ticket && <SupportStatusPill status={ticket.status} isAdmin={isAdmin} />}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {ticket ? (
                <>
                  {ticket.category} · opened {formatWhen(ticket.created_at)}
                  {isAdmin && (ticket.requester_email || ticket.requester_name) && (
                    <> · {ticket.requester_name ?? ""} {ticket.requester_email ? `<${ticket.requester_email}>` : ""}</>
                  )}
                </>
              ) : (
                "Loading…"
              )}
            </p>
          </div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <XClose className="size-4" />
          </button>
        </header>

        <div className="flex-1 space-y-4 overflow-y-auto bg-background/40 px-6 py-5">
          {isLoading && <p className="text-sm text-muted-foreground">Loading conversation…</p>}

          {ticket && (
            <section className="rounded-2xl border border-border bg-card p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Query details
              </h3>
              <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 text-xs sm:grid-cols-2">
                <Detail label="Subject" value={ticket.subject} />
                <Detail label="Category" value={ticket.category} />
                <Detail label="Status" value={visibleStatus(ticket.status, isAdmin)} />
                <Detail label="Submitted" value={formatWhen(ticket.created_at)} />
                <Detail label="Last update" value={formatWhen(ticket.updated_at)} />
                <Detail label="Reference" value={`#${ticket.id.slice(0, 8).toUpperCase()}`} />
                {(ticket.requester_name || ticket.requester_email) && (
                  <Detail
                    label="Requester"
                    value={[ticket.requester_name, ticket.requester_email].filter(Boolean).join(" · ")}
                  />
                )}
              </dl>
              <div className="mt-3 border-t border-border pt-3">
                <p className="text-[11px] font-medium text-muted-foreground">Message</p>
                <p className="mt-1 whitespace-pre-wrap text-sm text-foreground">{ticket.message}</p>
              </div>
            </section>
          )}

          {ticket && (
            <p className="pt-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Conversation
            </p>
          )}

          {ticket && (
            <Bubble
              side={isAdmin ? "left" : "right"}
              author={isAdmin ? ticket.requester_name || ticket.requester_email || "Client" : "You"}
              when={ticket.created_at}
              body={ticket.message}
            />
          )}

          {messages.map((m) => {
            const fromSupport = m.author_role === "support";
            const mine = isAdmin ? fromSupport : !fromSupport;
            return (
              <Bubble
                key={m.id}
                side={mine ? "right" : "left"}
                author={fromSupport ? (isAdmin ? "You (Support)" : "Avaloka Support") : isAdmin ? "Client" : "You"}
                when={m.created_at}
                body={m.body}
                accent={fromSupport}
              />
            );
          })}
          <div ref={endRef} />
        </div>

        <footer className="border-t border-border px-6 py-4">
          <textarea
            value={reply}
            onChange={(e) => setReply(e.target.value)}
            rows={3}
            maxLength={4000}
            placeholder={isAdmin ? "Write a reply to the client…" : "Reply to support…"}
            className="w-full resize-y rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground focus:border-brand-600 focus:ring-4 focus:ring-brand-600/10"
          />
          <div className="mt-3 flex flex-wrap items-center justify-end gap-2">
            {isAdmin && ticket && (
              <select
                value={visibleStatus(ticket.status, true)}
                onChange={(e) => changeStatus(e.target.value)}
                className="h-10 rounded-xl border border-border bg-background px-3 text-sm text-foreground outline-none"
                aria-label="Ticket status"
              >
                <option value="open">Open</option>
                <option value="answered">Answered</option>
                <option value="closed">Closed</option>
              </select>
            )}
            <Button
              type="button"
              onClick={send}
              disabled={sending || !reply.trim()}
              className="h-10 gap-2 rounded-xl bg-brand-600 px-5 text-white hover:bg-brand-700"
            >
              <Send01 className="size-4" />
              {sending ? "Sending…" : "Send reply"}
            </Button>
          </div>
        </footer>
      </div>
    </div>
  );

  if (typeof document === "undefined") return modal;
  return createPortal(modal, document.body);
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="truncate text-sm font-medium text-foreground">{value}</dd>
    </div>
  );
}

function Bubble({
  side,
  author,
  when,
  body,
  accent,
}: {
  side: "left" | "right";
  author: string;
  when: string;
  body: string;
  accent?: boolean;
}) {
  return (
    <div className={cn("flex", side === "right" ? "justify-end" : "justify-start")}>
      <div className="max-w-[80%]">
        <div
          className={cn(
            "rounded-2xl border px-4 py-3 text-sm whitespace-pre-wrap",
            accent
              ? "border-brand-600/30 bg-brand-600/10 text-foreground"
              : "border-border bg-card text-foreground",
          )}
        >
          {body}
        </div>
        <p
          className={cn(
            "mt-1 text-[11px] text-muted-foreground",
            side === "right" && "text-right",
          )}
        >
          {author} · {formatWhen(when)}
        </p>
      </div>
    </div>
  );
}
