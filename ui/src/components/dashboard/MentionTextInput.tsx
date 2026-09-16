import { useEffect, useMemo, useRef, useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { cx } from "@/lib/utils/cx";
import type { MentionCandidate } from "@/lib/mentions.functions";

let cache: MentionCandidate[] | null = null;
let inflight: Promise<MentionCandidate[]> | null = null;

export function useMentionCandidates(): MentionCandidate[] {
  const [people, setPeople] = useState<MentionCandidate[]>(cache ?? []);

  useEffect(() => {
    if (cache) return;
    let cancelled = false;
    if (!inflight) {
      inflight = import("@/lib/mentions.functions")
        .then((m) => m.listMentionCandidates())
        .then((rows) => {
          cache = rows;
          return rows;
        })
        .catch(() => [] as MentionCandidate[]);
    }
    void inflight.then((rows) => {
      if (!cancelled) setPeople(rows);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return people;
}

/** Extract profile ids of people mentioned by "@Name" in the text. */
export function extractMentionedProfileIds(text: string, people: MentionCandidate[]): string[] {
  const lower = text.toLowerCase();
  const ids = new Set<string>();
  for (const p of people) {
    if (lower.includes(`@${p.name.toLowerCase()}`)) ids.add(p.profileId);
  }
  return [...ids];
}

type Props = {
  value: string;
  onChange: (next: string) => void;
  people: MentionCandidate[];
  placeholder?: string;
  rows?: number;
  multiline?: boolean;
  className?: string;
  onEnterSend?: () => void;
};

/** Text input with a lightweight "@" mention autocomplete. */
export function MentionTextInput({
  value,
  onChange,
  people,
  placeholder,
  rows = 2,
  multiline = true,
  className,
  onEnterSend,
}: Props) {
  const ref = useRef<HTMLTextAreaElement & HTMLInputElement>(null);
  const [query, setQuery] = useState<string | null>(null);
  const [active, setActive] = useState(0);

  const matches = useMemo(() => {
    if (query === null) return [];
    const q = query.toLowerCase();
    return people.filter((p) => p.name.toLowerCase().includes(q) || (p.email ?? "").toLowerCase().includes(q)).slice(0, 6);
  }, [query, people]);

  const open = query !== null && matches.length > 0;

  const syncQuery = (text: string, caret: number) => {
    const before = text.slice(0, caret);
    const at = before.lastIndexOf("@");
    if (at < 0) return setQuery(null);
    const prev = at > 0 ? before[at - 1] : " ";
    if (prev && !/\s/.test(prev)) return setQuery(null);
    const token = before.slice(at + 1);
    if (/[\n]/.test(token) || token.length > 30) return setQuery(null);
    setQuery(token);
    setActive(0);
  };

  const applyMention = (person: MentionCandidate) => {
    const el = ref.current;
    const caret = el?.selectionStart ?? value.length;
    const before = value.slice(0, caret);
    const at = before.lastIndexOf("@");
    if (at < 0) return;
    const next = `${value.slice(0, at)}@${person.name} ${value.slice(caret)}`;
    onChange(next);
    setQuery(null);
    requestAnimationFrame(() => {
      const pos = at + person.name.length + 2;
      el?.focus();
      el?.setSelectionRange(pos, pos);
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (open) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((i) => (i + 1) % matches.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((i) => (i - 1 + matches.length) % matches.length);
        return;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        applyMention(matches[active]!);
        return;
      }
      if (e.key === "Escape") {
        setQuery(null);
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey && onEnterSend) {
      e.preventDefault();
      onEnterSend();
    }
  };

  const shared = {
    ref: ref as never,
    value,
    placeholder,
    onKeyDown: handleKeyDown,
    onBlur: () => setTimeout(() => setQuery(null), 120),
    onChange: (e: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) => {
      onChange(e.target.value);
      syncQuery(e.target.value, e.target.selectionStart ?? e.target.value.length);
    },
  };

  return (
    <div className="relative flex-1">
      {multiline ? (
        <textarea {...shared} rows={rows} className={className} />
      ) : (
        <input {...shared} className={className} />
      )}
      {open ? (
        <ul className="absolute bottom-full left-0 z-50 mb-1 max-h-56 w-64 overflow-auto rounded-lg border border-secondary bg-primary py-1 shadow-lg">
          {matches.map((p, i) => (
            <li key={p.profileId}>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => applyMention(p)}
                className={cx(
                  "flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm",
                  i === active ? "bg-secondary/60 text-primary" : "text-secondary hover:bg-secondary/40",
                )}
              >
                <Avatar className="size-6">
                  <AvatarImage src={p.avatarUrl ?? undefined} alt={p.name} className="object-cover" />
                  <AvatarFallback className="bg-[#1565ef]/10 text-[10px] font-semibold text-[#1565ef]">
                    {p.name.slice(0, 2).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <span className="min-w-0 flex-1 truncate">{p.name}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
