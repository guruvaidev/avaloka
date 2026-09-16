import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import type { TypingUser } from "@/hooks/useTypingPresence";

function initials(name: string) {
  const clean = name.trim();
  if (clean.includes("@")) return clean[0]?.toUpperCase() ?? "?";
  const parts = clean.split(/\s+/);
  return ((parts[0]?.[0] ?? "") + (parts[1]?.[0] ?? "")).toUpperCase() || "?";
}

export function TypingIndicator({ users }: { users: TypingUser[] }) {
  if (users.length === 0) return null;

  const shown = users.slice(0, 3);
  const extra = users.length - shown.length;

  return (
    <div className="mb-2 flex items-center gap-2">
      <div className="flex -space-x-2">
        {shown.map((u) => (
          <Avatar key={u.profileId} className="size-6 ring-2 ring-primary" title={u.name}>
            <AvatarImage src={u.avatarUrl ?? undefined} alt={u.name} className="object-cover" />
            <AvatarFallback className="bg-[#1565ef]/10 text-[10px] font-semibold text-[#1565ef]">
              {initials(u.name)}
            </AvatarFallback>
          </Avatar>
        ))}
        {extra > 0 && (
          <span className="grid size-6 place-items-center rounded-full bg-secondary text-[10px] font-semibold text-secondary ring-2 ring-primary">
            +{extra}
          </span>
        )}
      </div>
      <span className="flex items-center gap-0.5" aria-hidden>
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="size-1.5 animate-bounce rounded-full bg-fg-quaternary"
            style={{ animationDelay: `${delay}ms`, animationDuration: "1s" }}
          />
        ))}
      </span>
      <span className="sr-only" aria-live="polite">
        {users.map((u) => u.name).join(", ")} typing
      </span>
    </div>
  );
}
