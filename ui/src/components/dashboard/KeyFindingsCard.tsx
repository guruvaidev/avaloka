import { useEffect, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export const KEY_FINDINGS_OPEN_KEY = "avaloka.keyFindingsOpen";

export function readKeyFindingsOpen(storage?: Pick<Storage, "getItem"> | null): boolean {
  try {
    const s = storage ?? (typeof window !== "undefined" ? window.localStorage : null);
    return s?.getItem(KEY_FINDINGS_OPEN_KEY) !== "false";
  } catch {
    return true;
  }
}

export function writeKeyFindingsOpen(open: boolean, storage?: Pick<Storage, "setItem"> | null): void {
  try {
    const s = storage ?? (typeof window !== "undefined" ? window.localStorage : null);
    s?.setItem(KEY_FINDINGS_OPEN_KEY, String(open));
  } catch {
    /* session-only state */
  }
}

/** "Key findings" card under the chart carousel; bullets come from the same list the chat shows. */
export function KeyFindingsCard({ findings, children }: { findings?: string[]; children?: ReactNode }) {
  const [open, setOpen] = useState(true);
  useEffect(() => setOpen(readKeyFindingsOpen()), []);
  const toggle = () => {
    setOpen((prev) => {
      writeKeyFindingsOpen(!prev);
      return !prev;
    });
  };
  if (!children && !findings?.length) return null;
  return (
    <section aria-label="Key findings" className="mt-4 rounded-[12px] border border-border bg-card text-foreground">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left"
      >
        <span className="text-[14px] font-semibold text-foreground">Key findings</span>
        <ChevronDown className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>
      {open ? (
        <div className="border-t border-border px-4 py-3 text-sm text-foreground">
          {children ?? (
            <ul className="list-disc space-y-1.5 pl-5">
              {findings!.map((text) => (
                <li key={text}>{text}</li>
              ))}
            </ul>
          )}
        </div>
      ) : null}
    </section>
  );
}
