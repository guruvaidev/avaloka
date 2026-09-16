import { ShieldTick } from "@untitledui/icons";

export function SecurityFooter() {
  return (
    <div className="flex items-center justify-center gap-2 border-t border-secondary bg-brand-primary/60 px-4 py-3 text-xs font-medium text-brand-secondary">
      <ShieldTick className="size-4" />
      <span>Your data is encrypted, securely processed, and never shared outside your workspace.</span>
    </div>
  );
}
