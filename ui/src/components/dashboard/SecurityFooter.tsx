import { ShieldTick } from "@untitledui/icons";

export function SecurityFooter() {
  return (
    <div className="flex items-center justify-center gap-2 border-t border-secondary bg-brand-primary/60 px-4 py-3 text-xs font-medium text-brand-secondary">
      <ShieldTick className="size-4" />
      <span>
        Cloud and service credentials are encrypted at rest. With a hosted model, your schema and
        sample rows are sent to the model provider you configure; a local model keeps them on your
        machine.
      </span>
    </div>
  );
}
