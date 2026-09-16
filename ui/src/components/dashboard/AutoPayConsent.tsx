import { cn } from "@/lib/utils";
import { AUTO_PAY_CONSENT_TEXT } from "@/lib/billing-details";

/**
 * Mandatory recurring-payment mandate. Shown in every checkout and
 * change-payment-method form (Stripe and PayPal alike).
 */
export function AutoPayConsent({
  checked,
  onChange,
  error,
  disabled,
  className,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  error?: string;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1", className)}>
      <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-border p-3 text-xs leading-relaxed text-muted-foreground">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-0.5 h-4 w-4 shrink-0 accent-[#1565EF]"
          aria-invalid={!!error}
        />
        <span>{AUTO_PAY_CONSENT_TEXT}</span>
      </label>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

export function FieldError({ message }: { message?: string }) {
  if (!message) return null;
  return <p className="mt-1 text-xs text-destructive">{message}</p>;
}
