// Serial-number helpers for the subscriptions table.
// A serial is generated once, when a subscription row is first created, and is
// never rotated on renewals, plan changes, or provider updates.
import { randomBytes } from "crypto";

/** ENT-XXXX-XXXX-XXXX */
export function generateSubscriptionSerial(): string {
  const hex = randomBytes(6).toString("hex").toUpperCase();
  return `ENT-${hex.slice(0, 4)}-${hex.slice(4, 8)}-${hex.slice(8, 12)}`;
}

/**
 * Ensure an upsert payload carries a serial_number:
 *  - existing row with a serial  -> reuse it (never rotate)
 *  - existing row without one    -> backfill a fresh serial
 *  - no existing row             -> generate a fresh serial
 *
 * Silently no-ops if the column doesn't exist yet (schema drift safe).
 */
export async function attachSubscriptionSerial(
  supabase: any,
  payload: Record<string, any>,
  match: { column: string; value: string | null | undefined },
): Promise<Record<string, any>> {
  if (payload.serial_number) return payload;

  if (match.value) {
    try {
      const { data, error } = await supabase
        .from("subscriptions")
        .select("serial_number")
        .eq(match.column, match.value)
        .maybeSingle();
      // Column missing from the schema — leave the payload untouched.
      if (error && /serial_number/i.test(error.message ?? "")) return payload;
      if (data?.serial_number) {
        payload.serial_number = data.serial_number as string;
        return payload;
      }
    } catch (err) {
      console.warn("[subscription-serial] lookup failed", err);
    }
  }

  payload.serial_number = generateSubscriptionSerial();
  return payload;
}
