// Client-safe constant: the single account allowed to run the support desk.
export const SUPPORT_ADMIN_EMAIL = "support@avaloka.ai";

/** True when the error means the support conversation table isn't provisioned. */
export function isMissingMessagesTable(err: unknown): boolean {
  const e = err as { code?: string; message?: string } | null;
  return (
    e?.code === "PGRST205" ||
    e?.code === "42P01" ||
    /support_ticket_messages/.test(String(e?.message ?? ""))
  );
}

export const SUPPORT_TABLE_MISSING_MESSAGE =
  "Support conversations aren't set up yet on this database. Run the support-conversations SQL to enable replies.";
