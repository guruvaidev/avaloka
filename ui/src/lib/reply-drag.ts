/**
 * Drag-to-reply may start only from the dedicated drag handle, and never while
 * the user is selecting text (so message text stays selectable and copyable).
 */
export function shouldStartReplyDrag(input: { fromHandle: boolean; selectionText?: string | null; button?: number }): boolean {
  if (!input.fromHandle) return false;
  if (input.button !== undefined && input.button !== 0) return false;
  if (input.selectionText && input.selectionText.length > 0) return false;
  return true;
}

export function currentSelectionText(): string {
  try {
    return typeof window !== "undefined" ? (window.getSelection()?.toString() ?? "") : "";
  } catch {
    return "";
  }
}
