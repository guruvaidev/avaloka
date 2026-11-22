/** Auto Insights tabs that decide whether the Insights Assistant starts open. */
export type AssistantView = "stack" | "grid" | "columns";

/**
 * Default open state when a tab is shown: Stack opens the panel unless the user
 * already closed it in this Auto Insights view; Overview and Column statistics
 * start closed (only the floating avatar shows).
 */
export function assistantOpenOnView(view: AssistantView, dismissedInStack: boolean): boolean {
  return view === "stack" && !dismissedInStack;
}
