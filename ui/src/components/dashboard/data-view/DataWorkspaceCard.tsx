import type { ReactNode } from "react";

/** The workspace owns height; table and chart behavior stays in its children. */
export function DataWorkspaceCard({ children }: { children: ReactNode }) {
  return <section aria-label="Data View" className="data-workspace-card">{children}</section>;
}