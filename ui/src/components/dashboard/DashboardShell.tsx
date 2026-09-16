import type { ReactNode } from "react";
import { IconRail } from "./IconRail";
import { AnalysisSidebar, type AnalysisItem } from "./AnalysisSidebar";

export function DashboardShell({
  children,
  analyses,
  onRailItemClick,
}: {
  children: ReactNode;
  analyses?: AnalysisItem[];
  onRailItemClick?: (id: string) => void;
}) {
  return (
    <div className="flex h-screen w-full bg-primary text-primary">
      <IconRail active="chat" onItemClick={onRailItemClick} />
      <AnalysisSidebar analyses={analyses} />
      <main className="relative flex flex-1 flex-col overflow-hidden">
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-full bg-[radial-gradient(150%_100%_at_42.83%_20.65%,#4680e400_43.87%,#1565EF_150%,#fff)]" />
        <div className="scrollbar-hide relative flex flex-1 flex-col overflow-y-auto">{children}</div>
      </main>
    </div>
  );
}
