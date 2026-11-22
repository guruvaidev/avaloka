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
    <div className="flex h-screen w-full bg-primary text-primary dark:bg-[#131316]">
      <IconRail onItemClick={onRailItemClick} />
      <AnalysisSidebar analyses={analyses} />
      <main className="relative my-3 mr-3 ml-3 flex flex-1 flex-col overflow-hidden rounded-l-xl rounded-r-none border border-border-secondary bg-primary dark:border-white/10 dark:bg-[#1A1A1E]">
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-full bg-[radial-gradient(150%_100%_at_42.83%_20.65%,#4680e400_43.87%,#1565EF_150%,#fff)] dark:bg-[radial-gradient(100%_80%_at_50%_100%,#1565EF66_0%,transparent_80%)]" />
        <div className="scrollbar-hide relative flex flex-1 flex-col overflow-y-auto">{children}</div>
      </main>
    </div>
  );
}
