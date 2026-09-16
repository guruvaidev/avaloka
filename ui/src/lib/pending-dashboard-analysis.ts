/** Session key for an analysis uploaded from /dashboard without a project. */
export const PENDING_DASHBOARD_ANALYSIS_KEY = "dashboard:pendingAnalysisId";

export function setPendingDashboardAnalysis(analysisId: string): void {
  sessionStorage.setItem(PENDING_DASHBOARD_ANALYSIS_KEY, analysisId);
}

export function getPendingDashboardAnalysis(): string | null {
  return sessionStorage.getItem(PENDING_DASHBOARD_ANALYSIS_KEY);
}

export function clearPendingDashboardAnalysis(): void {
  sessionStorage.removeItem(PENDING_DASHBOARD_ANALYSIS_KEY);
}
