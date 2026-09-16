export const AUTO_INSIGHTS_BOT_START_EVENT = "avaloka:auto-insights-bot-start";
export const AUTO_INSIGHTS_BOT_FINISH_EVENT = "avaloka:auto-insights-bot-finish";

const STORAGE_KEY = "analysis:auto-insights-bot-active";
let activeInMemory = false;

export function startAutoInsightsBot() {
  if (typeof window === "undefined") return;
  activeInMemory = true;
  try {
    window.sessionStorage.setItem(STORAGE_KEY, "true");
  } catch {
    /* The event still keeps the current page in sync when storage is unavailable. */
  }
  window.dispatchEvent(new CustomEvent(AUTO_INSIGHTS_BOT_START_EVENT));
}

export function finishAutoInsightsBot() {
  if (typeof window === "undefined") return;
  if (!isAutoInsightsBotActive()) return;
  activeInMemory = false;
  try {
    window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* The event still keeps the current page in sync when storage is unavailable. */
  }
  window.dispatchEvent(new CustomEvent(AUTO_INSIGHTS_BOT_FINISH_EVENT));
}

export function isAutoInsightsBotActive() {
  if (typeof window === "undefined") return false;
  try {
    return activeInMemory || window.sessionStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    return activeInMemory;
  }
}