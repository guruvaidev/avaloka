import { getAccessToken, proxyFetch } from "./api/backendApi";
import { rayDashboardBase, rayJobUrl } from "./rayDashboard";

export class RayRequestError extends Error {
  constructor(public status: number, public reason?: string) { super(`Ray request failed (${status})`); }
}

async function request(path: string, method: "GET" | "POST", signal?: AbortSignal) {
  const token = await getAccessToken();
  const init: RequestInit = { method, headers: { Authorization: `Bearer ${token}` }, credentials: "include", signal };
  // GETs use the general proxy. POSTs (session, heartbeat) go to this site's own
  // /api/ray/*, which the UI's nginx forwards to the backend with Set-Cookie intact,
  // so the /ray cookie is stored on the same origin that serves /ray/.
  // (API_BASE is an in-cluster address; the browser cannot reach it.)
  const response = method === "GET"
    ? await proxyFetch(path, init)
    : await fetch(new URL(path, rayDashboardBase()), init);   // same host as /ray/, so its cookie lands there
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new RayRequestError(response.status, body?.reason);
  }
  return response;
}

export type RaySession = {
  state: "running" | "starting" | "stopped" | "stopping";
  url: string | null;
  message?: string;
  /** Minutes without activity in the tab before heartbeats stop; 0 = never. */
  idle_minutes?: number;
};

export const rayApi = {
  status: () => request("/api/ray/status", "GET"),
  session: async (signal?: AbortSignal): Promise<RaySession> =>
    (await request("/api/ray/session", "POST", signal)).json(),
  heartbeat: () => request("/api/ray/heartbeat", "POST"),
};

const POLL_MS = 5_000;
// A user's own cluster can take several minutes to schedule and start; the
// backend waits up to 10 minutes for one before a job falls back, so the tab
// waits as long.
const START_TIMEOUT_MS = 10 * 60_000;
const HEARTBEAT_MS = 60_000;
const DEFAULT_IDLE_MINUTES = 15;
const ACTIVITY_EVENTS = ["mousemove", "keydown", "click", "scroll"] as const;
let stopHeartbeat: (() => void) | undefined;
const pending = new Set<() => void>();

function heartbeat(tab: Window, idleMinutes: number) {
  stopHeartbeat?.();
  const idleMs = idleMinutes > 0 ? idleMinutes * 60_000 : Number.POSITIVE_INFINITY;
  let lastActivity = Date.now();
  let inFlight = false;
  const activity = () => { lastActivity = Date.now(); };
  for (const event of ACTIVITY_EVENTS) window.addEventListener(event, activity, { passive: true });
  const stop = () => {
    window.clearInterval(timer);
    for (const event of ACTIVITY_EVENTS) window.removeEventListener(event, activity);
    if (stopHeartbeat === stop) stopHeartbeat = undefined;
  };
  const timer = window.setInterval(() => {
    if (tab.closed || Date.now() - lastActivity >= idleMs) { stop(); return; }
    if (inFlight) return;
    inFlight = true;
    void rayApi.heartbeat().catch((error) => {
      if (error instanceof RayRequestError && (error.status === 401 || error.status === 403)) stop();
    }).finally(() => { inFlight = false; });
  }, HEARTBEAT_MS);
  stopHeartbeat = stop;
}

export function stopRayDashboardSessions() {
  stopHeartbeat?.();
  for (const cancel of pending) cancel();
}

function writeStatus(tab: Window, text: string) {
  tab.document.title = "Ray dashboard";
  tab.document.body.textContent = text;
}

/** Called directly from the user's click, before any asynchronous work. */
export function openRayDashboard(onError: (error: unknown) => void, jobId?: string) {
  const tab = window.open("about:blank", "_blank");
  if (!tab) { onError(new Error("Please allow pop-ups to open Ray dashboard.")); return; }
  tab.opener = null;
  const controller = new AbortController();
  let polling: number | undefined;
  let finished = false;
  const cleanup = () => {
    finished = true;
    controller.abort();
    window.clearTimeout(deadline);
    window.clearInterval(closedCheck);
    if (polling !== undefined) window.clearTimeout(polling);
    pending.delete(cancel);
  };
  const cancel = () => { cleanup(); tab.close(); };
  const deadline = window.setTimeout(() => {
    cleanup();
    if (!tab.closed) writeStatus(tab, "Your Ray cluster did not start. Try again from Avaloka.");
  }, START_TIMEOUT_MS);
  const closedCheck = window.setInterval(() => { if (tab.closed) cleanup(); }, 1_000);
  pending.add(cancel);
  const startedAt = Date.now();
  const attempt = async () => {
    try {
      const session = await rayApi.session(controller.signal);
      if (finished || tab.closed) return;
      if (session.state === "running") {
        cleanup();
        tab.location.href = jobId ? rayJobUrl(jobId) : rayDashboardBase();
        heartbeat(tab, session.idle_minutes ?? DEFAULT_IDLE_MINUTES);
      } else if (session.state === "starting" || session.state === "stopped" || session.state === "stopping") {
        const waited = Math.round((Date.now() - startedAt) / 1000);
        const text = session.message || "Starting your Ray cluster. This usually takes a few minutes.";
        writeStatus(tab, `${text} (${Math.floor(waited / 60)}m ${waited % 60}s)`);
        polling = window.setTimeout(() => void attempt(), POLL_MS);
      } else { throw new Error("Could not open Ray dashboard. Try again from Avaloka."); }
    } catch (error) {
      if (finished) return;
      cleanup();
      tab.close();
      onError(error);
    }
  };
  void attempt();
}