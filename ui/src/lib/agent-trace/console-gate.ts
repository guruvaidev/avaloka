// The swarm console gate, as the UI reads it.
//
// The backend decides (app/api/swarm_gate.py): /api/capabilities carries
// `features.swarm_console`, and every swarm/trace route answers 403 with
// `detail.code === "swarm_console_unavailable"` outside Professional and
// Enterprise. The UI never reasons from the plan name — it reads the contract —
// so a self-hosted install, an unknown plan and a hosted Free user all behave
// the same way the API does. The swarm itself (`features.swarm`) is unaffected:
// Free users still get the mission told back in the chat.
import type { SwarmConsoleCapability } from "@/lib/agent-trace/types";

export const CONSOLE_GATE_CODE = "swarm_console_unavailable";

export type ConsoleState = "loading" | "open" | "closed";

/** What the capabilities document says about the console. Missing → closed. */
export function consoleState(caps: { features?: { swarm_console?: SwarmConsoleCapability | null } } | null | undefined): ConsoleState {
  const console = caps?.features?.swarm_console;
  return console?.enabled === true ? "open" : "closed";
}

/** The copy to show when the console is closed: the API's line, else ours. */
export function consoleMessage(caps: { features?: { swarm_console?: SwarmConsoleCapability | null } } | null | undefined): string {
  const message = caps?.features?.swarm_console?.message;
  return typeof message === "string" && message
    ? message
    : "The live swarm console is available in Avaloka Professional and Enterprise. Your swarm still runs here, and Avaloka tells you what the team did in the chat.";
}

/** True when a failed trace/swarm call was the gate, not the route. */
export function isConsoleGateError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { status?: number; code?: string; detail?: { code?: string } };
  return e.status === 403 && (e.code === CONSOLE_GATE_CODE || e.detail?.code === CONSOLE_GATE_CODE || e.code === undefined);
}
