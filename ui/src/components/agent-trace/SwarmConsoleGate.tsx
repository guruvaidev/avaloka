import { useEffect, useState, type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { backendApi } from "@/lib/api/backendApi";
import { consoleMessage, consoleState, type ConsoleState } from "@/lib/agent-trace/console-gate";

type CapsApi = Pick<typeof backendApi, "getCapabilities">;

/** Asks the API once whether this caller may open the swarm console. */
export function useSwarmConsole(api: CapsApi = backendApi): { state: ConsoleState; message: string } {
  const [state, setState] = useState<ConsoleState>("loading");
  const [message, setMessage] = useState<string>(consoleMessage(null));
  useEffect(() => {
    let live = true;
    api
      .getCapabilities()
      .then((caps) => {
        if (!live) return;
        setState(consoleState(caps));
        setMessage(consoleMessage(caps));
      })
      // No answer is a closed door: the routes behind it would 403 anyway.
      .catch(() => live && setState("closed"));
    return () => {
      live = false;
    };
  }, [api]);
  return { state, message };
}

/** The commercial-gate copy for a console page hit directly. */
export function SwarmConsoleNotice({ message, aid, title = "Swarm" }: { message: string; aid?: string; title?: string }) {
  const linkClass =
    "inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover";
  return (
    <main className="min-w-0 flex-1 overflow-y-auto bg-primary" data-testid="swarm-console-gate">
      <div className="mx-auto max-w-3xl space-y-4 px-6 py-10">
        <h1 className="text-xl font-semibold text-primary">{title}</h1>
        <div className="rounded-xl border border-secondary bg-secondary px-4 py-3 text-sm text-secondary">{message}</div>
        <div className="flex items-center gap-2">
          <Link to="/settings" search={{ tab: "billing" } as never} className={linkClass}>
            See plans
          </Link>
          {aid ? (
            <Link to="/analysis" search={{ aid } as never} className={linkClass}>
              Back to analysis
            </Link>
          ) : null}
        </div>
      </div>
    </main>
  );
}

/**
 * Wraps a console page: renders it only once the API has said the console is
 * open for this caller; shows the gate copy otherwise. While the capabilities
 * load nothing is drawn, so a Free user never sees the page flash past.
 */
export function SwarmConsoleGate({ children, aid, title, api }: { children: ReactNode; aid?: string; title?: string; api?: CapsApi }) {
  const { state, message } = useSwarmConsole(api);
  if (state === "loading") return null;
  if (state === "closed") return <SwarmConsoleNotice message={message} aid={aid} title={title} />;
  return <>{children}</>;
}
