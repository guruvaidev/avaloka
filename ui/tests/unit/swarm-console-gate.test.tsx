// The swarm console gate as the UI reads it: the capabilities document decides,
// a missing or failed answer closes the door, and a direct hit on /swarm or
// /agent-trace shows the commercial-gate copy instead of the page.
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => <a href={to}>{children}</a>,
}));

import { SwarmConsoleGate } from "@/components/agent-trace/SwarmConsoleGate";
import { consoleMessage, consoleState, isConsoleGateError } from "@/lib/agent-trace/console-gate";

const open = { features: { swarm_console: { enabled: true, reason: "open", page: "/swarm", message: "The live swarm console is available on this edition." } } };
const closed = {
  features: {
    swarm: { enabled: true, limits: { max_clones_per_mission: 2 } },
    swarm_console: { enabled: false, reason: "commercial", page: "/swarm", message: "The live swarm console is available in Avaloka Professional and Enterprise. Your swarm still runs here, and Avaloka tells you what the team did in the chat." },
  },
};

describe("consoleState", () => {
  it("is open only when the API says enabled", () => {
    expect(consoleState(open)).toBe("open");
    expect(consoleState(closed)).toBe("closed");
  });
  it("treats a document without the block as closed (an older API, or no edition)", () => {
    expect(consoleState({ features: {} })).toBe("closed");
    expect(consoleState(null)).toBe("closed");
  });
  it("uses the API's own line for the gate copy", () => {
    expect(consoleMessage(closed)).toMatch(/Professional and Enterprise/);
    expect(consoleMessage(null)).toMatch(/Professional and Enterprise/);
  });
  it("recognises the gate's 403 and not a route's 404", () => {
    expect(isConsoleGateError({ status: 403, code: "swarm_console_unavailable" })).toBe(true);
    expect(isConsoleGateError({ status: 404 })).toBe(false);
  });
});

describe("SwarmConsoleGate", () => {
  it("renders the page for a paid caller", async () => {
    const api = { getCapabilities: vi.fn(async () => open) };
    render(
      <SwarmConsoleGate api={api}>
        <p>the live page</p>
      </SwarmConsoleGate>,
    );
    await waitFor(() => expect(screen.getByText("the live page")).toBeTruthy());
    expect(screen.queryByTestId("swarm-console-gate")).toBeNull();
  });

  it("shows the commercial-gate copy to a Free caller and never the page", async () => {
    const api = { getCapabilities: vi.fn(async () => closed) };
    render(
      <SwarmConsoleGate api={api} aid="a-1" title="Swarm">
        <p>the live page</p>
      </SwarmConsoleGate>,
    );
    await waitFor(() => expect(screen.getByTestId("swarm-console-gate")).toBeTruthy());
    expect(screen.queryByText("the live page")).toBeNull();
    expect(screen.getByText(/Professional and Enterprise/)).toBeTruthy();
    expect(screen.getByText("Back to analysis")).toBeTruthy();
  });

  it("closes the door when the capabilities call fails", async () => {
    const api = { getCapabilities: vi.fn(async () => { throw new Error("offline"); }) };
    render(
      <SwarmConsoleGate api={api}>
        <p>the live page</p>
      </SwarmConsoleGate>,
    );
    await waitFor(() => expect(screen.getByTestId("swarm-console-gate")).toBeTruthy());
    expect(screen.queryByText("the live page")).toBeNull();
  });
});
