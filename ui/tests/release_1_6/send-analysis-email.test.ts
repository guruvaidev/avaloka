// @vitest-environment node
// UI-B05 (UI Bugs): "Send analysis by email" works. The UI posts to
// /api/public/send-analysis-email, which must forward the signed-in user's
// request to the mail function and relay its answer. Live delivery to a real
// mailbox stays a manual check.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({
  createFileRoute: () => (options: unknown) => ({ options }),
}));

type Handler = (ctx: { request: Request }) => Promise<Response>;
let post: Handler;
const fetchMock = vi.fn();

beforeEach(async () => {
  vi.resetModules();
  vi.stubEnv("SEND_ANALYSIS_EMAIL_URL", "https://mail.example.org/functions/v1/send-analysis-email");
  fetchMock.mockReset();
  // Default upstream answer, so a handler that wrongly calls it still returns a Response.
  fetchMock.mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const mod = (await import("@/routes/api/public/send-analysis-email")) as unknown as {
    Route: { options: { server: { handlers: { POST: Handler } } } };
  };
  post = mod.Route.options.server.handlers.POST;
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

const body = JSON.stringify({ analysis_id: "analysis-1", recipients: ["someone@example.com"], pdf_password: "secret-pw" });

describe("UI-B05 send analysis email route", () => {
  it("UI-B05: a signed-in request is forwarded to the mail function with its token and body, and the reply is relayed", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } }));
    const res = await post({
      request: new Request("http://ui.test/api/public/send-analysis-email", {
        method: "POST",
        headers: { Authorization: "Bearer user-token", "Content-Type": "application/json" },
        body,
      }),
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://mail.example.org/functions/v1/send-analysis-email");
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer user-token");
    expect(init.body).toBe(body);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
  });

  it("UI-B05: a request without a bearer token is refused with 401 and nothing is sent", async () => {
    const res = await post({
      request: new Request("http://ui.test/api/public/send-analysis-email", { method: "POST", body }),
    });
    expect(res.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("UI-B05: a failure from the mail function reaches the user as a failure, not a silent success", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ ok: false, error: "smtp down" }), { status: 502 }));
    const res = await post({
      request: new Request("http://ui.test/api/public/send-analysis-email", {
        method: "POST",
        headers: { Authorization: "Bearer user-token" },
        body,
      }),
    });
    expect(res.status).toBe(502);
    expect(await res.json()).toMatchObject({ ok: false, error: "smtp down" });
  });
});
