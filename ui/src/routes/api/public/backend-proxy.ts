import { createFileRoute } from "@tanstack/react-router";

import { BACKEND_API_BASE } from "@/lib/api/backend-config";

const DEFAULT_API_BASE = BACKEND_API_BASE;

const normBase = (u?: string | null) => (u ?? "").trim().replace(/\/+$/, "");

/**
 * Upstreams this proxy may forward to: the main backend plus the MCP services
 * (registration on MCP_API_BASE, tools on MCP_TOOL_BASE). A request picks one
 * with the X-Backend-Base header (see mcpProxyFetch in backendApi.ts); any
 * other value is ignored, so the proxy can never be pointed at an arbitrary host.
 */
function allowedBases(): string[] {
  return [DEFAULT_API_BASE, process.env.MCP_API_BASE, process.env.MCP_TOOL_BASE]
    .map(normBase)
    .filter(Boolean);
}

/** The upstream to use, or null when X-Backend-Base names a host that is not allowed. */
function resolveBase(requested: string | null): string | null {
  const wanted = normBase(requested);
  if (!wanted) return normBase(DEFAULT_API_BASE);
  return allowedBases().includes(wanted) ? wanted : null;
}

// Maximum time we wait on the upstream backend before returning the
// structured fallback body. Raised to the platform maximum (~10 min).
const UPSTREAM_TIMEOUT_MS = 600_000;

const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers":
    "Content-Type, Authorization, X-Avaloka-Session, X-Backend-Path, X-Backend-Base",
  "Access-Control-Max-Age": "86400",
};

function jsonError(message: string, status: number) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

async function forward(request: Request, method: string) {
  let targetUrl = "";
  try {
    const path = request.headers.get("x-backend-path");
    if (!path || !path.startsWith("/")) {
      console.error({
        proxyStage: "early-return",
        method,
        requestedPath: path,
        error: "Missing or invalid X-Backend-Path header",
      });
      return jsonError("Missing or invalid X-Backend-Path header", 400);
    }
    // Was always DEFAULT_API_BASE, so MCP calls (e.g. /customers/register)
    // went to the main backend and came back 404 Not Found.
    const base = resolveBase(request.headers.get("x-backend-base"));
    if (!base) {
      // Fail loudly rather than silently sending an MCP call to the main backend
      // (which answered 404 and hid a missing MCP_API_BASE / MCP_TOOL_BASE).
      console.error({
        proxyStage: "early-return",
        method,
        requestedPath: path,
        error: "X-Backend-Base is not an allowed upstream (check MCP_API_BASE / MCP_TOOL_BASE)",
      });
      return jsonError("X-Backend-Base is not an allowed upstream", 400);
    }

    const headers: Record<string, string> = {};
    const auth = request.headers.get("authorization");
    if (auth) headers["Authorization"] = auth;
    const session = request.headers.get("x-avaloka-session");
    if (session) headers["X-Avaloka-Session"] = session;
    // Forward content-type as-is (preserves multipart boundary for uploads)
    const ct = request.headers.get("content-type");
    if (ct) headers["Content-Type"] = ct;

    const body =
      method === "GET" || method === "DELETE" || method === "HEAD"
        ? undefined
        : await request.arrayBuffer();

    targetUrl = `${base}${path}`;
    // Log request metadata only. Never log the body: it can carry user
    // prompts, pasted data and connection credentials.
    console.log({
      proxyStage: "incoming",
      method,
      requestedPath: path,
      upstreamBaseUrl: base,
      contentType: ct ?? null,
      bodyBytes: body?.byteLength ?? 0,
    });

    const upstream = await fetch(targetUrl, {
      method,
      headers,
      body,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    console.log({ proxyStage: "upstream-response", targetUrl, status: upstream.status });
    const buf = await upstream.arrayBuffer();

    const respHeaders = new Headers(CORS);
    const upstreamCt = upstream.headers.get("content-type");
    if (upstreamCt) respHeaders.set("Content-Type", upstreamCt);

    return new Response(buf, { status: upstream.status, headers: respHeaders });
  } catch (err: any) {
    // Covers both an upstream timeout and an unreachable backend (no /health
    // pre-check: it cost an extra round-trip plus a LangGraph probe per call).
    const timedOut = err?.name === "TimeoutError" || err?.name === "AbortError";
    console.error({ proxyStage: "forwarding-error", targetUrl, error: String(err) });
    return new Response(
      JSON.stringify({
        error: timedOut ? "UPSTREAM_TIMEOUT" : "SERVICE_UNAVAILABLE",
        fallback: true,
        message: timedOut
          ? "The analysis backend took too long to respond. Please try again."
          : err?.message || String(err),
      }),
      { status: 200, headers: { "Content-Type": "application/json", ...CORS } },
    );
  }
}

export const Route = createFileRoute("/api/public/backend-proxy")({
  server: {
    handlers: {
      OPTIONS: async () => new Response(null, { status: 204, headers: CORS }),
      GET: async ({ request }) => forward(request, "GET"),
      POST: async ({ request }) => forward(request, "POST"),
      PUT: async ({ request }) => forward(request, "PUT"),
      DELETE: async ({ request }) => forward(request, "DELETE"),
    },
  },
});