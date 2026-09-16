import { createFileRoute } from "@tanstack/react-router";

import { BACKEND_API_BASE } from "@/lib/api/backend-config";

const DEFAULT_API_BASE = BACKEND_API_BASE;

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
  try {
    const path = request.headers.get("x-backend-path");
    if (!path || !path.startsWith("/")) {
      return jsonError("Missing or invalid X-Backend-Path header", 400);
    }
    const base = request.headers.get("x-backend-base") || DEFAULT_API_BASE;

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

    // Wait as long as the platform allows before giving up. Slow turns now
    // resolve asynchronously via /threads/{id}/pending-turn, so this timeout
    // is only a last-resort guard that yields a readable JSON error.
    const upstream = await fetch(`${base}${path}`, {
      method,
      headers,
      body,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const buf = await upstream.arrayBuffer();

    const respHeaders = new Headers(CORS);
    const upstreamCt = upstream.headers.get("content-type");
    if (upstreamCt) respHeaders.set("Content-Type", upstreamCt);

    return new Response(buf, { status: upstream.status, headers: respHeaders });
  } catch (err: any) {
    const timedOut = err?.name === "TimeoutError" || err?.name === "AbortError";
    console.error("[backend-proxy] upstream fetch failed", err);
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
