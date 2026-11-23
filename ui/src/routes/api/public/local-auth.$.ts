import { createFileRoute } from "@tanstack/react-router";

import { BACKEND_API_BASE } from "@/lib/api/backend-config";
import { isLocalAuth } from "@/integrations/supabase/config";

// AUTH_MODE=local only: the browser's Supabase client points here
// (integrations/supabase/config.ts), and this forwards each call to the
// API's local-auth routes (app/api/local_auth_routes.py). Same origin, so
// the browser needs neither CORS nor a second port; the API stays reachable
// from the UI server alone. In any other mode this route is a 404.

const UPSTREAM_TIMEOUT_MS = 30_000;

async function forward(request: Request, splat: string | undefined) {
  if (!isLocalAuth()) {
    return new Response(JSON.stringify({ msg: "Local auth mode is not enabled." }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }
  const search = new URL(request.url).search;
  const target = `${BACKEND_API_BASE}/local-auth/${splat ?? ""}${search}`;
  const headers: Record<string, string> = {};
  const auth = request.headers.get("authorization");
  if (auth) headers["Authorization"] = auth;
  const ct = request.headers.get("content-type");
  if (ct) headers["Content-Type"] = ct;
  const method = request.method.toUpperCase();
  const body = method === "GET" || method === "HEAD" ? undefined : await request.arrayBuffer();
  try {
    const upstream = await fetch(target, {
      method,
      headers,
      body,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const out = new Headers();
    for (const h of ["content-type", "content-range"]) {
      const v = upstream.headers.get(h);
      if (v) out.set(h, v);
    }
    // Never log bodies here: a sign-in carries the password.
    return new Response(upstream.status === 204 ? null : await upstream.arrayBuffer(), {
      status: upstream.status,
      headers: out,
    });
  } catch {
    return new Response(
      JSON.stringify({ code: 503, msg: "The Avaloka API is not reachable from the UI server." }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }
}

export const Route = createFileRoute("/api/public/local-auth/$")({
  server: {
    handlers: {
      GET: async ({ request, params }) => forward(request, params._splat),
      HEAD: async ({ request, params }) => forward(request, params._splat),
      POST: async ({ request, params }) => forward(request, params._splat),
      PUT: async ({ request, params }) => forward(request, params._splat),
      PATCH: async ({ request, params }) => forward(request, params._splat),
      DELETE: async ({ request, params }) => forward(request, params._splat),
    },
  },
});
