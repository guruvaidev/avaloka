// Main service for supabase/edge-runtime — the router the self-hosted stack needs.
//
// `supabase functions serve` (CLI) and the hosted platform both provide this
// dispatcher; the container image does not. It maps the first path segment to a
// function directory under /home/deno/functions and runs it in a user worker.
//
// Kong strips the /functions/v1 prefix, so a browser call to
//   /functions/v1/encrypt-mcp-credentials
// arrives here as /encrypt-mcp-credentials.

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const FUNCTIONS_ROOT = "/home/deno/functions";
const MEMORY_LIMIT_MB = 256;
const WORKER_TIMEOUT_MS = 60_000;

serve(async (req: Request) => {
  const { pathname } = new URL(req.url);
  const functionName = pathname.split("/").filter(Boolean)[0];

  if (!functionName) {
    return Response.json(
      { error: "missing function name in request path" },
      { status: 400 },
    );
  }

  const envVars = Object.entries(Deno.env.toObject());

  try {
    // deno-lint-ignore no-explicit-any
    const worker = await (globalThis as any).EdgeRuntime.userWorkers.create({
      servicePath: `${FUNCTIONS_ROOT}/${functionName}`,
      memoryLimitMb: MEMORY_LIMIT_MB,
      workerTimeoutMs: WORKER_TIMEOUT_MS,
      noModuleCache: false,
      importMapPath: null,
      envVars,
    });
    return await worker.fetch(req);
  } catch (err) {
    console.error(`worker for '${functionName}' failed:`, err);
    return Response.json(
      { error: String(err), function: functionName },
      { status: 500 },
    );
  }
});
