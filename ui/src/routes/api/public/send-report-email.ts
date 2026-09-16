import { createFileRoute } from "@tanstack/react-router";

const CORS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Max-Age": "86400",
};

const FUNCTION_URL = "https://dfhfytnflbybvgsifuam.supabase.co/functions/v1/send-report-email";

function json(body: Record<string, unknown>, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

export const Route = createFileRoute("/api/public/send-report-email")({
  server: {
    handlers: {
      OPTIONS: async () => new Response(null, { status: 204, headers: CORS }),
      POST: async ({ request }) => {
        const auth = request.headers.get("authorization");
        if (!auth?.startsWith("Bearer ")) return json({ error: "Unauthorized" }, 401);

        const body = await request.text();
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 120_000);
        let response: Response;
        try {
          response = await fetch(FUNCTION_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: auth },
            body,
            signal: controller.signal,
          });
        } catch (error) {
          const timedOut = error instanceof Error && error.name === "AbortError";
          return json(
            { error: timedOut ? "Report email timed out after 120 seconds" : "Email service is unavailable" },
            timedOut ? 504 : 502,
          );
        } finally {
          clearTimeout(timer);
        }

        const buf = await response.arrayBuffer();
        const headers: Record<string, string> = { ...CORS };
        const ct = response.headers.get("content-type");
        if (ct) headers["Content-Type"] = ct;
        const cd = response.headers.get("content-disposition");
        if (cd) headers["Content-Disposition"] = cd;
        return new Response(buf, { status: response.status, headers });

      },
    },
  },
});
