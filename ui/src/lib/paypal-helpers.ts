// Shared PayPal HTTP helpers. Server-only usage — always call inside a
// server function or server route handler (needs process.env).

export function getPayPalBase(): string {
  return process.env.PAYPAL_API_BASE || "https://api-m.sandbox.paypal.com";
}

export async function getPayPalAccessToken(): Promise<string> {
  const clientId = process.env.PAYPAL_CLIENT_ID;
  const clientSecret = process.env.PAYPAL_CLIENT_SECRET;
  if (!clientId || !clientSecret) {
    throw new Error("PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET are not configured");
  }
  const basic = Buffer.from(`${clientId}:${clientSecret}`).toString("base64");
  const res = await fetch(`${getPayPalBase()}/v1/oauth2/token`, {
    method: "POST",
    headers: {
      Authorization: `Basic ${basic}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: "grant_type=client_credentials",
  });
  const json = (await res.json()) as any;
  if (!res.ok || !json.access_token) {
    throw new Error(`PayPal token failed: ${res.status} ${JSON.stringify(json)}`);
  }
  return json.access_token as string;
}

export async function paypalFetch(
  path: string,
  init: RequestInit & { token?: string; debug?: boolean } = {},
): Promise<any> {
  const token = init.token || (await getPayPalAccessToken());
  const { token: _t, debug, headers, ...rest } = init as any;
  const method = (init.method ?? "GET").toUpperCase();
  const res = await fetch(`${getPayPalBase()}${path}`, {
    ...rest,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...(headers ?? {}),
    },
  });
  const text = await res.text();
  let json: any = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = { raw: text };
  }
  if (!res.ok) {
    console.error(`[paypal] ${method} ${path} FAILED ${res.status}`, {
      name: json?.name,
      message: json?.message,
      debug_id: json?.debug_id,
      details: json?.details,
    });
    const err: any = new Error(
      `PayPal ${method} ${path} failed: ${res.status} ${json?.name || ""} ${json?.message || ""} debug_id=${json?.debug_id || "n/a"}`,
    );
    err.paypal = json;
    err.status = res.status;
    throw err;
  }
  if (debug) {
    console.log(`[paypal] ${method} ${path} OK`, JSON.stringify(json));
  }
  return json;
}
