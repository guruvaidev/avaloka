// Generic SMTP mail sender (Office365 / STARTTLS) with raw-protocol error reporting.
// Called server-side only: requires the service-role key as bearer token,
// except for `{ diagnose: true }` which only performs a login handshake and
// returns the exact server reply (no secrets are ever returned).

const cors = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Max-Age": "86400",
};

function json(body: Record<string, unknown>, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, "Content-Type": "application/json" },
  });
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

class SmtpError extends Error {
  constructor(public stage: string, public reply: string) {
    super(`${stage}: ${reply}`);
  }
}

/** Minimal SMTP client that surfaces the raw server replies. */
class RawSmtp {
  private conn!: Deno.Conn;
  private reader!: ReadableStreamDefaultReader<Uint8Array>;
  private buffer = "";
  private encoder = new TextEncoder();
  private decoder = new TextDecoder();
  transcript: string[] = [];

  async connect(hostname: string, port: number) {
    this.conn = await Deno.connect({ hostname, port });
    this.attach();
    await this.expect("greeting", 220);
  }

  private attach() {
    this.reader = this.conn.readable.getReader();
  }

  private async readLine(): Promise<string> {
    while (!this.buffer.includes("\r\n")) {
      const { value, done } = await this.reader.read();
      if (done) throw new SmtpError("connection", "server closed the connection");
      this.buffer += this.decoder.decode(value, { stream: true });
    }
    const idx = this.buffer.indexOf("\r\n");
    const line = this.buffer.slice(0, idx);
    this.buffer = this.buffer.slice(idx + 2);
    return line;
  }

  private async readReply(): Promise<{ code: number; text: string }> {
    const lines: string[] = [];
    let line = await this.readLine();
    lines.push(line);
    while (line.length >= 4 && line[3] === "-") {
      line = await this.readLine();
      lines.push(line);
    }
    const text = lines.join("\n");
    this.transcript.push(`S: ${text}`);
    return { code: Number(text.slice(0, 3)), text };
  }

  private async expect(stage: string, ...ok: number[]) {
    const { code, text } = await this.readReply();
    if (!ok.includes(code)) throw new SmtpError(stage, text);
    return text;
  }

  async cmd(stage: string, command: string, redact = false, ...ok: number[]) {
    this.transcript.push(`C: ${redact ? "<redacted>" : command}`);
    await this.conn.write(this.encoder.encode(command + "\r\n"));
    return await this.expect(stage, ...ok);
  }

  async startTls(hostname: string) {
    await this.cmd("starttls", "STARTTLS", false, 220);
    this.reader.releaseLock();
    this.conn = await Deno.startTls(this.conn, { hostname });
    this.buffer = "";
    this.attach();
  }

  async dataBody(body: string) {
    await this.cmd("data", "DATA", false, 354);
    const escaped = body.replace(/\r?\n/g, "\r\n").replace(/\r\n\./g, "\r\n..");
    this.transcript.push("C: <message body>");
    await this.conn.write(this.encoder.encode(escaped + "\r\n.\r\n"));
    return await this.expect("message", 250);
  }

  close() {
    try {
      this.reader.releaseLock();
    } catch { /* ignore */ }
    try {
      this.conn.close();
    } catch { /* ignore */ }
  }
}

function b64(v: string) {
  return btoa(v);
}

function config() {
  const hostname = Deno.env.get("SMTP_HOST") ?? "";
  const username = Deno.env.get("SMTP_USERNAME") ?? "";
  const password = Deno.env.get("SMTP_PASSWORD") ?? "";
  const port = Number(Deno.env.get("SMTP_PORT") ?? "587");
  const configuredFrom = (Deno.env.get("EMAIL_FROM") || username).trim();
  const angleAddress = configuredFrom.match(/<\s*([^<>\s]+@[^<>\s]+)\s*>/i)?.[1];
  const plainAddress = configuredFrom.match(/[^\s<>,;]+@[^\s<>,;]+/)?.[0];
  const fromAddress = angleAddress || plainAddress || username.trim();
  const fromHeader = EMAIL_RE.test(configuredFrom) ? configuredFrom : `Avaloka <${fromAddress}>`;
  return { hostname, username, password, port, fromAddress, fromHeader };
}

async function login(client: RawSmtp, hostname: string, port: number, username: string, password: string) {
  await client.connect(hostname, port);
  await client.cmd("ehlo", "EHLO avaloka.ai", false, 250);
  if (port !== 465) {
    await client.startTls(hostname);
    await client.cmd("ehlo-tls", "EHLO avaloka.ai", false, 250);
  }
  await client.cmd("auth", "AUTH LOGIN", false, 334);
  await client.cmd("auth-user", b64(username), true, 334);
  await client.cmd("auth-pass", b64(password), true, 235);
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
  if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

  let body: {
    to?: string | string[];
    subject?: string;
    html?: string;
    text?: string;
    diagnose?: boolean;
    attachments?: { filename?: string; contentType?: string; base64?: string }[];
  } = {};

  try {
    body = await req.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  const cfg = config();
  if (!cfg.hostname || !cfg.username || !cfg.password) {
    return json({ error: "SMTP is not configured" }, 500);
  }

  // --- diagnostics: handshake only, returns the exact server reply ---
  if (body.diagnose) {
    const client = new RawSmtp();
    try {
      await login(client, cfg.hostname, cfg.port, cfg.username, cfg.password);
      client.close();
      return json({
        ok: true,
        host: cfg.hostname,
        port: cfg.port,
        user: cfg.username,
        message: "SMTP authentication succeeded",
      });
    } catch (err) {
      client.close();
      const e = err as SmtpError;
      return json({
        ok: false,
        host: cfg.hostname,
        port: cfg.port,
        user: cfg.username,
        stage: e.stage ?? "unknown",
        serverReply: e.reply ?? String(err),
        transcript: client.transcript,
      }, 200);
    }
  }

  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
  const internalToken = Deno.env.get("INTERNAL_MAIL_TOKEN") ?? "";
  const auth = req.headers.get("authorization") ?? "";
  const presented = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  const allowed = [serviceKey, internalToken].filter(Boolean);
  if (!presented || !allowed.includes(presented)) {
    return json({ error: "Unauthorized" }, 401);
  }

  const recipients = (Array.isArray(body.to) ? body.to : [body.to])
    .map((v) => String(v ?? "").trim().toLowerCase())
    .filter((v) => EMAIL_RE.test(v));
  if (!recipients.length) return json({ error: "No valid recipients" }, 400);
  const subject = String(body.subject ?? "").slice(0, 200);
  if (!subject) return json({ error: "Subject is required" }, 400);

  const client = new RawSmtp();
  try {
    await login(client, cfg.hostname, cfg.port, cfg.username, cfg.password);
    if (!EMAIL_RE.test(cfg.fromAddress)) {
      return json({ error: "SMTP sender address is invalid", stage: "configuration" }, 500);
    }
    await client.cmd("mail-from", `MAIL FROM:<${cfg.fromAddress}>`, false, 250);
    for (const rcpt of recipients) {
      await client.cmd("rcpt-to", `RCPT TO:<${rcpt}>`, false, 250, 251);
    }
    const isHtml = !!body.html;
    const bodyContent = isHtml ? (body.html ?? "") : (body.text ?? "");
    const attachments = (body.attachments ?? [])
      .filter((a) => a && typeof a.base64 === "string" && a.base64.length > 0)
      .slice(0, 5);
    const headers = [
      `From: ${cfg.fromHeader}`,
      `To: ${recipients.join(", ")}`,
      `Subject: ${subject}`,
      "MIME-Version: 1.0",
    ];
    let message: string;
    if (attachments.length === 0) {
      message = [
        ...headers,
        `Content-Type: ${isHtml ? "text/html" : "text/plain"}; charset=utf-8`,
        "",
        bodyContent,
      ].join("\n");
    } else {
      const boundary = `=_avaloka_${crypto.randomUUID().replace(/-/g, "")}`;
      const parts: string[] = [
        `--${boundary}`,
        `Content-Type: ${isHtml ? "text/html" : "text/plain"}; charset=utf-8`,
        "",
        bodyContent,
        "",
      ];
      for (const a of attachments) {
        const filename = String(a.filename ?? "attachment").replace(/["\r\n]/g, "").slice(0, 120);
        const contentType = String(a.contentType || "application/octet-stream").replace(/["\r\n]/g, "");
        const data = String(a.base64).replace(/\s+/g, "");
        parts.push(
          `--${boundary}`,
          `Content-Type: ${contentType}; name="${filename}"`,
          "Content-Transfer-Encoding: base64",
          `Content-Disposition: attachment; filename="${filename}"`,
          "",
          (data.match(/.{1,76}/g) ?? []).join("\r\n"),
          "",
        );
      }
      parts.push(`--${boundary}--`, "");
      message = [
        ...headers,
        `Content-Type: multipart/mixed; boundary="${boundary}"`,
        "",
        ...parts,
      ].join("\n");
    }

    await client.dataBody(message);
    await client.cmd("quit", "QUIT", false, 221);
    client.close();
    return json({ ok: true, recipients });
  } catch (err) {
    client.close();
    const e = err as SmtpError;
    console.error("send-smtp-email failure", e.stage, e.reply ?? String(err));
    return json(
      {
        error: e.reply ?? (err instanceof Error ? err.message : "Unknown error"),
        stage: e.stage ?? "unknown",
      },
      502,
    );
  }
});
