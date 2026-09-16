// Minimal SMTP client (STARTTLS / implicit TLS) with MIME attachment support.
// Surfaces the raw server replies so failures are diagnosable.

export class SmtpError extends Error {
  constructor(public stage: string, public reply: string) {
    super(`${stage}: ${reply}`);
  }
}

const SMTP_REPLY_TIMEOUT_MS = 20_000;
const SMTP_CONNECT_TIMEOUT_MS = 15_000;
const SMTP_MESSAGE_TIMEOUT_MS = 90_000;

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, stage: string): Promise<T> {
  let timer: number | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(
          () => reject(new SmtpError(stage, `timed out after ${Math.ceil(timeoutMs / 1000)} seconds`)),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

class RawSmtp {
  private conn!: Deno.Conn;
  private reader!: ReadableStreamDefaultReader<Uint8Array>;
  private buffer = "";
  private encoder = new TextEncoder();
  private decoder = new TextDecoder();

  async connect(hostname: string, port: number) {
    this.conn = await withTimeout(
      port === 465 ? Deno.connectTls({ hostname, port }) : Deno.connect({ hostname, port }),
      SMTP_CONNECT_TIMEOUT_MS,
      "connect",
    );
    this.attach();
    await this.expect("greeting", [220], SMTP_CONNECT_TIMEOUT_MS);
  }

  private attach() {
    this.reader = this.conn.readable.getReader();
  }

  private async writeAll(bytes: Uint8Array) {
    let offset = 0;
    while (offset < bytes.length) {
      const n = await this.conn.write(bytes.subarray(offset));
      if (n <= 0) throw new SmtpError("message-upload", "socket refused further writes");
      offset += n;
    }
  }

  private async readLine(timeoutMs = SMTP_REPLY_TIMEOUT_MS): Promise<string> {
    while (!this.buffer.includes("\r\n")) {
      const { value, done } = await withTimeout(this.reader.read(), timeoutMs, "server-reply");
      if (done) throw new SmtpError("connection", "server closed the connection");
      this.buffer += this.decoder.decode(value, { stream: true });
    }
    const idx = this.buffer.indexOf("\r\n");
    const line = this.buffer.slice(0, idx);
    this.buffer = this.buffer.slice(idx + 2);
    return line;
  }

  private async readReply(timeoutMs = SMTP_REPLY_TIMEOUT_MS): Promise<{ code: number; text: string }> {
    const lines: string[] = [];
    let line = await this.readLine(timeoutMs);
    lines.push(line);
    while (line.length >= 4 && line[3] === "-") {
      line = await this.readLine(timeoutMs);
      lines.push(line);
    }
    const text = lines.join("\n");
    return { code: Number(text.slice(0, 3)), text };
  }

  private async expect(stage: string, ok: number[], timeoutMs = SMTP_REPLY_TIMEOUT_MS) {
    const { code, text } = await this.readReply(timeoutMs);
    if (!ok.includes(code)) throw new SmtpError(stage, text);
    return text;
  }

  async cmd(stage: string, command: string, ...ok: number[]) {
    await this.writeAll(this.encoder.encode(command + "\r\n"));
    return await this.expect(stage, ok);
  }

  async startTls(hostname: string) {
    await this.cmd("starttls", "STARTTLS", 220);
    this.reader.releaseLock();
    this.conn = await Deno.startTls(this.conn, { hostname });
    this.buffer = "";
    this.attach();
  }

  async dataBody(body: string) {
    await this.cmd("data", "DATA", 354);
    const escaped = body.replace(/\r?\n/g, "\r\n").replace(/\r\n\./g, "\r\n..");
    await withTimeout(
      this.writeAll(this.encoder.encode(escaped + "\r\n.\r\n")),
      SMTP_MESSAGE_TIMEOUT_MS,
      "message-upload",
    );
    // Large messages can take a while to be accepted; use the longer budget.
    return await this.expect("message", [250], SMTP_MESSAGE_TIMEOUT_MS);
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

function wrap76(s: string): string {
  return (s.match(/.{1,76}/g) ?? []).join("\r\n");
}

export interface SmtpAttachment {
  filename: string;
  base64: string;
  contentType: string;
}

export async function sendMail(opts: {
  to: string[];
  subject: string;
  html: string;
  attachments?: SmtpAttachment[];
}) {
  const hostname = Deno.env.get("SMTP_HOST") ?? "";
  const username = Deno.env.get("SMTP_USERNAME") ?? "";
  const password = Deno.env.get("SMTP_PASSWORD") ?? "";
  if (!hostname || !username || !password) throw new Error("SMTP is not configured");
  const port = Number(Deno.env.get("SMTP_PORT") ?? "587");
  const from = Deno.env.get("EMAIL_FROM") || username;
  // EMAIL_FROM may be "Name <addr@domain>"; the SMTP envelope needs the bare address.
  const fromAddress = (from.match(/<([^>]+)>/)?.[1] ?? from).trim();

  const client = new RawSmtp();
  try {
    await client.connect(hostname, port);
    await client.cmd("ehlo", "EHLO avaloka.ai", 250);
    if (port !== 465) {
      await client.startTls(hostname);
      await client.cmd("ehlo-tls", "EHLO avaloka.ai", 250);
    }
    await client.cmd("auth", "AUTH LOGIN", 334);
    await client.cmd("auth-user", btoa(username), 334);
    await client.cmd("auth-pass", btoa(password), 235);

    await client.cmd("mail-from", `MAIL FROM:<${fromAddress}>`, 250);
    for (const rcpt of opts.to) await client.cmd("rcpt-to", `RCPT TO:<${rcpt}>`, 250, 251);

    const boundary = `=_avaloka_${crypto.randomUUID().replace(/-/g, "")}`;
    const attachments = opts.attachments ?? [];
    const headers = [
      `From: ${from}`,
      `To: ${opts.to.join(", ")}`,
      `Subject: ${opts.subject}`,
      "MIME-Version: 1.0",
    ];

    let message: string;
    if (attachments.length === 0) {
      message = [...headers, "Content-Type: text/html; charset=utf-8", "", opts.html].join("\n");
    } else {
      const parts = [
        `--${boundary}`,
        "Content-Type: text/html; charset=utf-8",
        "",
        opts.html,
        "",
      ];
      for (const a of attachments) {
        parts.push(
          `--${boundary}`,
          `Content-Type: ${a.contentType}; name="${a.filename}"`,
          "Content-Transfer-Encoding: base64",
          `Content-Disposition: attachment; filename="${a.filename}"`,
          "",
          wrap76(a.base64.replace(/\s+/g, "")),
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
    await client.cmd("quit", "QUIT", 221);
    client.close();
  } catch (err) {
    client.close();
    throw err;
  }
}
