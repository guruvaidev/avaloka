// Sends an analysis's results by email as a password-protected PDF attachment.
// Delivery uses SMTP (SMTP_HOST/SMTP_PORT/SMTP_USERNAME/SMTP_PASSWORD, EMAIL_FROM).
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.4";
import { PDFDocument, StandardFonts, rgb } from "https://esm.sh/@cantoo/pdf-lib@2.7.4";
import { sendMail } from "../_shared/raw-smtp.ts";

const sendViaSmtp = sendMail;


const cors = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Max-Age": "86400",
};

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

type JsonRecord = Record<string, unknown>;

function json(body: JsonRecord, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, "Content-Type": "application/json" },
  });
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string,
  );
}

function asRecord(value: unknown): JsonRecord | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : null;
}

function arrayValue(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null;
}

function pickOutputRows(messageOutput: unknown, analysisVizConfig: unknown): unknown[] | null {
  const output = asRecord(messageOutput);
  if (output) {
    const directRows = arrayValue(output.output_json);
    if (directRows?.length) return directRows;
    const outputFileData = arrayValue(output.output_file_data);
    if (outputFileData?.length) return outputFileData;
    const nested = asRecord(output.data);
    const nestedRows = nested ? arrayValue(nested.output_json) : null;
    if (nestedRows?.length) return nestedRows;
  }
  const viz = asRecord(analysisVizConfig);
  const charts = arrayValue(viz?.charts);
  const firstChart = asRecord(charts?.[0]);
  const chartRows = firstChart ? arrayValue(firstChart.data) : null;
  return chartRows?.length ? chartRows : null;
}

function pickVizConfig(messageOutput: unknown, analysisVizConfig: unknown): JsonRecord | null {
  const output = asRecord(messageOutput);
  const fromMessage = asRecord(output?.viz_config) ?? asRecord(output?.visualization_config);
  return fromMessage ?? asRecord(analysisVizConfig);
}

const CHART_COLORS = ["#1565ef", "#16a34a", "#f97316", "#a855f7", "#dc2626", "#0ea5e9", "#ca8a04", "#0891b2"];

function coerceNum(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function buildChartJsConfig(chart: JsonRecord, fallbackRows: unknown[] | null): Record<string, unknown> | null {
  const rawType = String(chart.type ?? "").toLowerCase();
  const typeMap: Record<string, string> = {
    bar: "bar", column: "bar", line: "line", area: "line",
    scatter: "scatter", pie: "pie", donut: "doughnut", doughnut: "doughnut",
  };
  const cjsType = typeMap[rawType];
  if (!cjsType) return null;

  const rows = (arrayValue(chart.data) ?? fallbackRows ?? []) as JsonRecord[];
  if (!rows.length) return null;

  const rowKeys = Object.keys(rows[0] ?? {});
  const resolveKey = (want: unknown): string => {
    const s = typeof want === "string" ? want : "";
    if (!s) return "";
    if (rowKeys.includes(s)) return s;
    const norm = s.trim().toLowerCase().replace(/[\s_-]+/g, "");
    const hit = rowKeys.find((k) => k.trim().toLowerCase().replace(/[\s_-]+/g, "") === norm);
    return hit ?? "";
  };

  const encodings = asRecord(chart.encodings) ?? {};
  const xRawCandidates = [
    (encodings.x as JsonRecord | undefined)?.field, encodings.x,
    chart.xKey, chart.x_key, chart.xAxis, chart.x_field, chart.x,
  ];
  let xKey = "";
  for (const c of xRawCandidates) { xKey = resolveKey(c); if (xKey) break; }

  const yRawCandidates = [
    (encodings.y as JsonRecord | undefined)?.field, encodings.y,
    chart.yKey, chart.y_key, chart.yAxis, chart.y_field, chart.y,
  ];
  let yKey = "";
  for (const c of yRawCandidates) { yKey = resolveKey(c); if (yKey) break; }

  const seriesRawCandidates = [
    (encodings.series as JsonRecord | undefined)?.field, encodings.series,
    (encodings.color as JsonRecord | undefined)?.field, encodings.color,
    chart.seriesField, chart.series_field, chart.groupBy, chart.group_by,
  ];
  let seriesKey = "";
  for (const c of seriesRawCandidates) { seriesKey = resolveKey(c); if (seriesKey) break; }

  if (!xKey && rowKeys.length) xKey = rowKeys[0];
  if (!yKey && rowKeys.length) {
    const numericKey = rowKeys.slice(1).find((k) => coerceNum(rows[0][k]) !== null);
    yKey = numericKey ?? rowKeys[rowKeys.length - 1];
  }
  if (xKey === yKey && rowKeys.length > 1) {
    yKey = rowKeys.find((k) => k !== xKey) ?? yKey;
  }
  if (!xKey || !yKey) return null;

  const title = String(chart.title ?? "Chart");

  if (cjsType === "scatter") {
    const pts = rows
      .map((r) => ({ x: coerceNum(r[xKey]), y: coerceNum(r[yKey]) }))
      .filter((p): p is { x: number; y: number } => p.x !== null && p.y !== null)
      .slice(0, 200);
    if (!pts.length) return null;
    return {
      type: "scatter",
      data: { datasets: [{ label: title, data: pts, backgroundColor: CHART_COLORS[0] }] },
      options: {
        plugins: { title: { display: true, text: title }, legend: { display: false } },
        scales: { x: { title: { display: true, text: xKey } }, y: { title: { display: true, text: yKey } } },
      },
    };
  }

  const labels = rows.slice(0, 50).map((r) => String(r[xKey] ?? ""));

  if (cjsType === "pie" || cjsType === "doughnut") {
    const data = rows.slice(0, 50).map((r) => coerceNum(r[yKey]) ?? 0);
    return {
      type: cjsType,
      data: { labels, datasets: [{ data, backgroundColor: CHART_COLORS }] },
      options: { plugins: { title: { display: true, text: title } } },
    };
  }

  const groups = new Map<string, Map<string, number>>();
  const allLabels = new Set<string>();
  for (const r of rows.slice(0, 500)) {
    const label = String(r[xKey] ?? "");
    allLabels.add(label);
    const s = seriesKey ? String(r[seriesKey] ?? "value") : "value";
    const y = coerceNum(r[yKey]) ?? 0;
    if (!groups.has(s)) groups.set(s, new Map());
    groups.get(s)!.set(label, y);
  }
  const orderedLabels = Array.from(allLabels).slice(0, 50);
  const datasets = Array.from(groups.entries()).slice(0, 8).map(([name, m], i) => ({
    label: name,
    data: orderedLabels.map((l) => m.get(l) ?? 0),
    backgroundColor: CHART_COLORS[i % CHART_COLORS.length],
    borderColor: CHART_COLORS[i % CHART_COLORS.length],
    fill: rawType === "area",
  }));

  return {
    type: cjsType,
    data: { labels: orderedLabels, datasets },
    options: {
      plugins: { title: { display: true, text: title }, legend: { display: datasets.length > 1 || seriesKey !== "" } },
      scales: { y: { title: { display: true, text: yKey } }, x: { title: { display: true, text: xKey } } },
    },
  };
}

async function renderChartPng(config: Record<string, unknown>): Promise<Uint8Array | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);
    const res = await fetch("https://quickchart.io/chart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chart: config, width: 700, height: 380, format: "png", backgroundColor: "white" }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    return new Uint8Array(await res.arrayBuffer());
  } catch {
    return null;
  }
}

// Sanitize text for WinAnsi (StandardFonts) — strip characters outside Latin-1
function safeText(s: string): string {
  // Replace common smart chars
  const cleaned = s
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/\u2013|\u2014/g, "-")
    .replace(/\u2026/g, "...");
  // Drop any remaining non-latin1
  let out = "";
  for (const ch of cleaned) {
    const code = ch.charCodeAt(0);
    out += code >= 32 && code <= 255 ? ch : "?";
  }
  return out;
}

type Pair = { question: string; output: unknown };

async function buildEncryptedPdf(
  title: string,
  senderName: string,
  message: string,
  pairs: Pair[],
  analysisViz: unknown,
  password: string,
): Promise<Uint8Array> {
  const pdfDoc = await PDFDocument.create();
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const fontBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

  const PAGE_W = 595.28; // A4
  const PAGE_H = 841.89;
  const MARGIN = 48;
  const CONTENT_W = PAGE_W - MARGIN * 2;

  let page = pdfDoc.addPage([PAGE_W, PAGE_H]);
  let y = PAGE_H - MARGIN;

  const newPage = () => {
    page = pdfDoc.addPage([PAGE_W, PAGE_H]);
    y = PAGE_H - MARGIN;
  };
  const ensureSpace = (h: number) => {
    if (y - h < MARGIN) newPage();
  };
  const drawText = (
    text: string,
    opts: { size?: number; bold?: boolean; color?: [number, number, number]; gap?: number } = {},
  ) => {
    const size = opts.size ?? 11;
    const f = opts.bold ? fontBold : font;
    const color = opts.color ? rgb(opts.color[0], opts.color[1], opts.color[2]) : rgb(0.06, 0.09, 0.16);
    const gap = opts.gap ?? 4;
    // wrap
    const words = safeText(text).split(/\s+/);
    let line = "";
    const lines: string[] = [];
    for (const w of words) {
      const test = line ? `${line} ${w}` : w;
      const width = f.widthOfTextAtSize(test, size);
      if (width > CONTENT_W && line) {
        lines.push(line);
        line = w;
      } else {
        line = test;
      }
    }
    if (line) lines.push(line);
    for (const ln of lines) {
      ensureSpace(size + gap);
      page.drawText(ln, { x: MARGIN, y: y - size, size, font: f, color });
      y -= size + gap;
    }
  };

  // Header
  drawText(safeText(title), { size: 20, bold: true, gap: 6 });
  drawText(`Shared by ${senderName} — ${pairs.length} result${pairs.length === 1 ? "" : "s"}`, {
    size: 10, color: [0.4, 0.45, 0.55], gap: 10,
  });

  if (message) {
    drawText("Note", { size: 11, bold: true, gap: 2 });
    drawText(message, { size: 11, color: [0.2, 0.25, 0.35], gap: 10 });
  }

  if (!pairs.length) {
    drawText("No results were found for this analysis.", { color: [0.4, 0.45, 0.55] });
  }

  for (let idx = 0; idx < pairs.length; idx++) {
    const p = pairs[idx];
    ensureSpace(40);
    y -= 8;
    drawText(`Result ${idx + 1} of ${pairs.length}`, { size: 9, bold: true, color: [0.4, 0.45, 0.55], gap: 6 });

    if (p.question) {
      drawText("Question", { size: 11, bold: true, gap: 2 });
      drawText(p.question, { size: 11, gap: 8 });
    }

    const viz = pickVizConfig(p.output, analysisViz);
    const rows = pickOutputRows(p.output, viz);

    // Charts
    const charts = arrayValue(asRecord(viz)?.charts)?.slice(0, 2) ?? [];
    for (const chartRaw of charts) {
      const chart = asRecord(chartRaw);
      if (!chart) continue;
      const cfg = buildChartJsConfig(chart, rows);
      if (!cfg) continue;
      const png = await renderChartPng(cfg);
      if (!png) continue;
      try {
        const img = await pdfDoc.embedPng(png);
        const maxW = CONTENT_W;
        const scale = Math.min(1, maxW / img.width);
        const w = img.width * scale;
        const h = img.height * scale;
        ensureSpace(h + 12);
        page.drawImage(img, { x: MARGIN, y: y - h, width: w, height: h });
        y -= h + 12;
      } catch {
        // skip on embed failure
      }
    }

    // Table (first ~30 rows, up to 6 cols)
    if (Array.isArray(rows) && rows.length) {
      const objs = rows.filter((r) => r && typeof r === "object") as Record<string, unknown>[];
      if (objs.length) {
        const cols = Array.from(new Set(objs.flatMap((r) => Object.keys(r)))).slice(0, 6);
        const colW = CONTENT_W / cols.length;
        const rowH = 14;
        const maxRows = Math.min(objs.length, 30);

        drawText("Data", { size: 11, bold: true, gap: 4 });
        // header
        ensureSpace(rowH);
        for (let i = 0; i < cols.length; i++) {
          const t = safeText(cols[i]).slice(0, 24);
          page.drawText(t, { x: MARGIN + i * colW, y: y - 10, size: 9, font: fontBold, color: rgb(0.06, 0.09, 0.16) });
        }
        y -= rowH;
        page.drawLine({
          start: { x: MARGIN, y: y + 2 },
          end: { x: MARGIN + CONTENT_W, y: y + 2 },
          thickness: 0.5,
          color: rgb(0.8, 0.83, 0.87),
        });

        for (let r = 0; r < maxRows; r++) {
          ensureSpace(rowH);
          for (let i = 0; i < cols.length; i++) {
            const v = objs[r][cols[i]];
            const t = safeText(v == null ? "" : String(v)).slice(0, 28);
            page.drawText(t, {
              x: MARGIN + i * colW,
              y: y - 10,
              size: 9,
              font,
              color: rgb(0.15, 0.19, 0.27),
            });
          }
          y -= rowH;
        }
        if (objs.length > maxRows) {
          drawText(`… ${objs.length - maxRows} more row(s) omitted`, { size: 9, color: [0.5, 0.55, 0.62], gap: 8 });
        } else {
          y -= 6;
        }
      }
    }
  }

  // Encrypt with user password. Owner password is random and discarded.
  const ownerBytes = new Uint8Array(24);
  crypto.getRandomValues(ownerBytes);
  const ownerPassword = btoa(String.fromCharCode(...ownerBytes));

  // @cantoo/pdf-lib encrypt API
  // deno-lint-ignore no-explicit-any
  (pdfDoc as any).encrypt({
    userPassword: password,
    ownerPassword,
    permissions: {
      printing: "highResolution",
      modifying: false,
      copying: false,
      annotating: false,
      fillingForms: false,
      contentAccessibility: true,
      documentAssembly: false,
    },
  });

  return await pdfDoc.save();
}

function toBase64(bytes: Uint8Array): string {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
  try {
    if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

    const authHeader = req.headers.get("Authorization") ?? "";
    if (!authHeader.toLowerCase().startsWith("bearer ")) {
      return json({ error: "Unauthorized" }, 401);
    }

    const analysisUrl = Deno.env.get("ANALYSIS_SUPABASE_URL");
    if (!analysisUrl) {
      // No default. This addressed Avaloka's own hosted project, so a
      // self-hosted deployment that never set the variable sent its users
      // links into somebody else's database — and nothing said so.
      throw new Error(
        "ANALYSIS_SUPABASE_URL is not set. Set it to your own Supabase project URL; " +
        "Avaloka does not ship a default because a default would point your users " +
        "at another deployment.",
      );
    }
    const analysisKey =
      Deno.env.get("ANALYSIS_SUPABASE_SERVICE_ROLE_KEY") ??
      Deno.env.get("ANALYSIS_SUPABASE_ANON_KEY") ??
        "";

    if (!analysisUrl || !analysisKey) {
      return json({
          error:
            "Server auth not configured: set ANALYSIS_SUPABASE_URL and one of \nANALYSIS_SUPABASE_SERVICE_ROLE_KEY / ANALYSIS_SUPABASE_ANON_KEY for this function. \nNo credential is baked into this source.",
        }, 500);
    }

    const token = authHeader.slice(7).trim();
    const analysisDb = createClient(analysisUrl, analysisKey, {
      global: { headers: { Authorization: authHeader } },
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { data: userData, error: userErr } = await analysisDb.auth.getUser(token);
    if (userErr || !userData.user) {
      return json({ error: "Unauthorized", detail: userErr?.message ?? "no user" }, 401);
    }

    const body = (await req.json().catch(() => ({}))) as {
      analysis_id?: string;
      recipients?: string[];
      message?: string | null;
      pdf_password?: string;
    };
    const analysisId = typeof body.analysis_id === "string" ? body.analysis_id.trim() : "";
    const recipients = Array.isArray(body.recipients)
      ? Array.from(new Set(body.recipients.map((r) => String(r).trim().toLowerCase()).filter((r) => EMAIL_RE.test(r))))
      : [];
    const message = (body.message ?? "").toString().slice(0, 5000);
    const pdfPassword = typeof body.pdf_password === "string" ? body.pdf_password : "";

    if (!analysisId || !UUID_RE.test(analysisId) || recipients.length === 0) {
      return json({ error: "A valid analysis_id and at least one recipient are required" }, 400);
    }
    if (recipients.length > 20) {
      return json({ error: "You can send results to up to 20 recipients at a time" }, 400);
    }
    if (pdfPassword.length < 6 || pdfPassword.length > 128) {
      return json({ error: "PDF password must be 6–128 characters" }, 400);
    }

    const { data: analysis, error: aErr } = await analysisDb
      .from("analyses")
      .select("id, name, viz_config")
      .eq("id", analysisId)
      .maybeSingle();

    if (aErr || !analysis) {
      return json({ error: aErr?.message || "Analysis not found" }, 404);
    }

    const { data: allMessages, error: mErr } = await analysisDb
      .from("analysis_messages")
      .select("role, content, output, created_at")
      .eq("analysis_id", analysisId)
      .order("created_at", { ascending: true });

    if (mErr) return json({ error: mErr.message }, 500);

    const msgs = (allMessages ?? []) as Array<{
      role: string;
      content: unknown;
      output: unknown;
      created_at: string;
    }>;

    const pairs: Pair[] = [];
    let lastUser = "";
    for (const m of msgs) {
      if (m.role === "user" && typeof m.content === "string") {
        lastUser = m.content;
      } else if (m.role === "assistant" && m.output != null) {
        pairs.push({ question: lastUser, output: m.output });
        lastUser = "";
      }
    }

    if (!Deno.env.get("SMTP_HOST") || !Deno.env.get("SMTP_USERNAME") || !Deno.env.get("SMTP_PASSWORD")) {
      return json({ error: "Email service is not configured. Please set the SMTP credentials." }, 500);
    }


    const title = (analysis as { name?: string }).name || "Analysis Results";
    const analysisViz = (analysis as { viz_config?: unknown }).viz_config;
    const senderName = userData.user.email || "A teammate";

    const pdfBytes = await buildEncryptedPdf(title, senderName, message, pairs, analysisViz, pdfPassword);
    const pdfBase64 = toBase64(pdfBytes);
    const safeName = title.replace(/[^\w\-. ]+/g, "_").slice(0, 80) || "analysis";

    const bodyHtml = `
      <div style="font-family:Inter,Arial,sans-serif;color:#0f172a;line-height:1.5">
        <h2 style="margin:0 0 8px 0">${esc(title)}</h2>
        <p style="margin:0 0 12px 0;color:#475569;font-size:14px">
          ${esc(senderName)} shared analysis results with you.
        </p>
        ${message ? `<div style="white-space:pre-wrap;padding:12px;border-left:3px solid #1565ef;background:#f8fafc;margin:12px 0;font-size:14px">${esc(message)}</div>` : ""}
        <p style="margin:16px 0;font-size:14px">
          The results are attached as a <strong>password-protected PDF</strong>.
          The sender will share the password with you separately.
        </p>
        <p style="margin:16px 0;color:#64748b;font-size:12px">
          If you cannot open the file, ask the sender to resend the password.
        </p>
      </div>
    `;

    try {
      await sendViaSmtp({
        to: recipients,
        subject: `Analysis results: ${title}`,
        html: bodyHtml,
        attachments: [
          { filename: `${safeName}.pdf`, base64: pdfBase64, contentType: "application/pdf" },
        ],
      });
    } catch (e) {
      return json({ error: `Email provider error: ${e instanceof Error ? e.message : String(e)}` }, 502);
    }


    return json({ ok: true, recipients });
  } catch (err) {
    console.error("send-analysis-email error", err);
    return json({ error: err instanceof Error ? err.message : "Unknown error" }, 500);
  }
});
