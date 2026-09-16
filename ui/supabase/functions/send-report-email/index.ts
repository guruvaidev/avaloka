// Sends a report's contents by email as a password-protected PDF attachment.
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
  const rawType = String(chart.type ?? chart.chart_type ?? "").toLowerCase();
  const typeMap: Record<string, string> = {
    bar: "bar", column: "bar", line: "line", area: "line",
    scatter: "scatter", pie: "pie", donut: "doughnut", doughnut: "doughnut",
  };
  const cjsType = typeMap[rawType] ?? "bar";

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
  let xKey = "";
  for (const c of [(encodings.x as JsonRecord | undefined)?.field, encodings.x, chart.xKey, chart.x_key, chart.xAxis, chart.x]) {
    xKey = resolveKey(c); if (xKey) break;
  }
  let yKey = "";
  for (const c of [(encodings.y as JsonRecord | undefined)?.field, encodings.y, chart.yKey, chart.y_key, chart.yAxis, chart.y]) {
    yKey = resolveKey(c); if (yKey) break;
  }
  let seriesKey = "";
  for (const c of [(encodings.series as JsonRecord | undefined)?.field, (encodings.color as JsonRecord | undefined)?.field, chart.seriesField, chart.series_field]) {
    seriesKey = resolveKey(c); if (seriesKey) break;
  }

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
    const timer = setTimeout(() => controller.abort(), 5000);
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

function safeText(s: string): string {
  const cleaned = s
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/\u2013|\u2014/g, "-")
    .replace(/\u2026/g, "...");
  let out = "";
  for (const ch of cleaned) {
    const code = ch.charCodeAt(0);
    out += code >= 32 && code <= 255 ? ch : "?";
  }
  return out;
}

type ReportWidget = {
  title: string;
  chart: JsonRecord | null;
  insights: string[];
};

function extractInsights(widget: JsonRecord, chart: JsonRecord | null): string[] {
  const candidates: unknown[] = [];
  const push = (v: unknown) => { if (v != null) candidates.push(v); };
  push(widget.insights); push((widget as { key_insights?: unknown }).key_insights);
  push((widget as { takeaways?: unknown }).takeaways); push((widget as { notes?: unknown }).notes);
  if (chart) {
    push(chart.insights); push((chart as { key_insights?: unknown }).key_insights);
    push((chart as { takeaways?: unknown }).takeaways); push((chart as { notes?: unknown }).notes);
    push((chart as { description?: unknown }).description); push((chart as { reason?: unknown }).reason);
  }
  const out: string[] = [];
  for (const c of candidates) {
    const arr = Array.isArray(c) ? c : [c];
    for (const it of arr) {
      if (typeof it === "string" && it.trim()) out.push(it.trim());
      else if (it && typeof it === "object") {
        const r = it as { text?: string; insight?: string; message?: string };
        const v = r.text ?? r.insight ?? r.message;
        if (v) out.push(v);
      }
    }
  }
  return Array.from(new Set(out)).slice(0, 8);
}

function extractWidgets(reportKeyInsights: unknown): JsonRecord[] {
  const rec = asRecord(reportKeyInsights);
  const w = rec ? arrayValue(rec.widgets) : arrayValue(reportKeyInsights);
  return (w ?? []).filter((x): x is JsonRecord => !!x && typeof x === "object");
}

function pickWidgetChart(widget: JsonRecord): JsonRecord | null {
  const cfg = asRecord(widget.config) ?? widget;
  const charts = arrayValue(cfg.charts) ?? [];
  if (!charts.length) return asRecord(cfg);
  const focusId = (cfg as { focusChartId?: unknown }).focusChartId;
  if (focusId != null) {
    const hit = charts.find((c) => asRecord(c) && String((c as JsonRecord).id) === String(focusId));
    if (hit) return asRecord(hit);
  }
  return asRecord(charts[0]);
}

type PdfComment = {
  author: string;
  when: string;
  text: string;
  isReply: boolean;
  attachments: string[];
};

async function fetchLogoBytes(url: string): Promise<{ bytes: Uint8Array; kind: "png" | "jpg" } | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return null;
    const ct = (res.headers.get("content-type") ?? "").toLowerCase();
    const buf = new Uint8Array(await res.arrayBuffer());
    const kind: "png" | "jpg" =
      ct.includes("png") || url.toLowerCase().endsWith(".png") ? "png" : "jpg";
    return { bytes: buf, kind };
  } catch {
    return null;
  }
}

async function buildEncryptedPdf(
  title: string,
  senderName: string,
  message: string,
  widgets: ReportWidget[],
  analysisSamples: unknown,
  password: string,
  comments: PdfComment[],
  orgName: string,
  orgTagline: string,
  orgLogoUrl: string,
): Promise<Uint8Array> {
  const pdfDoc = await PDFDocument.create();
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const fontBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

  const PAGE_W = 595.28, PAGE_H = 841.89, MARGIN = 48;
  const CONTENT_W = PAGE_W - MARGIN * 2;

  const WATERMARK_URL =
    "https://avaloka-flow.lovable.app/__l5e/assets-v1/1d8a02eb-7ed9-40a5-a86b-a57148ce3410/avaloka-watermark.png";

  // Preload watermark image once. Fall back to null on failure -> watermark skipped.
  let watermarkImg: any = null;
  try {
    const fetched = await fetchLogoBytes(WATERMARK_URL);
    if (fetched) {
      watermarkImg = fetched.kind === "png"
        ? await pdfDoc.embedPng(fetched.bytes)
        : await pdfDoc.embedJpg(fetched.bytes);
    }
  } catch { /* ignore */ }

  // Preload org logo (for the header) once as well.
  let orgLogoImg: any = null;
  if (orgLogoUrl) {
    try {
      const fetched = await fetchLogoBytes(orgLogoUrl);
      if (fetched) {
        orgLogoImg = fetched.kind === "png"
          ? await pdfDoc.embedPng(fetched.bytes)
          : await pdfDoc.embedJpg(fetched.bytes);
      }
    } catch { /* ignore */ }
  }

  const drawWatermark = (p: any) => {
    if (!watermarkImg) return;
    const size = 340;
    const scale = Math.min(size / watermarkImg.width, size / watermarkImg.height);
    const w = watermarkImg.width * scale;
    const h = watermarkImg.height * scale;
    p.drawImage(watermarkImg, {
      x: (PAGE_W - w) / 2,
      y: (PAGE_H - h) / 2,
      width: w,
      height: h,
      opacity: 0.08,
    });
  };

  let page = pdfDoc.addPage([PAGE_W, PAGE_H]);
  drawWatermark(page);
  let y = PAGE_H - MARGIN;

  const newPage = () => {
    page = pdfDoc.addPage([PAGE_W, PAGE_H]);
    drawWatermark(page);
    y = PAGE_H - MARGIN;
  };

  const ensureSpace = (h: number) => { if (y - h < MARGIN) newPage(); };
  const drawText = (
    text: string,
    opts: { size?: number; bold?: boolean; color?: [number, number, number]; gap?: number } = {},
  ) => {
    const size = opts.size ?? 11;
    const f = opts.bold ? fontBold : font;
    const color = opts.color ? rgb(opts.color[0], opts.color[1], opts.color[2]) : rgb(0.06, 0.09, 0.16);
    const gap = opts.gap ?? 4;
    const words = safeText(text).split(/\s+/);
    let line = "";
    const lines: string[] = [];
    for (const w of words) {
      const test = line ? `${line} ${w}` : w;
      const width = f.widthOfTextAtSize(test, size);
      if (width > CONTENT_W && line) { lines.push(line); line = w; }
      else line = test;
    }
    if (line) lines.push(line);
    for (const ln of lines) {
      ensureSpace(size + gap);
      page.drawText(ln, { x: MARGIN, y: y - size, size, font: f, color });
      y -= size + gap;
    }
  };

  // Header: company details card (logo + name + tagline)
  const HEADER_H = 64;
  const headerTop = y;
  const headerBottom = y - HEADER_H;
  // Card background
  page.drawRectangle({
    x: MARGIN - 6,
    y: headerBottom,
    width: CONTENT_W + 12,
    height: HEADER_H,
    color: rgb(0.965, 0.976, 0.996),
    borderColor: rgb(0.85, 0.89, 0.96),
    borderWidth: 0.75,
  });
  // Accent bar on the left
  page.drawRectangle({
    x: MARGIN - 6,
    y: headerBottom,
    width: 3,
    height: HEADER_H,
    color: rgb(0.086, 0.396, 0.937),
  });

  const LOGO_SIZE = 44;
  const logoX = MARGIN + 6;
  const logoTop = headerTop - 10;
  const hasLogo = !!orgLogoImg;
  if (hasLogo) {
    const scale = Math.min(LOGO_SIZE / orgLogoImg.width, LOGO_SIZE / orgLogoImg.height);
    const iw = orgLogoImg.width * scale;
    const ih = orgLogoImg.height * scale;
    page.drawImage(orgLogoImg, {
      x: logoX + (LOGO_SIZE - iw) / 2,
      y: logoTop - LOGO_SIZE + (LOGO_SIZE - ih) / 2,
      width: iw,
      height: ih,
    });
  }

  const textX = hasLogo ? logoX + LOGO_SIZE + 14 : MARGIN + 6;
  if (orgName) {
    page.drawText(safeText(orgName), {
      x: textX,
      y: headerTop - 26,
      size: 15,
      font: fontBold,
      color: rgb(0.06, 0.09, 0.16),
    });
  }
  if (orgTagline) {
    page.drawText(safeText(orgTagline), {
      x: textX,
      y: headerTop - 44,
      size: 10,
      font,
      color: rgb(0.4, 0.45, 0.55),
    });
  }
  y = headerBottom - 18;


  drawText(safeText(title), { size: 20, bold: true, gap: 6 });
  drawText(`Shared by ${senderName} — ${widgets.length} insight${widgets.length === 1 ? "" : "s"}`, {
    size: 10, color: [0.4, 0.45, 0.55], gap: 10,
  });

  if (message) {
    drawText("Note", { size: 11, bold: true, gap: 2 });
    drawText(message, { size: 11, color: [0.2, 0.25, 0.35], gap: 10 });
  }

  if (!widgets.length) {
    drawText("This report has no widgets to display.", { color: [0.4, 0.45, 0.55] });
  }

  const fallbackRows = arrayValue(analysisSamples);

  // Chart rendering is remote I/O. Run it concurrently so a report with many
  // widgets does not spend up to eight seconds per chart and hit the function
  // execution limit before SMTP delivery begins.
  const chartImages = await Promise.all(
    widgets.map(async (widget) => {
      if (!widget.chart) return null;
      const config = buildChartJsConfig(widget.chart, fallbackRows);
      return config ? await renderChartPng(config) : null;
    }),
  );

  for (let idx = 0; idx < widgets.length; idx++) {
    const w = widgets[idx];
    if (idx > 0) newPage();
    y -= 20;
    drawText(`Insight ${idx + 1} of ${widgets.length}`, { size: 9, bold: true, color: [0.4, 0.45, 0.55], gap: 6 });
    drawText(w.title || "Chart", { size: 13, bold: true, gap: 10 });

    const png = chartImages[idx];
    if (png) {
      try {
        const img = await pdfDoc.embedPng(png);
        const scale = Math.min(1, CONTENT_W / img.width);
        const cw = img.width * scale, ch = img.height * scale;
        ensureSpace(ch + 20);
        page.drawImage(img, { x: MARGIN, y: y - ch, width: cw, height: ch });
        y -= ch + 20;
      } catch { /* ignore */ }
    }

    if (w.insights.length) {
      drawText("Key insights", { size: 11, bold: true, gap: 4 });
      for (const ins of w.insights) {
        drawText(`• ${ins}`, { size: 11, color: [0.2, 0.25, 0.35], gap: 4 });
      }
      y -= 16;
    }
  }

  // Comments section
  ensureSpace(50);
  y -= 16;
  drawText(`Comments (${comments.length})`, { size: 14, bold: true, gap: 4 });
  ensureSpace(8);
  page.drawLine({
    start: { x: MARGIN, y: y - 2 },
    end: { x: PAGE_W - MARGIN, y: y - 2 },
    thickness: 0.75,
    color: rgb(0.75, 0.78, 0.83),
  });
  y -= 12;
  if (!comments.length) {
    drawText("No comments yet.", { size: 11, color: [0.4, 0.45, 0.55], gap: 4 });
  } else {
    for (const c of comments) {
      ensureSpace(18);
      drawText(`${c.author}: ${c.text}`, { size: 11, gap: 6 });
      for (const a of c.attachments) {
        drawText(`   • ${a}`, { size: 10, color: [0.4, 0.45, 0.55], gap: 3 });
      }
    }
  }

  const ownerBytes = new Uint8Array(24);
  crypto.getRandomValues(ownerBytes);
  const ownerPassword = btoa(String.fromCharCode(...ownerBytes));

  // deno-lint-ignore no-explicit-any
  (pdfDoc as any).encrypt({
    userPassword: password,
    ownerPassword,
    permissions: {
      printing: "highResolution", modifying: false, copying: false, annotating: false,
      fillingForms: false, contentAccessibility: true, documentAssembly: false,
    },
  });

  return await pdfDoc.save();
}

function toBase64(bytes: Uint8Array): string {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(bin);
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
  try {
    if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

    const authHeader = req.headers.get("Authorization") ?? "";
    if (!authHeader.toLowerCase().startsWith("bearer ")) return json({ error: "Unauthorized" }, 401);

    // The bearer token comes from the external "analysis" Supabase project
    // (whichever project ANALYSIS_SUPABASE_URL names), not the Cloud project this
    // deployed under. Verify + query against that project so JWT `kid`
    // resolves correctly and reports/analyses rows are visible.
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
    if (!analysisUrl || !analysisKey) return json({
          error:
            "Server auth not configured: set ANALYSIS_SUPABASE_URL and one of \nANALYSIS_SUPABASE_SERVICE_ROLE_KEY / ANALYSIS_SUPABASE_ANON_KEY for this function. \nNo credential is baked into this source.",
        }, 500);

    const token = authHeader.slice(7).trim();
    const db = createClient(analysisUrl, analysisKey, {
      global: { headers: { Authorization: authHeader } },
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { data: userData, error: userErr } = await db.auth.getUser(token);
    if (userErr || !userData.user) return json({ error: "Unauthorized", detail: userErr?.message ?? "no user" }, 401);

    const body = (await req.json().catch(() => ({}))) as {
      report_id?: string;
      recipients?: string[];
      message?: string | null;
      pdf_password?: string;
      download_only?: boolean;
    };
    const downloadOnly = body.download_only === true;
    const reportId = typeof body.report_id === "string" ? body.report_id.trim() : "";
    const recipients = Array.isArray(body.recipients)
      ? Array.from(new Set(body.recipients.map((r) => String(r).trim().toLowerCase()).filter((r) => EMAIL_RE.test(r))))
      : [];
    const message = (body.message ?? "").toString().slice(0, 5000);
    const pdfPassword = typeof body.pdf_password === "string" ? body.pdf_password : "";

    if (!reportId || !UUID_RE.test(reportId)) {
      return json({ error: "A valid report_id is required" }, 400);
    }
    if (!downloadOnly && recipients.length === 0) {
      return json({ error: "At least one recipient is required" }, 400);
    }
    if (recipients.length > 20) return json({ error: "Up to 20 recipients at a time" }, 400);
    if (pdfPassword.length < 6 || pdfPassword.length > 128) {
      return json({ error: "PDF password must be 6–128 characters" }, 400);
    }

    const { data: report, error: rErr } = await db
      .from("reports")
      .select("id, title, key_insights, analysis_id, analyses:analysis_id(name, viz_config, samples)")
      .eq("id", reportId)
      .maybeSingle();
    if (rErr || !report) return json({ error: rErr?.message || "Report not found" }, 404);

    const rec = report as {
      title?: string;
      key_insights?: unknown;
      analyses?: { name?: string; viz_config?: unknown; samples?: unknown } | null;
    };

    const widgets: ReportWidget[] = extractWidgets(rec.key_insights).map((w) => {
      const chart = pickWidgetChart(w);
      return {
        title: String(w.title ?? chart?.title ?? "Chart"),
        chart,
        insights: extractInsights(w, chart),
      };
    });

    const { data: commentRows } = await db
      .from("report_comments")
      .select(
        "id, body, created_at, parent_id, attachments, author:profiles!author_id(full_name)",
      )
      .eq("report_id", reportId)
      .order("created_at", { ascending: true });

    // Order rows so each reply appears directly under its parent comment.
    const rowsAll: any[] = commentRows ?? [];
    const idSet = new Set(rowsAll.map((r) => r.id));
    const childrenBy = new Map<string, any[]>();
    const roots: any[] = [];
    for (const r of rowsAll) {
      if (r.parent_id && idSet.has(r.parent_id)) {
        const arr = childrenBy.get(r.parent_id) ?? [];
        arr.push(r);
        childrenBy.set(r.parent_id, arr);
      } else {
        roots.push(r);
      }
    }
    const orderedRows: any[] = [];
    for (const root of roots) {
      orderedRows.push(root);
      for (const k of childrenBy.get(root.id) ?? []) orderedRows.push(k);
    }

    const pdfComments: PdfComment[] = orderedRows.map((r: any) => {
      const author = Array.isArray(r.author) ? r.author[0] : r.author;
      const atts = Array.isArray(r.attachments) ? r.attachments : [];
      return {
        author: (author && author.full_name) || "Unknown user",
        when: new Date(r.created_at).toLocaleString(),
        text: safeText(String(r.body ?? "")),
        isReply: Boolean(r.parent_id) && idSet.has(r.parent_id),
        attachments: atts.map((a: any) => safeText(String(a?.name ?? "attachment"))),
      };
    });

    if (!Deno.env.get("SMTP_HOST") || !Deno.env.get("SMTP_USERNAME") || !Deno.env.get("SMTP_PASSWORD")) {
      return json({ error: "Email service is not configured. Please set the SMTP credentials." }, 500);
    }


    const title = rec.title || rec.analyses?.name || "Report";
    const senderName = userData.user.email || "A teammate";

    let orgName = "";
    let orgTagline = "";
    let orgLogoUrl = "";
    try {
      const authUid = userData.user.id;

      // Use a service-role client for org metadata lookups so RLS on
      // app_users/profiles/organizations cannot silently return empty rows.
      const adminKey =
        Deno.env.get("ANALYSIS_SUPABASE_SERVICE_ROLE_KEY") ??
        Deno.env.get("PRIMARY_SUPABASE_SERVICE_ROLE_KEY") ??
        "";
      const admin = adminKey
        ? createClient(analysisUrl, adminKey, { auth: { persistSession: false, autoRefreshToken: false } })
        : db;

      // 1) app_users.auth_user_id -> organization_id
      const { data: appUser } = await admin
        .from("app_users")
        .select("organization_id")
        .eq("auth_user_id", authUid)
        .maybeSingle();
      let orgId = (appUser as any)?.organization_id ?? null;

      // 2) Resolve caller's profile (by user_id, then id) if still missing.
      let profileId: string | null = null;
      const { data: profByUser } = await admin
        .from("profiles")
        .select("id, organization_id")
        .eq("user_id", authUid)
        .maybeSingle();
      if (profByUser) {
        profileId = (profByUser as any).id ?? null;
        orgId = orgId ?? ((profByUser as any).organization_id ?? null);
      }
      if (!profileId) {
        const { data: profById } = await admin
          .from("profiles")
          .select("id, organization_id")
          .eq("id", authUid)
          .maybeSingle();
        if (profById) {
          profileId = (profById as any).id ?? null;
          orgId = orgId ?? ((profById as any).organization_id ?? null);
        }
      }

      if (orgId) {
        const { data: org, error: orgErr } = await admin
          .from("organizations")
          .select("name, logo_url, tagline")
          .eq("id", orgId)
          .maybeSingle();
        if (orgErr) console.log("[send-report-email] org select err", orgErr);
        orgName = (org as any)?.name ?? "";
        orgTagline = (org as any)?.tagline ?? "";
        orgLogoUrl = (org as any)?.logo_url ?? "";
      }

      // 3) Fallback: organization owned by this user's profile.
      if (!orgName && profileId) {
        const { data: owned } = await admin
          .from("organizations")
          .select("name, logo_url, tagline")
          .eq("owner_profile_id", profileId)
          .maybeSingle();
        orgName = (owned as any)?.name ?? orgName;
        orgTagline = (owned as any)?.tagline ?? orgTagline;
        orgLogoUrl = (owned as any)?.logo_url ?? orgLogoUrl;
      }

      console.log("[send-report-email] org lookup", { authUid, profileId, orgId, orgName, orgLogoUrl, usedAdmin: !!adminKey });
    } catch (e) {
      console.log("[send-report-email] org lookup failed", e);
    }



    const pdfStartedAt = Date.now();
    const pdfBytes = await buildEncryptedPdf(
      title, senderName, message, widgets, rec.analyses?.samples, pdfPassword, pdfComments,
      orgName, orgTagline, orgLogoUrl,
    );
    console.log("[send-report-email] pdf ready", {
      reportId,
      widgets: widgets.length,
      bytes: pdfBytes.length,
      durationMs: Date.now() - pdfStartedAt,
    });
    const safeName = title.replace(/[^\w\-. ]+/g, "_").slice(0, 80) || "report";

    if (downloadOnly) {
      return new Response(pdfBytes, {
        status: 200,
        headers: {
          ...cors,
          "Content-Type": "application/pdf",
          "Content-Disposition": `attachment; filename="${safeName}.pdf"`,
        },
      });
    }

    const pdfBase64 = toBase64(pdfBytes);

    const bodyHtml = `
      <div style="font-family:Inter,Arial,sans-serif;color:#0f172a;line-height:1.5">
        <h2 style="margin:0 0 8px 0">${esc(title)}</h2>
        <p style="margin:0 0 12px 0;color:#475569;font-size:14px">
          ${esc(senderName)} shared a report with you.
        </p>
        ${message ? `<div style="white-space:pre-wrap;padding:12px;border-left:3px solid #1565ef;background:#f8fafc;margin:12px 0;font-size:14px">${esc(message)}</div>` : ""}
        <p style="margin:16px 0;font-size:14px">
          The report is attached as a <strong>password-protected PDF</strong>.
          The sender will share the password with you separately.
        </p>
      </div>
    `;

    try {
      const smtpStartedAt = Date.now();
      await sendViaSmtp({
        to: recipients,
        subject: `Report: ${title}`,
        html: bodyHtml,
        attachments: [
          { filename: `${safeName}.pdf`, base64: pdfBase64, contentType: "application/pdf" },
        ],
      });
      console.log("[send-report-email] smtp delivered", {
        recipients: recipients.length,
        durationMs: Date.now() - smtpStartedAt,
      });
    } catch (e) {
      const raw = e instanceof Error ? (e.stack || e.message) : String(e);
      const detail = {
        error: `Email provider error: ${e instanceof Error ? e.message : String(e)}`,
        stage: "smtp-send",
        smtp_host: Deno.env.get("SMTP_HOST") ?? "",
        smtp_port: Deno.env.get("SMTP_PORT") ?? "587",
        smtp_user: Deno.env.get("SMTP_USERNAME") ?? "",
        from: Deno.env.get("EMAIL_FROM") || (Deno.env.get("SMTP_USERNAME") ?? ""),
        recipients,
        server_reply: raw.slice(0, 2000),
      };
      console.error("send-report-email smtp failure", JSON.stringify(detail));
      return json(detail, 502);
    }



    return json({ ok: true, recipients });
  } catch (err) {
    console.error("send-report-email error", err);
    return json({ error: err instanceof Error ? err.message : "Unknown error" }, 500);
  }
});
