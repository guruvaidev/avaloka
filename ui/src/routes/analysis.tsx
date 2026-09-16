import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { createFileRoute, useNavigate, useRouterState } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { analysesKey, useAnalysisMutations } from "@/lib/analyses";
import { deriveVizFromRows } from "@/lib/derive-viz";
import { buildInsightGroups } from "@/lib/insight-groups";


import { useProjects } from "@/lib/projects";
import { TagPicker } from "@/components/analysis/TagPicker";
const avatarBotAsset = { url: "/assets/dashboard/avatar_bot.png" };
const avatarBot = avatarBotAsset.url;
const avatarTtsVideo = "/assets/dashboard/avatar_tts.mp4";
const avatarTtsIdle = "/assets/dashboard/avatar_tts_idle.png";
import {
  MessageChatCircle,
  Stars02,
  Folder,
  User01,
  X,
  Plus,
  Tag01,
  Zap,
  Copy01,
  ClipboardPlus,
  MessageCircle02,
  Repeat02,
  CodeSnippet02,
  ThumbsUp,
  ThumbsDown,
  MessageChatSquare,
  AlertTriangle,
  ChevronDown,
  SearchLg,
  File02,
  ArrowLeft,
  ArrowRight,
  Database01,
  Database02,
  Check,
  ChevronRight,
  ChevronLeft,
  Table,
  Calendar,
  HelpCircle,
  ClockRewind,
} from "@untitledui/icons";
import { Dialog, DialogContent, DialogPortal, DialogOverlay, DialogTitle } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { analysisGroupName } from "@/lib/analysis-datasets";

import {
  ComposedChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { IconRail } from "@/components/dashboard/IconRail";
import { speakText, stopSpeaking, useVoiceInput } from "@/hooks/useVoiceInput";
import { AutoInsightsModal } from "@/components/dashboard/AutoInsightsModal";
import { collectVizInsights, DynamicChart, normalizeVizConfig, type Slide } from "@/components/dashboard/dynamicChart";
import { AddToDashboardModal } from "@/components/dashboard/AddToDashboardModal";
import { ConnectCloudModal } from "@/components/database/ConnectCloudModal";
import { InsightCommentCard } from "@/components/dashboard/AddCommentPopover";
import { CommentsPanel } from "@/components/dashboard/CommentsPanel";
import { ProjectNotificationsPopover } from "@/components/dashboard/ProjectNotificationsPopover";
import { ShareCollaboratorsButton } from "@/components/dashboard/ShareCollaboratorsButton";
import { BackendApiError, backendApi, isDeferredTurn } from "@/lib/api/backendApi";
import { startTrainingJob, claimTrainingResult, TRAINING_RESULT_EVENT } from "@/lib/training-jobs";
import { scheduledTaskStore, extractSchedulePhrase, currentUserLabel } from "@/lib/api/scheduled-tasks";
import { useTypingPresence } from "@/hooks/useTypingPresence";
import { TypingIndicator } from "@/components/dashboard/TypingIndicator";
import {
  appendMessage as persistMessage,
  createAnalysis as persistCreateAnalysis,
  getDefaultProjectId,
  loadAnalysis as persistLoadAnalysis,
  loadChildAnalyses,
  loadAnalysisGroupDatasets,
  loadMessages as persistLoadMessages,
  resolveMotherAnalysisId,
  updateAnalysisContext as persistUpdateAnalysisContext,
  type AnalysisMessage,
  type ChildAnalysisSummary,
} from "@/lib/analysis-messages";

import { cx } from "@/lib/utils/cx";
import { toast } from "sonner";
import { resolveHistoryViewPayload, type HistoryViewPayload } from "@/lib/analysis-history";
import { HistoryCollaborationDetailModal } from "@/components/dashboard/HistoryCollaborationDetailModal";
import { supabase } from "@/integrations/supabase/client";
import { getCurrentProfileId } from "@/lib/current-profile";

import { BACKEND_API_BASE as BACKEND_CODE_BASE } from "@/lib/api/backend-config";
import { VERSION_RESTORED_EVENT, type VersionRestoredDetail } from "@/lib/api/analysis-versions";
import {
  analysisDashboardTab,
  analysisDashboardTabId,
  chartIndicesForDashboardTab,
  insightPayloadFromVizConfig,
  type DashboardInsightPayload,
  type DashboardTab,
} from "@/lib/api/project-dashboard";
import { HistoryVersionPanel } from "@/components/dashboard/HistoryVersionPanel";
import { AllInsightsPanel, type InsightSource } from "@/components/dashboard/AllInsightsPanel";
import { ProjectDashboardView } from "@/components/dashboard/ProjectDashboardView";
import { DataSetBrowserModal } from "@/components/dashboard/DataSetBrowserModal";
import { UploadModal } from "@/components/dashboard/UploadModal";
import { DataSourceGrid } from "@/components/dashboard/DataSourceGrid";
const bgPattern = { url: "/assets/dashboard/Background_pattern_decorative.png" };
const illustration = { url: "/assets/dashboard/Upload_Illustration.png" };
import { useUpgradeGate } from "@/components/dashboard/UpgradeGate";


export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "New Analysis · Avaloka AI" },
      { name: "description", content: "Generate AI insights for your dataset." },
    ],
  }),
  component: AnalysisPage,
});

type ChatMessage = {
  id: string;
  role: "ai" | "user";
  content: string;
  time: string;
  thinking?: boolean;
  insight?: boolean;
  sectionId?: string;
  error?: boolean;
  replyTo?: { label: string; content: string };
  authorId?: string | null;
  authorName?: string;
  authorAvatar?: string | null;
  isSelf?: boolean;
};

const ANALYSIS_COMPLETE_MESSAGE = "Analysis complete. Please check your output table and charts for analysis.";

/** True when a turn actually produced an execution result. */
function hasExecutionResult(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") return false;
  const data = payload as Record<string, unknown>;
  const oj = data.output_json;
  const hasJson = Array.isArray(oj) ? oj.length > 0 : oj != null;
  return hasJson || data.output_file_data != null;
}

type ParsedTable = { title?: string; headers: string[]; rows: Record<string, unknown>[] };

/** Split a pipe-delimited markdown table row into cells. */
function splitPipeRow(line: string): string[] {
  const trimmed = line.trim();
  const withoutOuterPipes = trimmed.replace(/^\|/, "").replace(/\|$/, "");
  return withoutOuterPipes.split("|").map((c) => c.trim());
}

/**
 * Parse a single markdown pipe table from a block of text.
 * Returns null if the block does not look like a header + separator + body.
 */
function parseMarkdownTableBlock(block: string): ParsedTable | null {
  const lines = block
    .split("\n")
    .map((l) => l.trimEnd())
    .filter((l) => l.trim().length > 0);
  if (lines.length < 3) return null;

  const headerCells = splitPipeRow(lines[0]!);
  if (headerCells.length === 0 || headerCells.every((c) => c.length === 0)) return null;

  const separator = lines[1]!;
  if (!/^\s*\|?[\s:|-]+\|\s*$/.test(separator) || !/-/.test(separator)) return null;

  const rows: Record<string, unknown>[] = [];
  for (let i = 2; i < lines.length; i++) {
    const cells = splitPipeRow(lines[i]!);
    if (cells.length === 0 || cells.every((c) => c.length === 0)) continue;
    const row: Record<string, unknown> = {};
    headerCells.forEach((h, idx) => {
      const key = h.trim() || `col_${idx + 1}`;
      row[key] = cells[idx] ?? "";
    });
    rows.push(row);
  }
  if (rows.length === 0) return null;

  return { headers: headerCells, rows };
}

/**
 * Find all markdown pipe tables in `raw` and parse them into structured rows.
 * Returns an empty array when no tables are found.
 */
function parseMarkdownTables(raw: string): ParsedTable[] {
  if (!raw) return [];
  const tables: ParsedTable[] = [];
  // Match a header row, a separator row containing '-', and any following body rows.
  const tableRegex =
    /(^|\n)[ \t]*\|?[^\n]*\|[^\n]*\n[ \t]*\|?[\s:|-]*-[\s:|-]*\n(?:[ \t]*\|?[^\n]*\|[^\n]*\n?)*/g;
  let match: RegExpExecArray | null;
  const seen = new Set<string>();
  while ((match = tableRegex.exec(raw)) !== null) {
    const block = match[0].replace(/^\n/, "");
    const parsed = parseMarkdownTableBlock(block);
    if (!parsed) continue;
    const key = JSON.stringify({ headers: parsed.headers, rows: parsed.rows.slice(0, 3) });
    if (seen.has(key)) continue;
    seen.add(key);
    tables.push(parsed);
  }
  return tables;
}

/**
 * Extract tabular payloads out of assistant text so the result table only ever
 * appears in the Data table panel — never duplicated inside the chat bubble.
 * Returns the cleaned text plus any parsed markdown tables.
 */
export function extractTabularContent(raw: string): { text: string; tables: ParsedTable[] } {
  if (!raw) return { text: "", tables: [] };
  let text = raw;

  // Fenced blocks that are only CSV / JSON row dumps.
  text = text.replace(/```[a-zA-Z]*\s*\n([\s\S]*?)```/g, (block, inner: string) => {
    const body = inner.trim();
    if (!body) return "";
    const looksJsonRows = /^\[\s*[{[]/.test(body) && /[}\]]\s*\]$/.test(body);
    const lines = body.split("\n").filter((l) => l.trim());
    const looksCsv =
      lines.length > 1 &&
      lines.every((l) => (l.match(/,/g)?.length ?? 0) >= 1) &&
      new Set(lines.map((l) => (l.match(/,/g)?.length ?? 0))).size <= 2;
    return looksJsonRows || looksCsv ? "" : block;
  });

  // Markdown pipe tables (header + --- separator + body rows).
  const tables = parseMarkdownTables(text);
  text = text.replace(
    /(^|\n)[ \t]*\|?[^\n]*\|[^\n]*\n[ \t]*\|?[\s:|-]*-[\s:|-]*\n(?:[ \t]*\|?[^\n]*\|[^\n]*\n?)*/g,
    "$1",
  );

  // Standalone "N rows × M columns" preambles (e.g. "Transformed data — 20 rows × 7 columns.").
  text = text.replace(/(^|\n)[^\n]*\b\d+\s*rows?\s*[x×*]\s*\d+\s*columns?\b[^\n]*(?=\n|$)/gi, "$1");

  // Leftover orphan pipe rows.
  text = text
    .split("\n")
    .filter((l) => !/^\s*\|.*\|\s*$/.test(l))
    .join("\n");

  return { text: text.replace(/\n{3,}/g, "\n\n").trim(), tables };
}

/** Backwards-compatible helper that only returns the text portion. */
export function stripTabularContent(raw: string): string {
  return extractTabularContent(raw).text;
}

function chatDisplayContent(content: string, output?: unknown): string {
  const cleaned = stripTabularContent(content ?? "");
  if (cleaned.length > 0) return cleaned;
  return hasExecutionResult(output) || (content && content.trim().length > 0)
    ? ANALYSIS_COMPLETE_MESSAGE
    : content;
}




type WorkspaceSection = { id: string; title?: string };

function mainSectionId(tabId: string) {
  return `${tabId}-main`;
}

type OutputFile = { filename: string; content: string };

type DatasetChip = {
  id: string;
  name: string;
  sessionId?: string;
  threadId?: string;
  schema?: unknown;
  samples?: Record<string, unknown>[];
  rowsSampled?: number;
  sizeMb?: number;
  visualizationConfig?: unknown;
  visualizationStatus?: string;
};

/** Total dataset size in MB from a register/upload response (0 / missing → undefined). */
function datasetSizeMb(source: any): number | undefined {
  const mb =
    typeof source?.file_size_mb === "number"
      ? source.file_size_mb
      : typeof source?.file_size_bytes === "number"
        ? source.file_size_bytes / (1024 * 1024)
        : undefined;
  return mb && mb > 0 ? mb : undefined;
}

export function formatDatasetSize(mb?: number): string | null {
  if (!mb || mb <= 0) return null;
  if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(mb < 10 ? 2 : 1)} MB`;
}



function normalizeDatasetSchema(schema: unknown, samples: unknown[]): string[] {
  if (Array.isArray(schema)) {
    const columns = schema
      .map((column) =>
        typeof column === "string"
          ? column
          : column && typeof column === "object" && "name" in column
            ? String((column as { name: unknown }).name)
            : "",
      )
      .filter(Boolean);
    if (columns.length) return columns;
  }
  if (schema && typeof schema === "object") return Object.keys(schema);
  const first = samples[0];
  if (first && typeof first === "object" && !Array.isArray(first)) return Object.keys(first);
  if (Array.isArray(first)) return first.map((_, index) => `col_${index + 1}`);
  return [];
}

function populatedDatasetFromResponse(response: any, filename: string): UploadedDataset | null {
  if (
    typeof response?.dataset_id !== "string" ||
    !response.dataset_id ||
    !Array.isArray(response.samples) ||
    response.samples.length === 0
  ) {
    return null;
  }
  return {
    filename,
    schema: normalizeDatasetSchema(response.schema, response.samples),
    samples: response.samples,
    rows_sampled: response.rows_sampled ?? response.samples.length,
    size_mb: datasetSizeMb(response),

  };
}

function datasetChipsFromResponse(response: any, fallbackFilename?: string): DatasetChip[] {
  const source = Array.isArray(response?.datasets) && response.datasets.length
    ? response.datasets
    : response?.dataset_id
      ? [{ ...response, filename: fallbackFilename }]
      : [];
  const byId = new Map<string, DatasetChip>();
  source
    .filter((dataset: any) => typeof dataset?.dataset_id === "string" && dataset.dataset_id)
    .forEach((dataset: any) => {
      if (byId.has(dataset.dataset_id)) return;
      byId.set(dataset.dataset_id, {
        id: dataset.dataset_id,
        name: dataset.alias || dataset.filename || dataset.dataset_id,
        sessionId: dataset.session_id ?? response?.session_id,
        threadId: dataset.thread_id ?? response?.thread_id,
        schema: dataset.schema ?? dataset.columns,
        samples: Array.isArray(dataset.samples)
          ? dataset.samples
          : Array.isArray(dataset.rows)
            ? dataset.rows
            : undefined,
        rowsSampled: dataset.rows_sampled,
        sizeMb: datasetSizeMb(dataset) ?? datasetSizeMb(response),

        visualizationConfig: dataset.visualization_config ?? dataset.visualization_configs,
        visualizationStatus: dataset.visualization_status,
      });

    });
  return Array.from(byId.values());
}

/** Short, readable chip label: tail after the last "/", ellipsized in the middle. */
function datasetChipLabel(name: string): string {
  const tail = String(name ?? "").split("/").filter(Boolean).pop() ?? String(name ?? "");
  if (tail.length <= 26) return tail;
  return `${tail.slice(0, 14)}…${tail.slice(-10)}`;
}

const BATCH_KEY = "analysis:batch";
type StoredBatch = { chips: DatasetChip[]; selectedId: string | null };

function storeDatasetBatch(analysisId: string | null | undefined, batch: StoredBatch) {
  try {
    const raw = JSON.stringify(batch);
    sessionStorage.setItem(BATCH_KEY, raw);
    if (analysisId) sessionStorage.setItem(`${BATCH_KEY}:${analysisId}`, raw);
  } catch {
    /* ignore */
  }
}

function readDatasetBatch(analysisId: string | null | undefined): StoredBatch | null {
  try {
    const raw =
      (analysisId ? sessionStorage.getItem(`${BATCH_KEY}:${analysisId}`) : null) ??
      sessionStorage.getItem(BATCH_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredBatch;
    return Array.isArray(parsed?.chips) && parsed.chips.length ? parsed : null;
  } catch {
    return null;
  }
}

type UserInsightEntry = { id: string; sectionId: string; vizConfig: unknown; samples?: unknown[] };

// Each analysis run's tabular output, persisted per run (parallel to charts).
type TableRun = {
  id: string;
  sectionId: string;
  title: string;
  rows: Record<string, unknown>[];
  /** Code that produced THIS result (from coder_definition.code). */
  code?: string | null;
  /** Version stamp for this result, used by GET /analysis/{id}/code?prompt_ts= */
  promptTs?: string | null;
  /** CSV payload produced by this result. */
  file?: OutputFile | null;
};

type PastedAnalysis = TableRun & {
  /** Chart config copied along with the run (falls back to derived charts). */
  vizConfig?: unknown | null;
  /** Key insights copied from the source analysis tab. */
  extraInsights?: string[];
  /** All results pasted in the same action share this id (one rendered block). */
  groupId?: string;
  /** Ready-made slides captured from the source tab (rendered in a carousel). */
  slides?: unknown[];
};


function tableRunFingerprint(rows: unknown[]): string {
  try {
    return JSON.stringify(rows).slice(0, 5000);
  } catch {
    return String(rows.length);
  }
}

function mergeTableRun(prev: TableRun[], run: TableRun): TableRun[] {
  const fp = tableRunFingerprint(run.rows);
  if (prev.some((r) => r.id === run.id || tableRunFingerprint(r.rows) === fp)) return prev;
  return [...prev, run];
}

/** Clipboard marker used so a pasted analysis blob can be detected safely. */
const AVALOKA_CLIP_MARKER = "__avaloka_analysis_clip__";

type AnalysisClipRun = {
  id: string;
  title: string;
  rows: Record<string, unknown>[];
  code?: string | null;
  promptTs?: string | null;
  file?: OutputFile | null;
  /** Chart config for THIS run, so the paste target renders the same graphs. */
  vizConfig?: unknown | null;
};

type AnalysisClip = {
  marker: typeof AVALOKA_CLIP_MARKER;
  v: 1;
  /** Back-compat: the single run that was visible when copying. */
  run: AnalysisClipRun;
  /** All results of the copied tab, each with its own charts. */
  runs?: AnalysisClipRun[];
  /** Key insights of the copied tab. */
  insights?: string[];
  /** Every chart rendered in the copied tab, serialized as-is. */
  slides?: unknown[];
  fidelity?: string;
  vizConfig?: unknown;
};


function isAnalysisClip(text: string): AnalysisClip | null {
  try {
    const parsed = JSON.parse(text);
    if (parsed && parsed.marker === AVALOKA_CLIP_MARKER && parsed.run && Array.isArray(parsed.run.rows)) {
      return parsed as AnalysisClip;
    }
  } catch {
    /* not JSON / not our clip */
  }
  return null;
}

function vizFingerprint(viz: unknown): string {
  try {
    return JSON.stringify(viz);
  } catch {
    return String(viz);
  }
}

function normalizeInsightTitle(title: unknown): string {
  return String(title ?? "")
    .trim()
    .toLowerCase();
}

function slideFingerprint(slide: Slide): string {
  try {
    return JSON.stringify({
      title: normalizeInsightTitle(slide.title),
      type: slide.type,
      xKey: slide.xKey,
      series: slide.series.map((s) => ({ dataKey: s.dataKey, name: s.name })),
    });
  } catch {
    return normalizeInsightTitle(slide.title) || String(slide.title ?? "chart");
  }
}

function dedupeSlidesByInsight(slides: Slide[], excludeTitles?: Set<string>): Slide[] {
  const seen = new Set<string>();
  const out: Slide[] = [];
  for (const slide of slides) {
    const title = normalizeInsightTitle(slide.title);
    if (title && excludeTitles?.has(title)) continue;
    const key = title || slideFingerprint(slide);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(slide);
  }
  return out;
}

function extractUserInsightsFromMessages(
  rows: AnalysisMessage[],
  autoVizConfig: unknown | null,
  defaultSectionId: string,
): UserInsightEntry[] {
  const autoKey = autoVizConfig ? vizFingerprint(autoVizConfig) : null;
  const results: UserInsightEntry[] = [];
  const seen = new Set<string>();

  for (const row of rows) {
    if (row.role !== "assistant") continue;
    const out = row.output as { viz_config?: unknown; output_json?: unknown[]; section_id?: string } | null;
    if (!out) continue;

    let viz = out.viz_config;
    if (!viz && Array.isArray(out.output_json) && out.output_json.length > 0) {
      viz = deriveVizFromRows(out.output_json);
    }
    if (!viz) continue;

    const key = vizFingerprint(viz);
    if (autoKey && key === autoKey) continue;
    const sectionId = typeof out.section_id === "string" ? out.section_id : defaultSectionId;
    const dedupeKey = `${sectionId}:${key}`;
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    results.push({
      id: row.id,
      sectionId,
      vizConfig: viz,
    });
  }

  return results;
}

// Rebuild the per-run table history from persisted analysis messages so the
// Data table view survives reloads exactly like the charts do.
function extractTableRunsFromMessages(rows: AnalysisMessage[], defaultSectionId: string): TableRun[] {
  let runs: TableRun[] = [];
  let lastPrompt = "";
  for (const row of rows) {
    if (row.role === "user") {
      lastPrompt = String(row.content ?? "").trim();
      continue;
    }
    if (row.role !== "assistant") continue;
    const out = row.output as
      | {
          output_json?: unknown[];
          extracted_tables?: ParsedTable[];
          section_id?: string;
          code?: string | null;
          prompt_ts?: string | null;
          output_file?: OutputFile | null;
        }
      | null;

    // Pasted results are stored as their own message kind and rendered in a
    // dedicated section below the analysis — never as a normal result page.
    if (
      out &&
      ((out as { pasted_analysis?: unknown }).pasted_analysis ||
        (out as { pasted_analyses?: unknown }).pasted_analyses)
    )
      continue;


    // Structured backend rows.
    const hasStructured = !!(out && Array.isArray(out.output_json) && out.output_json.length > 0);
    if (hasStructured) {
      runs = mergeTableRun(runs, {
        id: row.id,
        sectionId: typeof out!.section_id === "string" ? out!.section_id : defaultSectionId,
        title: lastPrompt || `Result ${runs.length + 1}`,
        rows: out!.output_json as Record<string, unknown>[],
        code: typeof out!.code === "string" ? out!.code : null,
        promptTs: typeof out!.prompt_ts === "string" ? out!.prompt_ts : null,
        file: out!.output_file ?? null,
      });
    }

    // Tables embedded in the assistant's markdown content (e.g. "list cloud
    // connections"). Only used when the same reply carried no structured rows —
    // otherwise one request would yield two result pages.
    if (!hasStructured) {
      const extracted = extractTabularContent(row.content ?? "");
      const extractedRows = extracted.tables.flatMap((t) => t.rows);
      if (extractedRows.length > 0) {
        runs = mergeTableRun(runs, {
          id: `${row.id}-tables`,
          sectionId: typeof out?.section_id === "string" ? out.section_id : defaultSectionId,
          title: lastPrompt || `Result ${runs.length + 1}`,
          rows: extractedRows,
          code: null,
          promptTs: null,
          file: null,
        });
      }
    }

  }
  return runs;
}

/** Pasted analyses persisted as assistant messages carrying `pasted_analysis`. */
function extractPastedAnalysesFromMessages(
  rows: AnalysisMessage[],
  defaultSectionId: string,
): PastedAnalysis[] {
  const out: PastedAnalysis[] = [];
  for (const row of rows) {
    if (row.role !== "assistant") continue;
    const raw = (row.output as { pasted_analysis?: unknown; pasted_analyses?: unknown } | null) ?? null;
    const payloads = (
      Array.isArray(raw?.pasted_analyses) ? raw!.pasted_analyses : raw?.pasted_analysis ? [raw.pasted_analysis] : []
    ) as {
      id?: string;
      sectionId?: string;
      title?: string;
      rows?: unknown;
      code?: string | null;
      promptTs?: string | null;
      file?: OutputFile | null;
      vizConfig?: unknown | null;
      extraInsights?: unknown;
      groupId?: string;
      slides?: unknown;
    }[];
    payloads.forEach((payload, i) => {
      if (!payload || !Array.isArray(payload.rows) || payload.rows.length === 0) return;
      out.push({
        id: payload.id || `pasted-${row.id}-${i}`,
        sectionId: payload.sectionId || defaultSectionId,
        title: payload.title || "Pasted result",
        rows: payload.rows as Record<string, unknown>[],
        code: typeof payload.code === "string" ? payload.code : null,
        promptTs: typeof payload.promptTs === "string" ? payload.promptTs : null,
        file: payload.file ?? null,
        vizConfig: payload.vizConfig ?? null,
        extraInsights: Array.isArray(payload.extraInsights) ? (payload.extraInsights as string[]) : undefined,
        groupId: payload.groupId || `pasted-msg-${row.id}`,
        slides: Array.isArray(payload.slides) ? (payload.slides as unknown[]) : undefined,
      });
    });

  }
  return out;
}




function tableRunToCsv(rows: Record<string, unknown>[]): string {
  const keys = new Set<string>();
  rows.slice(0, 200).forEach((r) => Object.keys(r ?? {}).forEach((k) => keys.add(k)));
  const columns = Array.from(keys);
  const escape = (v: unknown) => {
    if (v == null) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.join(","), ...rows.map((r) => columns.map((c) => escape(r[c])).join(","))].join("\n");
}


function extraSectionsFromUserInsights(tabId: string, insights: UserInsightEntry[]): WorkspaceSection[] {
  const mainId = mainSectionId(tabId);
  const extras: WorkspaceSection[] = [];
  const seen = new Set<string>();
  for (const entry of insights) {
    if (entry.sectionId === mainId || seen.has(entry.sectionId)) continue;
    if (!entry.sectionId.startsWith(`${tabId}-`)) continue;
    seen.add(entry.sectionId);
    extras.push({ id: entry.sectionId });
  }
  return extras;
}

// Slides carry the id of the analysis run they came from so the Chart and
// Data table views can stay on the same run.
type RunSlide = Slide & { runId?: string };

function userChartSlidesForSection(
  entries: UserInsightEntry[],
  sectionId: string,
  tab: Pick<DashboardTab, "id" | "tabIndex"> | undefined,
  samples: unknown[] | undefined,
): RunSlide[] {
  const slides: RunSlide[] = [];
  for (const entry of entries) {
    if (entry.sectionId !== sectionId) continue;
    slides.push(
      ...slidesForDashboardTab(entry.vizConfig, entry.samples ?? samples, tab).map(
        (s) => ({ ...s, runId: entry.id }) as RunSlide,
      ),
    );
  }
  return slides;
}

function slidesForDashboardTab(
  vizConfig: unknown,
  samples: unknown[] | undefined,
  tab: Pick<DashboardTab, "id" | "tabIndex"> | undefined,
): Slide[] {
  if (!vizConfig) return [];
  const allSlides = normalizeVizConfig(vizConfig, samples) as Slide[];
  if (!tab) return allSlides;
  const indices = chartIndicesForDashboardTab(vizConfig, tab);
  return indices.map((i) => allSlides[i]).filter((s): s is Slide => s != null);
}

function collectSlidesInsights(slides: Slide[]) {
  const summaries: string[] = [];
  const insights: string[] = [];
  const seen = new Set<string>();
  const add = (list: string[], text?: string | null) => {
    const t = text?.trim();
    if (!t || seen.has(t)) return;
    seen.add(t);
    list.push(t);
  };
  for (const slide of slides) {
    add(summaries, slide.subtitle);
    slide.insights.forEach((item) => add(insights, item));
  }
  return { summaries, insights };
}

const chartData = [
  { month: "Jan", y2020: 48, y2021: -38 },
  { month: "Feb", y2020: 76, y2021: 58 },
  { month: "March", y2020: -20, y2021: 68 },
  { month: "April", y2020: 98, y2021: -50 },
  { month: "May", y2020: 40, y2021: 90 },
  { month: "June", y2020: 72, y2021: 98 },
  { month: "July", y2020: 88, y2021: 78 },
  { month: "Aug", y2020: 42, y2021: -22 },
  { month: "Sept", y2020: 18, y2021: 22 },
  { month: "Oct", y2020: -96, y2021: 80 },
];

function AnalysisPage() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const search = useRouterState({ select: (s) => s.location.search as Record<string, unknown> });
  const aidFromUrl = useMemo(() => {
    const v = search?.aid;
    return typeof v === "string" ? v : null;
  }, [search]);
  const childIdFromUrl = useMemo(() => {
    const v = search?.child;
    return typeof v === "string" ? v : null;
  }, [search]);
  const dsFromUrl = useMemo(() => {
    const v = (search as Record<string, unknown> | undefined)?.ds;
    return typeof v === "string" ? v : null;
  }, [search]);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatOpen, setChatOpen] = useState(true);
  const [chatWidth, setChatWidth] = useState(360);
  const chatAsideRef = useRef<HTMLElement | null>(null);
  const chatResizingRef = useRef(false);
  type ChatDock = "left" | "right" | "float";
  const [chatDock, setChatDock] = useState<ChatDock>(() => {
    if (typeof window === "undefined") return "right";
    const v = window.localStorage.getItem("analysis:chatDock");
    return v === "left" || v === "float" ? v : "right";
  });
  useEffect(() => {
    try {
      window.localStorage.setItem("analysis:chatDock", chatDock);
    } catch {
      /* ignore */
    }
  }, [chatDock]);
  const [chatFloatPos, setChatFloatPos] = useState<{ x: number; y: number }>({ x: 120, y: 96 });
  const [chatFloatSize, setChatFloatSize] = useState<{ w: number; h: number }>({ w: 420, h: 620 });
  const floatDraggingRef = useRef(false);
  const floatAsideRef = useRef<HTMLElement | null>(null);
  const [avatarMode, setAvatarMode] = useState(false);
  const [voiceListening, setVoiceListening] = useState(false);
  const [view, setView] = useState<"table" | "chart" | "preview">("chart");
  
  const [moveOpen, setMoveOpen] = useState(false);
  const { blocked: upgradeBlocked, dialog: upgradeDialog } = useUpgradeGate();

  const [moveSuccess, setMoveSuccess] = useState<string | null>(null);
  const [pane, setPane] = useState<"workspace" | "project">("workspace");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [restorePanelOpen, setRestorePanelOpen] = useState(false);
  const [insightsOpen, setInsightsOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [currentDataset, setCurrentDataset] = useState<CloudDataset | null>(null);
  const [connectCloudOpen, setConnectCloudOpen] = useState(false);
  const [autoInsightsOpen, setAutoInsightsOpen] = useState(false);
  const [autoInsightsConfig, setAutoInsightsConfig] = useState<any>(null);
  const [autoInsightsStatus, setAutoInsightsStatus] = useState<string | undefined>(undefined);
  const [autoInsightsDatasetName, setAutoInsightsDatasetName] = useState<string | undefined>(undefined);

  const [autoInsightsSamples, setAutoInsightsSamples] = useState<any[] | undefined>(undefined);
  // User Insights: latest chart derived from a user prompt reply (separate
  // from the stable Auto Insights generated at upload time).
  const [userInsightsOpen, setUserInsightsOpen] = useState(false);
  const [userInsightsViz, setUserInsightsViz] = useState<any>(null);
  const [userInsightsList, setUserInsightsList] = useState<UserInsightEntry[]>([]);
  const [userInsightsRows, setUserInsightsRows] = useState<Record<string, unknown>[] | null>(null);
  const navState = useRouterState({ select: (s: any) => s?.location?.state }) as any;
  const consumedUploadRef = useRef<string | null>(null);

  // Backend session for chat
  const [threadId, setThreadId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>([]);
  const [analysisFidelity, setAnalysisFidelity] = useState<
    "quick_sample" | "portfolio_samples" | "entire_dataset" | undefined
  >(undefined);
  const [selectedSampleName, setSelectedSampleName] = useState<string | undefined>(undefined);
  const [datasetChips, setDatasetChips] = useState<DatasetChip[]>([]);

  // Analysis output from sendMessage
  const [analysisRows, setAnalysisRows] = useState<Record<string, unknown>[] | null>(null);
  // Every analysis run's table output, kept so nothing is overwritten.
  const [tableRuns, setTableRuns] = useState<TableRun[]>([]);
  // Copied results are previews appended below the destination analysis. They
  // must not enter its result pager or replace its active charts/table.
  const [pastedAnalyses, setPastedAnalyses] = useState<PastedAnalysis[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const pushTableRun = useCallback((run: TableRun) => {
    setTableRuns((prev) => {
      const next = mergeTableRun(prev, run);
      return next;
    });
    setSelectedRunId(run.id);
  }, []);
  /** Patch one result in place (used by Refresh / Save & Execute). */
  const updateTableRun = useCallback((runId: string, patch: Partial<TableRun>) => {
    setTableRuns((prev) => prev.map((r) => (r.id === runId ? { ...r, ...patch } : r)));
  }, []);
  const [analysisVizConfig, setAnalysisVizConfig] = useState<any>(null);
  const [persistedDataset, setPersistedDataset] = useState<UploadedDataset | null>(null);
  const [outputFile, setOutputFile] = useState<OutputFile | null>(null);
  const [isSending, setIsSending] = useState(false);

  // Planner graph: some backends report planner_graph_status/display_url,
  // older ones only return a planner_definition. Treat any of those as a
  // signal that a graph exists, and keep it sticky for the thread (a later
  // turn without the field shouldn't hide an already-available graph).
  const [plannerGraphAvailable, setPlannerGraphAvailable] = useState(false);
  const [plannerGraphUrl, setPlannerGraphUrl] = useState<string | null>(null);
  const [plannerGraphLoading, setPlannerGraphLoading] = useState(false);
  const [plannerGraphError, setPlannerGraphError] = useState<string | null>(null);

  const updatePlannerGraphAvailability = (res: any) => {
    if (!res) return;
    const failed = res.planner_graph_status === "failure";
    const hasGraph =
      (res.planner_graph_status === "success" && !!res.planner_graph_display_url) ||
      !!res.planner_graph_display_url ||
      !!res.planner_definition;
    if (hasGraph && !failed) setPlannerGraphAvailable(true);
  };


  // Auth-protected PNG — fetch as blob (a plain <img src> can't send the
  // bearer token), then show it via an object URL in a lightbox.
  const openPlannerGraph = async () => {
    if (!threadId || plannerGraphLoading) return;
    setPlannerGraphLoading(true);
    setPlannerGraphError(null);
    try {
      const blob = await backendApi.getPlannerGraph(threadId);
      const objectUrl = URL.createObjectURL(blob);
      setPlannerGraphUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return objectUrl;
      });
    } catch (err: any) {
      if (err?.status === 404) {
        // No planner graph for this thread yet — hide the button.
        setPlannerGraphAvailable(false);
        setPlannerGraphError("No planner graph available for this analysis yet.");

      } else if (err?.status === 401) {
        setPlannerGraphError("Your session has expired — please sign in again.");
      } else {
        setPlannerGraphError(err?.message || "Could not load the planner graph.");
      }
    } finally {
      setPlannerGraphLoading(false);
    }
  };

  const closePlannerGraph = () => {
    setPlannerGraphUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setPlannerGraphError(null);
  };

  // Voice mode reads the assistant's newest reply aloud inside <VoicePanel />.
  useEffect(() => {
    if (!avatarMode) stopSpeaking();
  }, [avatarMode]);




  // Persistence: the analysis_id is the stable key for this chat thread.
  // It comes from Postgres (gen_random_uuid()) on the first insert into
  // `analyses`, is mirrored into the URL as ?aid=..., and is then used for
  // every analysis_messages row we write or read.
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [motherAnalysisId, setMotherAnalysisId] = useState<string | null>(null);
  const [motherAnalysisName, setMotherAnalysisName] = useState("");
  const [childAnalyses, setChildAnalyses] = useState<ChildAnalysisSummary[]>([]);
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(null);
  const [childUploadOpen, setChildUploadOpen] = useState(false);
  const [childPickerOpen, setChildPickerOpen] = useState(false);
  const [threadInsightSources, setThreadInsightSources] = useState<InsightSource[]>([]);
  const [dismissedInsightCardsByAnalysis, setDismissedInsightCardsByAnalysis] = useState<Record<string, string[]>>({});
  const tabHydrateTokenRef = useRef(0);
  const [hydrating, setHydrating] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return new URLSearchParams(window.location.search).has("aid");
  });
  const isPersistedAnalysis = Boolean(aidFromUrl ?? analysisId);

  const chartSamples = useMemo(
    () => autoInsightsSamples ?? persistedDataset?.samples ?? undefined,
    [autoInsightsSamples, persistedDataset],
  );

  const selectDatasetChip = useCallback(
    (dataset: DatasetChip, opts?: { silent?: boolean }) => {
      // Toggle membership only — the uploaded group (datasetChips) is the source
      // of truth and must never be replaced by a single dataset.
      let nowSelected = true;
      setSelectedDatasetIds((prev) => {
        if (prev.includes(dataset.id)) {
          // Keep at least one dataset selected.
          if (prev.length <= 1) {
            nowSelected = true;
            return prev;
          }
          nowSelected = false;
          return prev.filter((id) => id !== dataset.id);
        }
        nowSelected = true;
        return [...prev, dataset.id];
      });

      if (dataset.sessionId) setSessionId(dataset.sessionId);
      if (dataset.threadId) setThreadId(dataset.threadId);

      setDatasetChips((chips) => {
        storeDatasetBatch(analysisId ?? aidFromUrl ?? null, { chips, selectedId: dataset.id });
        return chips;
      });

      if (!nowSelected) return;

      const label = datasetChipLabel(dataset.name);
      setAutoInsightsDatasetName(dataset.name);
      setCtx((prev) => (prev ? { ...prev, name: label } : prev));

      if (!dataset.samples?.length) {
        // No inline rows for THIS dataset — fetch its own preview using its own
        // dataset_id (never the primary/first one) so the tab can't show
        // another dataset's rows.
        const sess = dataset.sessionId ?? sessionId ?? null;
        setPersistedDataset(null);
        setAutoInsightsSamples(undefined);
        (async () => {
          try {
            const preview: any = sess
              ? await backendApi.getDatasetPreview(dataset.id, sess)
              : await backendApi.previewUploadedDataset(dataset.id);
            const rows: any[] = Array.isArray(preview?.samples)
              ? preview.samples
              : Array.isArray(preview?.rows)
                ? preview.rows
                : [];
            if (!rows.length) return;
            const cols = normalizeDatasetSchema(preview?.schema ?? preview?.columns, rows);
            const cfg =
              preview?.visualization_config ??
              preview?.visualization_configs ??
              deriveVizFromRows(rows);
            setDatasetChips((chips) =>
              chips.map((c) =>
                c.id === dataset.id
                  ? { ...c, samples: rows, schema: cols, visualizationConfig: cfg, visualizationStatus: "ready" }
                  : c,
              ),
            );
            setPersistedDataset({
              filename: dataset.name,
              schema: cols,
              samples: rows,
              rows_sampled: preview?.rows_sampled ?? rows.length,
              size_mb: dataset.sizeMb,
            });
            setAutoInsightsSamples(rows);
            setAutoInsightsConfig(cfg);
            setAnalysisVizConfig(cfg);
            setAutoInsightsStatus(cfg ? "ready" : undefined);
          } catch (err) {
            console.warn("[dataset-tab] preview failed", dataset.id, err);
          }
        })();
        return;
      }

      const schema = normalizeDatasetSchema(dataset.schema, dataset.samples);
      setPersistedDataset({
        filename: dataset.name,
        schema,
        samples: dataset.samples,
        rows_sampled: dataset.rowsSampled ?? dataset.samples.length,
        size_mb: dataset.sizeMb,

      });
      setAutoInsightsSamples(dataset.samples);

      const config = dataset.visualizationConfig ?? deriveVizFromRows(dataset.samples);
      setAutoInsightsConfig(config);
      setAnalysisVizConfig(config);
      setAutoInsightsStatus(dataset.visualizationStatus ?? (config ? "ready" : undefined));
    },
    [analysisId, aidFromUrl, sessionId],
  );


  /** Per-dataset tabs for the Auto Insights modal (grouped uploads).
   *  One entry per DISTINCT dataset_id, each carrying its OWN config + OWN rows.
   *  Never pairs one dataset's chart spec with another dataset's rows. */
  const autoInsightsDatasets = useMemo(() => {
    const byId = new Map<string, { id: string; name: string; config: any; status?: string; samples?: any[] }>();
    datasetChips.forEach((d) => {
      if (!d?.id || byId.has(d.id)) return;
      // Own rows only: chip samples, or the persisted preview when it belongs
      // to THIS dataset (same filename), otherwise none.
      const ownSamples =
        (Array.isArray(d.samples) && d.samples.length ? d.samples : undefined) ??
        (persistedDataset?.filename && persistedDataset.filename === d.name
          ? persistedDataset.samples
          : undefined);
      const ownConfig =
        d.visualizationConfig ??
        (ownSamples?.length ? deriveVizFromRows(ownSamples) : null);
      byId.set(d.id, {
        id: d.id,
        name: d.name,
        config: ownConfig,
        status: d.visualizationStatus ?? (ownConfig ? "ready" : undefined),
        samples: ownSamples,
      });
    });
    return Array.from(byId.values());
  }, [datasetChips, persistedDataset]);



  const vizInsights = useMemo(
    () => collectVizInsights(analysisVizConfig, chartSamples),
    [analysisVizConfig, chartSamples],
  );
  const hasBackendInsights = vizInsights.summaries.length > 0 || vizInsights.insights.length > 0;

  // ── Auto Insights bootstrap ────────────────────────────────────────────────
  // A brand-new analysis (e.g. opened from "Open for analysis") has no saved
  // viz_config, so the Chart panel and Key Insights render empty. Generate the
  // insights once for the selected dataset and persist them. Never runs when
  // the analysis already has saved insights — existing ones are never
  // overwritten — and never posts any chat message.
  const autoInsightsBootstrapRef = useRef<string | null>(null);
  const [insightsGenerating, setInsightsGenerating] = useState(false);
  useEffect(() => {
    const aid = analysisId ?? aidFromUrl ?? null;
    if (!aid || hydrating) return;
    if (autoInsightsBootstrapRef.current === aid) return;
    // Already has insights → leave untouched.
    if (autoInsightsConfig || analysisVizConfig || userInsightsList.length) return;
    const dsId = selectedDatasetIds[selectedDatasetIds.length - 1] ?? datasetChips[0]?.id ?? null;
    const samples = chartSamples;
    if (!dsId && !samples?.length) return;

    autoInsightsBootstrapRef.current = aid;
    let cancelled = false;
    setInsightsGenerating(true);
    (async () => {
      let cfg: any = null;
      let rowsFromPreview: any[] | null = null;
      try {
        if (dsId) {
          // Same generation source the upload flow uses: the backend's
          // visualization config for this dataset. It can come back
          // "pending" while the backend is still generating, so poll.
          for (let attempt = 0; attempt < 8 && !cancelled; attempt++) {
            let preview: any = null;
            try {
              preview = sessionId
                ? await backendApi.getDatasetPreview(dsId, sessionId, attempt > 0)
                : await backendApi.previewUploadedDataset(dsId);
            } catch {
              try {
                preview = await backendApi.previewUploadedDataset(dsId);
              } catch {
                preview = null;
              }
            }
            const rows = Array.isArray(preview?.samples)
              ? preview.samples
              : Array.isArray(preview?.rows)
                ? preview.rows
                : null;
            if (rows?.length) rowsFromPreview = rows;
            const candidate = preview?.visualization_config ?? preview?.visualization_configs ?? null;
            const status = String(preview?.visualization_status ?? "").toLowerCase();
            if (candidate) {
              cfg = candidate;
              break;
            }
            if (status !== "pending" && status !== "processing" && status !== "running") break;
            await new Promise((r) => setTimeout(r, 2500));
          }
        }
        if (!cfg && rowsFromPreview?.length) cfg = deriveVizFromRows(rowsFromPreview);
        if (!cfg && samples?.length) cfg = deriveVizFromRows(samples);
        if (cancelled || !cfg) return;
        setAutoInsightsConfig(cfg);
        setAutoInsightsStatus("ready");
        setAnalysisVizConfig((prev: any) => prev ?? cfg);
        setView("chart");
        persistUpdateAnalysisContext(aid, { viz_config: cfg }).catch((err) =>
          console.warn("[auto-insights] could not persist viz_config", err),
        );
      } finally {
        if (!cancelled) setInsightsGenerating(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    analysisId,
    aidFromUrl,
    hydrating,
    autoInsightsConfig,
    analysisVizConfig,
    userInsightsList.length,
    selectedDatasetIds,
    datasetChips,
    chartSamples,
    sessionId,
  ]);



  // Restore stored session for fresh uploads only (not when opening ?aid= from Supabase).
  useEffect(() => {
    if (typeof window === "undefined" || aidFromUrl) return;
    try {
      const raw = sessionStorage.getItem("analysis:session");
      if (raw) {
        const s = JSON.parse(raw);
        if (s?.threadId) setThreadId(s.threadId);
        if (s?.sessionId) setSessionId(s.sessionId);
        if (Array.isArray(s?.datasetChips)) setDatasetChips(s.datasetChips);
      }
      const batch = readDatasetBatch(null);
      if (batch) {
        setDatasetChips(batch.chips);
        const active = batch.chips.find((c) => c.id === batch.selectedId) ?? batch.chips[0];
        if (active) selectDatasetChip(active, { silent: true });
      }
    } catch {}
  }, [aidFromUrl, selectDatasetChip]);

  // ?ds=<dataset_id> focuses one dataset of the group (e.g. from the Recent
  // sidebar file list) WITHOUT dropping the other selected datasets.
  const focusedDsRef = useRef<string | null>(null);
  useEffect(() => {
    if (!dsFromUrl || focusedDsRef.current === dsFromUrl) return;
    const chip = datasetChips.find((c) => c.id === dsFromUrl);
    if (!chip) return;
    focusedDsRef.current = dsFromUrl;
    setSelectedDatasetIds((prev) => (prev.includes(dsFromUrl) ? prev : [...prev, dsFromUrl]));
    if (!selectedDatasetIds.includes(dsFromUrl)) {
      selectDatasetChip(chip, { silent: true });
    } else {
      setAutoInsightsDatasetName(chip.name);
    }
  }, [dsFromUrl, datasetChips, selectedDatasetIds, selectDatasetChip]);

  // Restore chart config after refresh for fresh uploads only.
  useEffect(() => {
    if (typeof window === "undefined" || aidFromUrl) return;
    try {
      const raw = sessionStorage.getItem("analysis:dataset");
      if (!raw) return;
      const parsed = JSON.parse(raw);
      const cfg = parsed?.visualization_config;
      const status = parsed?.visualization_status as string | undefined;
      if (cfg && (!status || status === "ready")) {
        setAnalysisVizConfig((prev: any) => prev ?? cfg);
        if (Array.isArray(parsed.samples)) {
          setAutoInsightsSamples((prev: any) => prev ?? parsed.samples);
        }
      }
    } catch {}
  }, [aidFromUrl]);

  // Open the Auto Insights popup immediately on arrival from a fresh upload,
  // before the (slower) upload-binding effect finishes, so the dataset view is
  // never visible in between.
  useEffect(() => {
    if ((navState as any)?.showAutoInsights) setAutoInsightsOpen(true);
  }, [navState]);

  useEffect(() => {
    const up = navState?.uploadResponse;
    if (!up) return;
    const key = up.dataset_id || up.thread_id || JSON.stringify(up).slice(0, 64);
    if (consumedUploadRef.current === key) return;

    // Database table imports and standalone uploads can navigate here while a
    // project-scoped analysis is already mounted. Clear that old project
    // identity immediately so the destination never inherits the compact
    // project rail, project subtitle, or project-only toolbar while hydrating.
    if ((navState as any)?.standaloneAnalysis) {
      setLoadedProjectId(null);
      setCtx({
        name: String(navState?.filename ?? "New Analysis").replace(/\.[^.]+$/, ""),
        project: "",
        projectId: null,
        analysisId: ((navState as any)?.newAnalysisId as string | null | undefined) ?? null,
      });
      setPane("workspace");
    }

    // A fresh upload / cloud-register / DB-query result trumps any stale ?aid=
    // still sitting in the URL. If the caller pre-created an analyses row and
    // put its id in navState.newAnalysisId, keep that aid; otherwise strip
    // the stale aid before we bind so the hydrator can't reload the previous
    // analysis on top of the one the user just chose.
    const freshAid = (navState as any)?.newAnalysisId as string | undefined;
    if ((aidFromUrl && aidFromUrl !== freshAid) || childIdFromUrl) {
      navigate({
        to: "/analysis",
        search: (freshAid ? { aid: freshAid } : {}) as any,
        replace: true,
        state: navState as any,
      });
      return; // re-run once the URL matches
    }

    // Drop cached previews for the previous dataset and clear any stale
    // dataset / session / analysis state so nothing bleeds into the new bind.
    try { backendApi.abortAllPreviews(); } catch { /* ignore */ }
    try {
      sessionStorage.removeItem("analysis:session");
      sessionStorage.removeItem("analysis:dataset");
    } catch { /* ignore */ }
    setAnalysisId(null);
    setMotherAnalysisId(null);
    setMotherAnalysisName("");
    setChildAnalyses([]);
    setActiveAnalysisId(null);
    setMessages([]);
    setUserInsightsList([]);
    setUserInsightsViz(null);
    setUserInsightsRows(null);
    setAnalysisRows(null);
    setTableRuns([]);
    setPastedAnalyses([]);
    setSelectedRunId(null);
    setPersistedDataset(null);
    setDatasetChips([]);
    setSelectedDatasetIds([]);
    setThreadId(null);
    setSessionId(null);

    consumedUploadRef.current = key;
    const populatedDataset = populatedDatasetFromResponse(up, navState?.filename ?? up.dataset_id);
    const cfg =
      up.visualization_config ??
      up.visualization_configs ??
      (populatedDataset ? deriveVizFromRows(populatedDataset.samples) : null);
    const status = up.visualization_status as string | undefined;
    setAutoInsightsConfig(cfg);
    setAutoInsightsStatus(status ?? (cfg ? "ready" : undefined));
    setAutoInsightsDatasetName(navState?.filename ?? up.dataset_id);
    setAutoInsightsSamples(populatedDataset?.samples);
    setPersistedDataset(populatedDataset);
    if (cfg && (!status || status === "ready")) {
      setAnalysisVizConfig(cfg);
      setView("chart");
    } else if (populatedDataset) {
      setAnalysisVizConfig(null);
      setView("table");
    }

    // Fresh uploads that asked for it land straight on the Auto Insights cards.
    if ((navState as any)?.showAutoInsights) {
      setAutoInsightsOpen(true);
    }


    // Bind the chat session to the identifiers returned by THIS response.
    if (up.thread_id) setThreadId(up.thread_id);
    if (up.session_id) setSessionId(up.session_id);
    if (up.analysis_fidelity) setAnalysisFidelity(up.analysis_fidelity);
    if (up.selected_sample_name) setSelectedSampleName(up.selected_sample_name);
    const chips = datasetChipsFromResponse(up, navState?.filename);
    if (chips.length) {
      const activeId = up.dataset_id || chips[0].id;
      setDatasetChips(chips);
      // Select every dataset from a multi-file upload so the first chat call
      // carries all dataset_ids and cross-dataset compare/join works.
      setSelectedDatasetIds(chips.map((c) => c.id));
      storeDatasetBatch((navState as any)?.newAnalysisId ?? aidFromUrl ?? null, {
        chips,
        selectedId: activeId,
      });
    }

  }, [navState, aidFromUrl, childIdFromUrl, navigate]);

  const [addToDashboardOpen, setAddToDashboardOpen] = useState(false);
  const [dashboardChartIndex, setDashboardChartIndex] = useState(0);
  const [codeOpen, setCodeOpen] = useState(false);
  const [codeText, setCodeText] = useState("");
  const [originalCodeText, setOriginalCodeText] = useState("");
  const [codeLoading, setCodeLoading] = useState(false);
  const [codeDownloadUrl, setCodeDownloadUrl] = useState<string | null>(null);
  // Which result the "Analysis code" panel is currently bound to, plus any
  // unsaved edits kept per result so switching pages never mixes scripts.
  const [codeRunId, setCodeRunId] = useState<string | null>(null);
  const [runCodeEdits, setRunCodeEdits] = useState<Record<string, string>>({});
  const [insightFeedback, setInsightFeedback] = useState<"positive" | "negative" | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState<"positive" | "negative" | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [savingExec, setSavingExec] = useState(false);
  const [plannerNotes, setPlannerNotes] = useState<string | null>(null);
  const [validationFeedback, setValidationFeedback] = useState<string | null>(null);
  const [newVersionPromptTs, setNewVersionPromptTs] = useState<string | null>(null);
  void newVersionPromptTs;

  type VersionEntry = {
    prompt_ts: string;
    code_object_key?: string | null;
    output_object_key?: string | null;
    viz_object_key?: string | null;
    is_current?: boolean;
    has_code?: boolean;
    has_output?: boolean;
    has_viz?: boolean;
  };
  const [versionsOpen, setVersionsOpen] = useState(false);
  const [versions, setVersions] = useState<VersionEntry[]>([]);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [restoringTs, setRestoringTs] = useState<string | null>(null);

  const formatPromptTs = (ts: string) => {
    // ts format: YYYYMMDD_HHMMSS
    const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(ts);
    if (!m) return ts;
    const [, y, mo, d, h, mi] = m;
    const dt = new Date(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi));
    return dt.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  };

  const fetchVersions = async () => {
    const aid = resolvedAnalysisId ?? new URLSearchParams(window.location.search).get("aid");
    if (!aid) return;
    const { data: sessionData } = await supabase.auth.getSession();
    const token = sessionData.session?.access_token;
    if (!token) return;
    setVersionsLoading(true);
    try {
      const res = await fetch("/api/public/backend-proxy", {
        headers: {
          Authorization: `Bearer ${token}`,
          "X-Backend-Path": `/analysis/${aid}/versions`,
          "X-Backend-Base": BACKEND_CODE_BASE,
        },
      });
      if (!res.ok) {
        setVersions([]);
        return;
      }
      const body = await res.json();
      setVersions(Array.isArray(body?.versions) ? body.versions : []);
    } catch {
      setVersions([]);
    } finally {
      setVersionsLoading(false);
    }
  };

  const handleRestore = async (promptTs: string) => {
    const aid = resolvedAnalysisId ?? new URLSearchParams(window.location.search).get("aid");
    if (!aid) return;
    const { data: sessionData } = await supabase.auth.getSession();
    const token = sessionData.session?.access_token;
    if (!token) {
      toast.error("Please sign in.");
      return;
    }
    setRestoringTs(promptTs);
    try {
      const res = await fetch("/api/public/backend-proxy", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "X-Backend-Path": `/analysis/${aid}/restore`,
          "X-Backend-Base": BACKEND_CODE_BASE,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt_ts: promptTs }),
      });
      let body: any = null;
      try {
        body = await res.json();
      } catch {}
      if (!res.ok || body?.available === false) {
        toast.error(body?.detail || `Restore failed (${res.status})`);
        return;
      }
      if (typeof body?.code === "string") {
        setCodeText(body.code);
        setOriginalCodeText(body.code);
      }
      if (body?.output_signed_url) {
        try {
          const r = await fetch(body.output_signed_url);
          if (r.ok) {
            const data = await r.json();
            const rows = Array.isArray(data) ? data : Array.isArray(data?.output_json) ? data.output_json : null;
            if (rows) setAnalysisRows(rows as Record<string, unknown>[]);
          }
        } catch {}
      }
      if (body?.viz_signed_url) {
        try {
          const r = await fetch(body.viz_signed_url);
          if (r.ok) {
            const cfg = await r.json();
            setAnalysisVizConfig(cfg);
          }
        } catch {}
      }
      toast.success(`Restored to version from ${formatPromptTs(body?.restored_prompt_ts ?? promptTs)}`);
      await fetchVersions();
    } catch (err: any) {
      toast.error(err?.message || "Restore failed");
    } finally {
      setRestoringTs(null);
    }
  };

  // A restore triggered from the sidebar's version history panel should refresh
  // the code panel, data table and chart of the analysis currently open here.
  useEffect(() => {
    const onRestored = async (event: Event) => {
      const detail = (event as CustomEvent<VersionRestoredDetail>).detail;
      const aid = new URLSearchParams(window.location.search).get("aid");
      if (!detail || !aid || detail.analysisId !== aid) return;
      const body = detail.result;
      if (typeof body.code === "string") {
        setCodeText(body.code);
        setOriginalCodeText(body.code);
      }
      if (body.output_signed_url) {
        try {
          const r = await fetch(body.output_signed_url);
          if (r.ok) {
            const data = await r.json();
            const rows = Array.isArray(data) ? data : Array.isArray(data?.output_json) ? data.output_json : null;
            if (rows) setAnalysisRows(rows as Record<string, unknown>[]);
          }
        } catch {}
      }
      if (body.viz_signed_url) {
        try {
          const r = await fetch(body.viz_signed_url);
          if (r.ok) setAnalysisVizConfig(await r.json());
        } catch {}
      }
      if (versionsOpen) void fetchVersions();
    };
    window.addEventListener(VERSION_RESTORED_EVENT, onRestored as EventListener);
    return () => window.removeEventListener(VERSION_RESTORED_EVENT, onRestored as EventListener);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [versionsOpen]);



  const handleRefresh = async () => {
    if (refreshing) return;
    let aid = resolvedAnalysisId ?? new URLSearchParams(window.location.search).get("aid");
    if (!aid) aid = await ensureAnalysisId();
    if (!aid) {
      toast.error("No analysis to refresh.");
      return;
    }
    const { data: sessionData } = await supabase.auth.getSession();
    const token = sessionData.session?.access_token;
    if (!token) {
      toast.error("Please sign in.");
      return;
    }
    const path = `/analysis/${aid}/refresh`;
    setRefreshing(true);
    try {
      const res = await fetch("/api/public/backend-proxy", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "X-Backend-Path": path, "X-Backend-Base": BACKEND_CODE_BASE },
      });
      if (res.status === 404) {
        toast.message("No code to refresh for this analysis.");
        return;
      }
      let body: any = null;
      try {
        body = await res.json();
      } catch {}
      if (!res.ok) {
        toast.error(body?.detail || `Refresh failed (${res.status})`);
        return;
      }
      const { status, output_json, output_file_data, new_version_prompt_ts, checkpoint_created } = body ?? {};
      if (new_version_prompt_ts) setNewVersionPromptTs(new_version_prompt_ts);
      if (Array.isArray(output_json) && output_json.length > 0) {
        setAnalysisRows(output_json as Record<string, unknown>[]);
        const file =
          output_file_data?.content && output_file_data?.filename
            ? { filename: output_file_data.filename as string, content: output_file_data.content as string }
            : null;
        const targetRunId = selectedRunId;
        if (targetRunId && tableRuns.some((r) => r.id === targetRunId)) {
          // Refresh re-runs THIS result — update it in place.
          updateTableRun(targetRunId, {
            rows: output_json as Record<string, unknown>[],
            code: typeof body?.code === "string" ? body.code : undefined,
            promptTs: new_version_prompt_ts ?? undefined,
            file,
          });
        } else {
          pushTableRun({
            id: `refresh-${Date.now()}`,
            sectionId: activeSectionId ?? "",
            title: "Refreshed result",
            rows: output_json as Record<string, unknown>[],
            promptTs: new_version_prompt_ts ?? null,
            file,
          });
        }
        if (file) setOutputFile(file);

        if (checkpoint_created === false) {
          toast.message("No changes since last run");
        } else {
          toast.success("Refreshed");
          if (versionsOpen) void fetchVersions();
        }
      } else if (status === "error") {
        toast.error("Refresh failed — code did not run successfully");
      } else {
        toast.message("Refresh ran but produced no output");
      }
    } catch (err: any) {
      toast.error(err?.message || "Refresh failed");
    } finally {
      setRefreshing(false);
    }
  };

  const handleSaveAndExecute = async () => {
    if (savingExec) return;
    let aid = resolvedAnalysisId ?? new URLSearchParams(window.location.search).get("aid");
    if (!aid) aid = await ensureAnalysisId();
    if (!aid) {
      toast.error("No analysis to execute.");
      return;
    }
    const { data: sessionData } = await supabase.auth.getSession();
    const token = sessionData.session?.access_token;
    if (!token) {
      toast.error("Please sign in.");
      return;
    }
    setSavingExec(true);
    setPlannerNotes(null);
    setValidationFeedback(null);
    try {
      const res = await fetch("/api/public/backend-proxy", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "X-Backend-Path": `/analysis/${aid}/save-and-execute`,
          "X-Backend-Base": BACKEND_CODE_BASE,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ code: codeText }),
      });
      let body: any = null;
      try {
        body = await res.json();
      } catch {}
      if (!res.ok) {
        toast.error(body?.detail || "Save & Execute failed");
        return;
      }
      const {
        status,
        planner_status,
        planner_notes,
        final_code,
        corrected_code,
        validation_passed,
        validation_feedback,
        output_json,
        output_file_data,
        new_version_prompt_ts,
      } = body ?? {};

      const plannerCorrected = planner_status === "corrected";
      if (plannerCorrected) {
        setPlannerNotes(typeof planner_notes === "string" && planner_notes ? planner_notes : "Planner adjusted the code before running.");
        if (typeof final_code === "string" && final_code) {
          setCodeText(final_code);
        }
      }

      if (status === "validation_failed" || validation_passed === false) {
        setValidationFeedback(typeof validation_feedback === "string" ? validation_feedback : "Validation failed.");
        if (typeof corrected_code === "string" && corrected_code) {
          setCodeText(corrected_code);
        }
        toast.error("Couldn't run — the code still needs fixing");
        return;
      }

      // Success path — applies to the currently viewed result only.
      const savedCode = typeof final_code === "string" && final_code ? final_code : codeText;
      const file =
        output_file_data?.content && output_file_data?.filename
          ? { filename: output_file_data.filename as string, content: output_file_data.content as string }
          : null;
      const targetRunId = codeRunId ?? selectedRunId;
      if (Array.isArray(output_json)) {
        setAnalysisRows(output_json as Record<string, unknown>[]);
        if (targetRunId && tableRuns.some((r) => r.id === targetRunId)) {
          updateTableRun(targetRunId, {
            rows: output_json as Record<string, unknown>[],
            code: savedCode,
            promptTs: new_version_prompt_ts ?? undefined,
            file,
          });
          setSelectedRunId(targetRunId);
        } else {
          pushTableRun({
            id: `save-exec-${Date.now()}`,
            sectionId: activeSectionId ?? "",
            title: "Save & Execute result",
            rows: output_json as Record<string, unknown>[],
            code: savedCode,
            promptTs: new_version_prompt_ts ?? null,
            file,
          });
        }
      } else if (targetRunId) {
        updateTableRun(targetRunId, { code: savedCode });
      }
      if (file) setOutputFile(file);
      // Mark this run's code as the new "saved" baseline so dirty clears.
      setOriginalCodeText(savedCode);
      setCodeText(savedCode);
      if (targetRunId) {
        setRunCodeEdits((prev) => {
          const next = { ...prev };
          delete next[targetRunId];
          return next;
        });
      }

      if (new_version_prompt_ts) setNewVersionPromptTs(new_version_prompt_ts);
      await fetchVersions();
      toast.success(
        plannerCorrected
          ? "Saved & executed — new version created (planner corrected your code)"
          : "Saved & executed — new version created",
      );
    } catch (err: any) {
      toast.error(err?.message || "Save & Execute failed");
    } finally {
      setSavingExec(false);
    }
  };

  const sendInsightFeedback = async (
    feedbackType: "positive" | "negative",
    comment: string | null,
  ): Promise<boolean> => {
    let aid = resolvedAnalysisId ?? new URLSearchParams(window.location.search).get("aid");
    if (!aid) aid = await ensureAnalysisId();
    if (!aid) {
      toast.error("No analysis id in URL");
      return false;
    }
    const { data: sessionData } = await supabase.auth.getSession();
    const token = sessionData.session?.access_token;
    if (!token) {
      toast.error("Please log in");
      return false;
    }
    const insightId =
      (workspaceHeading || "key-insights")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "") || "key-insights";
    const path = `/analysis/${aid}/feedback`;
    const body = { insight_id: insightId, feedback_type: feedbackType, comment };
    console.log("[insight-feedback] POST", path, body);
    try {
      const res = await fetch("/api/public/backend-proxy", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
          "X-Backend-Path": path,
          "X-Backend-Base": BACKEND_CODE_BASE,
        },
        body: JSON.stringify(body),
      });
      console.log("[insight-feedback] status", res.status);
      if (!res.ok) {
        let detail = "";
        try {
          detail = (await res.json())?.detail;
        } catch {}
        toast.error(detail || `Feedback failed (${res.status})`);
        return false;
      }
      return true;
    } catch (err: any) {
      toast.error(err?.message || "Feedback failed");
      return false;
    }
  };

  const handleThumb = async (type: "positive" | "negative") => {
    if (feedbackSubmitting) return;
    setFeedbackSubmitting(type);
    const ok = await sendInsightFeedback(type, null);
    if (ok) {
      setInsightFeedback(type);
      toast.success(type === "positive" ? "Marked as helpful" : "Marked as not helpful");
    }
    setFeedbackSubmitting(null);
  };


  /**
   * Load the code belonging to ONE result. Prefers the code returned with
   * that result (coder_definition.code); falls back to
   * GET /analysis/{aid}/code[?prompt_ts=] — never a thread-wide value.
   */
  const loadCodeForRun = async (runId: string | null, opts?: { silent?: boolean }) => {
    setCodeRunId(runId);
    // Banners belong to a single result.
    setPlannerNotes(null);
    setValidationFeedback(null);
    setCodeDownloadUrl(null);

    const run = runId ? tableRuns.find((r) => r.id === runId) : undefined;
    const edited = runId ? runCodeEdits[runId] : undefined;

    if (run?.code) {
      setOriginalCodeText(run.code);
      setCodeText(edited ?? run.code);
      setCodeLoading(false);
      return true;
    }

    let aid = resolvedAnalysisId ?? new URLSearchParams(window.location.search).get("aid");
    if (!aid) aid = await ensureAnalysisId();
    if (!aid) {
      if (!opts?.silent) toast.error("No analysis id in URL");
      return false;
    }
    const { data: sessionData } = await supabase.auth.getSession();
    const token = sessionData.session?.access_token;
    if (!token) {
      if (!opts?.silent) toast.error("Please log in");
      return false;
    }
    const qs = run?.promptTs ? `?prompt_ts=${encodeURIComponent(run.promptTs)}` : "";
    const path = `/analysis/${aid}/code${qs}`;
    setCodeLoading(true);
    setCodeText("");
    setOriginalCodeText("");
    try {
      const res = await fetch("/api/public/backend-proxy", {
        headers: { Authorization: `Bearer ${token}`, "X-Backend-Path": path, "X-Backend-Base": BACKEND_CODE_BASE },
      });
      if (res.status === 404) {
        if (!opts?.silent) toast.message("No code available for this result yet.");
        return false;
      }
      if (!res.ok) {
        let detail = "";
        try {
          detail = (await res.json())?.detail;
        } catch {}
        if (!opts?.silent) toast.error(detail || "Failed to load code");
        return false;
      }
      const body = (await res.json()) as { available?: boolean; code?: string; signed_url?: string };
      if (body.available === false || !body.code) {
        if (!opts?.silent) toast.message("No code available for this result yet.");
        return false;
      }
      if (runId) updateTableRun(runId, { code: body.code });
      setOriginalCodeText(body.code);
      setCodeText(edited ?? body.code);
      setCodeDownloadUrl(body.signed_url ?? null);
      return true;
    } catch (err: any) {
      if (!opts?.silent) toast.error(err?.message || "Failed to load code");
      return false;
    } finally {
      setCodeLoading(false);
    }
  };

  const handleOpenCode = async () => {
    if (codeOpen) {
      setCodeOpen(false);
      return;
    }
    setCodeOpen(true);
    const ok = await loadCodeForRun(selectedRunId);
    if (!ok) setCodeOpen(false);
  };

  /** Track edits per result so each panel keeps its own draft. */
  const handleCodeChange = (next: string) => {
    setCodeText(next);
    if (codeRunId) setRunCodeEdits((prev) => ({ ...prev, [codeRunId]: next }));
  };

  const [commentsOpen, setCommentsOpen] = useState(false);
  const [shareCollaboratorsOpen, setShareCollaboratorsOpen] = useState(false);
  const [collaborationRevision, setCollaborationRevision] = useState(0);
  const [collaborationDetail, setCollaborationDetail] = useState<Extract<
    HistoryViewPayload,
    { kind: "share" } | { kind: "collaborator" }
  > | null>(null);
  const handleCollaborationChange = useCallback(() => {
    setCollaborationRevision((n) => n + 1);
  }, []);
  const [insightCommentOpen, setInsightCommentOpen] = useState(false);
  const [replyContext, setReplyContext] = useState<{ label: string; content: string } | null>(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [ctx, setCtx] = useState<{
    name: string;
    project: string;
    projectId?: string | null;
    analysisId?: string | null;
  } | null>(null);
  const [loadedProjectId, setLoadedProjectId] = useState<string | null>(null);
  const [activeTabId, setActiveTabId] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [extraSectionsByTab, setExtraSectionsByTab] = useState<Record<string, WorkspaceSection[]>>({});
  const [activeSectionId, setActiveSectionId] = useState("");
  const activeSectionIdRef = useRef("");
  const resolvedTabIdRef = useRef("");

  const focusSection = useCallback((sectionId: string) => {
    activeSectionIdRef.current = sectionId;
    setActiveSectionId(sectionId);
    sectionRefs.current[sectionId]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const sectionHighlightClass = useCallback(
    (_sectionId: string) => "border-transparent",
    [],
  );


  // When an analysis is loaded via ?aid=, the row in Supabase is the source of
  // truth for project_id (a stale ctx from sessionStorage must not promote a
  // standalone analysis into a project one).
  const resolvedProjectId = analysisId ? loadedProjectId : (ctx?.projectId ?? loadedProjectId ?? null);
  const resolvedAnalysisId = activeAnalysisId ?? analysisId ?? ctx?.analysisId ?? null;
  const { typingUsers, notifyTyping, stopTyping } = useTypingPresence(resolvedAnalysisId);

  // Keep ?aid=<resolvedAnalysisId> in the URL so buttons that read the URL
  // (Refresh, Open Code, Feedback, Versions/Restore) always have an id even
  // when the page was navigated to without one.
  useEffect(() => {
    if (typeof window === "undefined" || !resolvedAnalysisId) return;
    const url = new URL(window.location.href);
    if (url.searchParams.get("aid") === resolvedAnalysisId) return;
    url.searchParams.set("aid", resolvedAnalysisId);
    window.history.replaceState({}, "", url.toString());
  }, [resolvedAnalysisId]);

  const dismissedInsightCardKeys = useMemo(
    () => new Set(dismissedInsightCardsByAnalysis[resolvedAnalysisId ?? ""] ?? []),
    [dismissedInsightCardsByAnalysis, resolvedAnalysisId],
  );

  const dismissInsightCard = useCallback(
    (cardKey: string) => {
      if (!resolvedAnalysisId) return;
      setDismissedInsightCardsByAnalysis((prev) => ({
        ...prev,
        [resolvedAnalysisId]: [...(prev[resolvedAnalysisId] ?? []), cardKey],
      }));
    },
    [resolvedAnalysisId],
  );

  const workspaceHeading = useMemo(() => {
    if (resolvedAnalysisId && resolvedAnalysisId === motherAnalysisId) {
      return analysisGroupName(motherAnalysisName || ctx?.name || "Analysis", datasetChips.length);
    }
    const child = childAnalyses.find((c) => c.id === resolvedAnalysisId);
    if (child) return child.name;
    return ctx?.name ?? "Analysis";
  }, [resolvedAnalysisId, motherAnalysisId, motherAnalysisName, childAnalyses, ctx?.name, datasetChips.length]);

  const tabs = useMemo((): DashboardTab[] => {
    if (!resolvedProjectId || !motherAnalysisId) return [];
    return [
      analysisDashboardTab(motherAnalysisId, motherAnalysisName || "Analysis"),
      ...childAnalyses.map((child) => analysisDashboardTab(child.id, child.name)),
    ];
  }, [resolvedProjectId, motherAnalysisId, motherAnalysisName, childAnalyses]);

  const activeTab = tabs.find((t) => t.id === activeTabId) ?? tabs[0];
  const resolvedTabId = resolvedAnalysisId ? analysisDashboardTabId(resolvedAnalysisId) : "";

  useEffect(() => {
    if (resolvedTabId) setActiveTabId(resolvedTabId);
  }, [resolvedTabId]);

  useEffect(() => {
    resolvedTabIdRef.current = resolvedTabId;
  }, [resolvedTabId]);

  const [myProfileId, setMyProfileId] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    void import("@/lib/current-profile").then(async ({ getCurrentProfileId }) => {
      const pid = await getCurrentProfileId().catch(() => null);
      if (!cancelled) setMyProfileId(pid);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const applyPersistedAnalysis = useCallback(
    async (targetAnalysisId: string, rows: AnalysisMessage[], row: Awaited<ReturnType<typeof persistLoadAnalysis>>) => {
      const hydrateTabId = analysisDashboardTabId(targetAnalysisId);
      const hydrateDefaultSectionId = mainSectionId(hydrateTabId);

      setMessages([]);
      setAnalysisVizConfig(null);
      setAnalysisRows(null);
      setOutputFile(null);
      setUserInsightsViz(null);
      setUserInsightsList([]);
      setUserInsightsRows(null);
      setTableRuns([]);
      setPastedAnalyses([]);
      setSelectedRunId(null);

      if (rows.length) {
        const authorIds = Array.from(
          new Set(rows.filter((r) => r.role === "user" && r.author_id).map((r) => r.author_id as string)),
        );
        let authorMap: Record<string, { name?: string; avatar?: string | null }> = {};
        if (authorIds.length) {
          const { data: profs } = await supabase
            .from("profiles")
            .select("id, full_name, avatar_url")
            .in("id", authorIds);
          authorMap = Object.fromEntries(
            (profs ?? []).map((p: any) => [p.id, { name: p.full_name ?? undefined, avatar: p.avatar_url ?? null }]),
          );
        }
        const chatRows = rows.filter((r) => {
          const o = r.output as { pasted_analysis?: unknown; pasted_analyses?: unknown } | null;
          return !o?.pasted_analysis && !o?.pasted_analyses;
        });
        setMessages(
          chatRows.map((r, idx) => {
            const authorMeta = r.author_id ? authorMap[r.author_id] : undefined;
            const isSelf = r.role === "user" ? (myProfileId ? r.author_id === myProfileId : true) : false;
            const base = {
              id: r.id,
              role: r.role === "user" ? ("user" as const) : ("ai" as const),
              content:
                r.role === "assistant"
                  ? chatDisplayContent(extractTabularContent(r.content).text, r.output)
                  : r.content,
              time: new Date(r.created_at).toLocaleTimeString([], {
                hour: "numeric",
                minute: "2-digit",
              }),
              authorId: r.author_id,
              authorName: authorMeta?.name,
              authorAvatar: authorMeta?.avatar ?? null,
              isSelf,
            };
            if (r.role !== "user") return base;
            const next = chatRows[idx + 1];
            const out = next?.output as {
              viz_config?: unknown;
              output_json?: unknown[];
              section_id?: string;
            } | null;
            if (
              next?.role === "assistant" &&
              (out?.viz_config || (Array.isArray(out?.output_json) && out.output_json.length > 0))
            ) {
              if (Array.isArray(out?.output_json) && out.output_json.length > 0) {
                setUserInsightsRows(out.output_json as Record<string, unknown>[]);
              }
              return {
                ...base,
                sectionId: typeof out?.section_id === "string" ? out.section_id : hydrateDefaultSectionId,
                content: base.content,
              };
            }
            return base;
          }),
        );

      }

      const hydratedRuns = extractTableRunsFromMessages(rows, hydrateDefaultSectionId);
      setTableRuns(hydratedRuns);
      setSelectedRunId(hydratedRuns.length ? hydratedRuns[hydratedRuns.length - 1]!.id : null);
      setPastedAnalyses(extractPastedAnalysesFromMessages(rows, hydrateDefaultSectionId));

      const lastOutput = [...rows].reverse().find((r) => r.role === "assistant" && r.output != null);
      let lastMessageViz: unknown = null;
      if (lastOutput?.output) {
        const out = lastOutput.output as { output_json?: unknown[]; viz_config?: unknown };
        if (Array.isArray(out.output_json) && out.output_json.length > 0) {
          setAnalysisRows(out.output_json as Record<string, unknown>[]);
          setUserInsightsRows(out.output_json as Record<string, unknown>[]);
          setView("table");
        }
        if (out.viz_config) lastMessageViz = out.viz_config;
      }

      if (!row) return;

      setCtx((prev) => ({
        name: row.name,
        project: row.project_id ? (prev?.project ?? "") : "",
        projectId: row.project_id ?? null,
        analysisId: targetAnalysisId,
      }));
      // The persisted row is authoritative. In particular, a standalone
      // analysis must clear any project inherited from the previously opened
      // workspace instead of retaining its project-scoped chrome.
      setLoadedProjectId(row.project_id ?? null);
      if (row.thread_id) setThreadId(row.thread_id);
      if ((row as { session_id?: string | null }).session_id) {
        setSessionId((row as { session_id?: string | null }).session_id ?? null);
      }

      // A cloud batch registers several datasets against one analysis — restore
      // the whole batch so the user can toggle between them inline.
      const storedBatch = readDatasetBatch(targetAnalysisId);
      let restoredChips: DatasetChip[] = [];
      if (
        storedBatch &&
        (!row.dataset_id || storedBatch.chips.some((c) => c.id === row.dataset_id))
      ) {
        restoredChips = storedBatch.chips;
        setDatasetChips(storedBatch.chips);
        const activeId =
          storedBatch.selectedId ?? row.dataset_id ?? storedBatch.chips[0]?.id ?? null;
        setSelectedDatasetIds(activeId ? [activeId] : []);
        const active = storedBatch.chips.find((c) => c.id === activeId);
        if (active) {
          if (active.sessionId) setSessionId(active.sessionId);
          if (active.threadId) setThreadId(active.threadId);
        }
      } else {
        // Multi-file uploads live as a parent row + children linked by
        // parent_analysis_id. Restore the whole group so Selected Dataset
        // matches what a fresh upload showed.
        let groupChips: DatasetChip[] = [];
        try {
          const group = await loadAnalysisGroupDatasets(targetAnalysisId);
          groupChips = group.map((g) => ({ id: g.dataset_id, name: g.name }));
        } catch {
          /* fall back to the single row below */
        }
        if (!groupChips.length && row.dataset_id) {
          groupChips = [{ id: row.dataset_id, name: row.filename || "Dataset" }];
        }
        restoredChips = groupChips;
        setDatasetChips(groupChips);
        setSelectedDatasetIds(
          row.dataset_id && groupChips.some((c) => c.id === row.dataset_id)
            ? [row.dataset_id]
            : groupChips[0]
              ? [groupChips[0].id]
              : [],
        );
      }

      if (row.filename) setAutoInsightsDatasetName(row.filename);
      let resolvedSamples = Array.isArray(row.samples) ? (row.samples as Record<string, unknown>[]) : null;
      if (!resolvedSamples?.length) {
        try {
          const raw =
            sessionStorage.getItem(`analysis:dataset:${targetAnalysisId}`) ??
            sessionStorage.getItem("analysis:dataset");
          if (raw) {
            const parsed = JSON.parse(raw) as { samples?: unknown };
            if (Array.isArray(parsed.samples) && parsed.samples.length) {
              resolvedSamples = parsed.samples as Record<string, unknown>[];
            }
          }
        } catch {}
      }
      if (resolvedSamples?.length) {
        setAutoInsightsSamples(resolvedSamples);
      } else {
        setAutoInsightsSamples(undefined);
      }
      if (resolvedSamples?.length) {
        let schema: string[] = Array.isArray(row.schema) ? (row.schema as string[]) : [];
        if (!schema.length) {
          const first = resolvedSamples[0];
          if (first && typeof first === "object" && !Array.isArray(first)) {
            schema = Object.keys(first as Record<string, unknown>);
          } else if (Array.isArray(first)) {
            schema = (first as unknown[]).map((_, i) => `col_${i + 1}`);
          }
        }
        setPersistedDataset({
          filename: row.filename || "Dataset",
          schema,
          samples: resolvedSamples,
          rows_sampled: resolvedSamples.length,
        });
      } else {
        setPersistedDataset(null);
        // Fallback: fetch dataset preview from backend when the analysis row
        // does not carry samples/schema (older analyses, or samples too large
        // to persist inline).
        if (row.dataset_id) {
          const dsId = row.dataset_id;
          const sess = ((row as { session_id?: string | null }).session_id ?? sessionId ?? "") as string;
          (async () => {
            try {
              const preview: any = await backendApi.getDatasetPreview(dsId, sess);
              const samples: any[] = Array.isArray(preview?.samples)
                ? preview.samples
                : Array.isArray(preview?.rows)
                  ? preview.rows
                  : [];
              if (!samples.length) return;
              let schema: string[] = Array.isArray(preview?.schema)
                ? preview.schema
                : Array.isArray(preview?.columns)
                  ? preview.columns
                  : [];
              if (!schema.length) {
                const first = samples[0];
                if (first && typeof first === "object" && !Array.isArray(first)) {
                  schema = Object.keys(first);
                } else if (Array.isArray(first)) {
                  schema = (first as unknown[]).map((_, i) => `col_${i + 1}`);
                }
              }
              setPersistedDataset({
                filename: row.filename || preview?.filename || "Dataset",
                schema,
                samples,
                rows_sampled: preview?.rows_sampled ?? samples.length,
              });
              setAutoInsightsSamples((cur) => cur ?? samples);
            } catch {
              /* ignore */
            }
          })();
        }
      }

      if (row.viz_config) {
        setAutoInsightsConfig(row.viz_config as any);
        setAutoInsightsStatus("ready");
        setAnalysisVizConfig(row.viz_config);
        setView("chart");
      } else if (lastMessageViz) {
        setAutoInsightsConfig(null);
        setAutoInsightsStatus(undefined);
        setAnalysisVizConfig(lastMessageViz);
      } else {
        setAutoInsightsConfig(null);
        setAutoInsightsStatus(undefined);
      }

      const defaultSectionId = mainSectionId(hydrateTabId);
      const userInsights = extractUserInsightsFromMessages(rows, row.viz_config ?? null, defaultSectionId);
      setUserInsightsList(userInsights);
      setExtraSectionsByTab((prev) => {
        const restoredExtras = extraSectionsFromUserInsights(hydrateTabId, userInsights);
        return restoredExtras.length ? { ...prev, [hydrateTabId]: restoredExtras } : prev;
      });
      if (userInsights.length) {
        setUserInsightsViz(userInsights[userInsights.length - 1]!.vizConfig);
      } else if (lastMessageViz) {
        setUserInsightsViz(lastMessageViz);
      }

      activeSectionIdRef.current = defaultSectionId;
      setActiveSectionId(defaultSectionId);

      try {
        const sessionPayload = {
          threadId: row.thread_id,
          sessionId: (row as { session_id?: string | null }).session_id ?? null,
          datasetChips: restoredChips.length
            ? restoredChips
            : row.dataset_id
              ? [{ id: row.dataset_id, name: row.filename || "Dataset" }]
              : [],
        };
        const datasetPayload = {
          filename: row.filename,
          schema: row.schema,
          samples: row.samples,
          visualization_config: row.viz_config,
          visualization_status: row.viz_config ? "ready" : undefined,
        };
        sessionStorage.setItem("analysis:session", JSON.stringify(sessionPayload));
        sessionStorage.setItem("analysis:dataset", JSON.stringify(datasetPayload));
        sessionStorage.setItem(`analysis:session:${targetAnalysisId}`, JSON.stringify(sessionPayload));
        sessionStorage.setItem(`analysis:dataset:${targetAnalysisId}`, JSON.stringify(datasetPayload));
      } catch {}
    },
    [myProfileId],
  );

  const switchAnalysisTab = useCallback(
    async (targetAnalysisId: string) => {
      if (!targetAnalysisId || targetAnalysisId === activeAnalysisId) return;
      setActiveAnalysisId(targetAnalysisId);
      setActiveTabId(analysisDashboardTabId(targetAnalysisId));
      activeSectionIdRef.current = mainSectionId(analysisDashboardTabId(targetAnalysisId));
      setActiveSectionId(activeSectionIdRef.current);

      if (motherAnalysisId) {
        navigate({
          to: "/analysis",
          search: {
            aid: motherAnalysisId,
            ...(targetAnalysisId !== motherAnalysisId ? { child: targetAnalysisId } : {}),
          },
          replace: true,
        });
      }

      const token = ++tabHydrateTokenRef.current;
      try {
        const [rows, row] = await Promise.all([
          persistLoadMessages(targetAnalysisId),
          persistLoadAnalysis(targetAnalysisId),
        ]);
        if (token !== tabHydrateTokenRef.current) return;
        await applyPersistedAnalysis(targetAnalysisId, rows, row);
      } catch (err) {
        console.warn("Failed to load analysis tab", err);
      }
    },
    [activeAnalysisId, applyPersistedAnalysis, motherAnalysisId, navigate],
  );

  const openChildUpload = useCallback(() => {
    if (!motherAnalysisId || !resolvedProjectId) {
      toast.error("Save this analysis to a project before adding another dataset.");
      return;
    }
    setChildPickerOpen(true);
  }, [motherAnalysisId, resolvedProjectId]);


  const addWorkspaceSection = useCallback(() => {
    const tabId = resolvedTabIdRef.current || resolvedTabId;
    if (!tabId) return;
    const id = `${tabId}-extra-${Date.now()}`;
    setExtraSectionsByTab((prev) => ({
      ...prev,
      [tabId]: [...(prev[tabId] ?? []), { id }],
    }));
    activeSectionIdRef.current = id;
    setActiveSectionId(id);
    window.setTimeout(() => {
      sectionRefs.current[id]?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 50);
  }, [resolvedTabId]);

  const refreshThreadInsightSources = useCallback(async (motherId: string, children: ChildAnalysisSummary[]) => {
    const ids = [motherId, ...children.map((c) => c.id)];
    const sources: InsightSource[] = [];
    for (const id of ids) {
      const [row, rows] = await Promise.all([persistLoadAnalysis(id), persistLoadMessages(id)]);
      if (!row) continue;
      const tabId = analysisDashboardTabId(id);
      const defaultSectionId = mainSectionId(tabId);
      const userInsights = extractUserInsightsFromMessages(rows, row.viz_config ?? null, defaultSectionId);
      sources.push({
        analysisId: id,
        analysisName: row.name,
        vizConfig: row.viz_config,
        userVizConfigs: userInsights.map((entry) => entry.vizConfig),
        samples: Array.isArray(row.samples) ? (row.samples as unknown[]) : undefined,
      });
    }
    setThreadInsightSources(sources);
  }, []);

  const refreshOneThreadInsightSource = useCallback(async (analysisId: string) => {
    const [row, rows] = await Promise.all([persistLoadAnalysis(analysisId), persistLoadMessages(analysisId)]);
    if (!row) return;
    const tabId = analysisDashboardTabId(analysisId);
    const defaultSectionId = mainSectionId(tabId);
    const userInsights = extractUserInsightsFromMessages(rows, row.viz_config ?? null, defaultSectionId);
    const updated: InsightSource = {
      analysisId,
      analysisName: row.name,
      vizConfig: row.viz_config,
      userVizConfigs: userInsights.map((entry) => entry.vizConfig),
      samples: Array.isArray(row.samples) ? (row.samples as unknown[]) : undefined,
    };
    setThreadInsightSources((prev) => {
      const idx = prev.findIndex((s) => s.analysisId === analysisId);
      if (idx < 0) return [...prev, updated];
      const next = [...prev];
      next[idx] = updated;
      return next;
    });
  }, []);

  const workspaceSections = useMemo((): WorkspaceSection[] => {
    const tabId = resolvedTabId || activeTabId;
    if (!tabId) return [];
    return [{ id: mainSectionId(tabId), title: workspaceHeading }, ...(extraSectionsByTab[tabId] ?? [])];
  }, [resolvedTabId, activeTabId, workspaceHeading, extraSectionsByTab]);

  const currentMainSectionId = mainSectionId(resolvedTabId || activeTabId);

  const userInsightEntries = useMemo(() => {
    const autoFp = autoInsightsConfig ? vizFingerprint(autoInsightsConfig) : null;
    // Dedupe: never render a user-insight entry whose viz is the same as
    // the Auto Insights chart (would show the same chart twice), and drop
    // duplicates within the user list itself.
    const dedupe = (list: UserInsightEntry[]): UserInsightEntry[] => {
      const seen = new Set<string>();
      const out: UserInsightEntry[] = [];
      for (const entry of list) {
        const fp = vizFingerprint(entry.vizConfig);
        if (autoFp && fp === autoFp) continue;
        if (seen.has(fp)) continue;
        seen.add(fp);
        out.push(entry);
      }
      return out;
    };
    if (userInsightsList.length) return dedupe(userInsightsList);
    if (!userInsightsViz) return [];
    const sectionId = activeSectionId || currentMainSectionId;
    if (autoFp && vizFingerprint(userInsightsViz) === autoFp) return [];
    return [{ id: "latest-user-insight", sectionId, vizConfig: userInsightsViz }];
  }, [userInsightsList, userInsightsViz, autoInsightsConfig, activeSectionId, currentMainSectionId]);

  const autoInsightsViz = autoInsightsConfig;
  const legacyAutoViz = !autoInsightsConfig && !userInsightEntries.length ? analysisVizConfig : null;
  const resolvedAutoViz = autoInsightsViz ?? legacyAutoViz;

  // Same combined builder the Auto Insights modal uses: one group per distinct
  // dataset_id, each reading its OWN visualization_config + rows.
  const autoInsightGroups = useMemo(
    () =>
      buildInsightGroups({
        datasets: autoInsightsDatasets,
        config: resolvedAutoViz,
        status: autoInsightsStatus,
        samples: chartSamples,
        datasetName: autoInsightsDatasetName ?? undefined,
        primaryDatasetId: selectedDatasetIds[selectedDatasetIds.length - 1] ?? datasetChips[0]?.id ?? null,
      }),
    [
      autoInsightsDatasets,
      resolvedAutoViz,
      autoInsightsStatus,
      chartSamples,
      autoInsightsDatasetName,
      selectedDatasetIds,
      datasetChips,
    ],
  );

  const multiDatasetInsights = autoInsightGroups.length > 1;

  const autoChartSlides = useMemo(() => {
    if (multiDatasetInsights) {
      return dedupeSlidesByInsight(autoInsightGroups.flatMap((g) => g.slides) as any);
    }
    const raw = slidesForDashboardTab(resolvedAutoViz, chartSamples, undefined);
    return dedupeSlidesByInsight(raw);
  }, [multiDatasetInsights, autoInsightGroups, resolvedAutoViz, chartSamples]);

  const autoChartTitles = useMemo(
    () => new Set(autoChartSlides.map((s) => normalizeInsightTitle(s.title)).filter(Boolean)),
    [autoChartSlides],
  );


  const mainSectionUserSlides = useMemo(() => {
    const raw = userChartSlidesForSection(userInsightEntries, currentMainSectionId, undefined, chartSamples);
    return dedupeSlidesByInsight(raw, autoChartTitles);
  }, [userInsightEntries, currentMainSectionId, chartSamples, autoChartTitles]);

  // Selected analysis run drives both the Chart and Data table views.
  const selectedRunIndex = (() => {
    const i = tableRuns.findIndex((r) => r.id === selectedRunId);
    return i >= 0 ? i : tableRuns.length - 1;
  })();
  const selectedRun = selectedRunIndex >= 0 ? tableRuns[selectedRunIndex] : undefined;

  // Group everything pasted in one action into a single block: one carousel
  // holding every copied chart + a pager over the copied result tables.
  const pastedPreviews = useMemo(() => {
    const groups = new Map<string, PastedAnalysis[]>();
    for (const run of pastedAnalyses) {
      const gid = run.groupId || run.id;
      const list = groups.get(gid);
      if (list) list.push(run);
      else groups.set(gid, [run]);
    }
    return Array.from(groups.entries()).map(([groupId, runs]) => {
      // Prefer the exact slides copied from the source tab so the paste target
      // renders the same graphs; fall back to per-run derived charts.
      const copied = runs.flatMap((r) => (Array.isArray(r.slides) ? (r.slides as Slide[]) : []));
      const derived = runs.flatMap((r) => {
        const viz = r.vizConfig ?? deriveVizFromRows(r.rows);
        return viz ? (slidesForDashboardTab(viz, r.rows, undefined) as Slide[]) : [];
      });
      const slides = dedupeSlidesByInsight([...copied, ...derived] as any) as Slide[];
      const base = collectSlidesInsights(slides);
      const extras = runs.flatMap((r) => r.extraInsights ?? []);
      const extra = extras.filter(
        (i, idx) =>
          i && !base.insights.includes(i) && !base.summaries.includes(i) && extras.indexOf(i) === idx,
      );
      return { groupId, runs, slides, insights: { ...base, insights: [...base.insights, ...extra] } };
    });
  }, [pastedAnalyses]);

  // Keep the "Analysis code" panel in sync with the result pager: paging the
  // table/charts swaps the code panel to that same result's script.
  useEffect(() => {
    if (!codeOpen) return;
    if (selectedRunId === codeRunId) return;
    void loadCodeForRun(selectedRunId, { silent: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [codeOpen, selectedRunId, codeRunId]);

  const goToRun = useCallback(
    (dir: -1 | 1) => {
      if (tableRuns.length < 2) return;
      const from = selectedRunIndex >= 0 ? selectedRunIndex : tableRuns.length - 1;
      const next = tableRuns[(from + dir + tableRuns.length) % tableRuns.length];
      if (next) setSelectedRunId(next.id);
    },
    [tableRuns, selectedRunIndex],
  );

  /** Copy only the current result's analysis data (table rows + code +
   *  fidelity) to the clipboard — never chat messages, never viz config. */
  /** When clipboard APIs are blocked (iframe permission policy etc.), show
   *  the serialized clip in a modal so the user can copy it manually. */
  const [copyFallbackText, setCopyFallbackText] = useState<string | null>(null);

  const copyAnalysisData = useCallback(async () => {
    if (!selectedRun) {
      toast.error("No analysis result to copy");
      return;
    }
    // Copy EVERY result of the current tab (not just the visible page), each
    // with its own chart config, plus the tab's key insights.
    const tabId = resolvedTabId || activeTabId;
    const inTab = (sid: string) =>
      sid === currentMainSectionId || (!!tabId && sid.startsWith(`${tabId}-`));
    const runsInTab = tableRuns.filter((r) => inTab(r.sectionId));
    const sourceRuns = runsInTab.length ? runsInTab : [selectedRun];

    const toClipRun = (r: TableRun): AnalysisClipRun => ({
      id: r.id,
      title: r.title,
      rows: r.rows,
      code: r.code ?? null,
      promptTs: r.promptTs ?? null,
      file: r.file ?? null,
      vizConfig:
        userInsightEntries.find((e) => e.id === r.id)?.vizConfig ?? deriveVizFromRows(r.rows) ?? null,
    });

    // Capture EVERY chart currently rendered for this tab (auto insights,
    // user insights and each result's own charts) so paste reproduces them all.
    const runSlides = sourceRuns.flatMap((r) => {
      const viz = userInsightEntries.find((e) => e.id === r.id)?.vizConfig ?? deriveVizFromRows(r.rows);
      return viz ? (slidesForDashboardTab(viz, r.rows, undefined) as Slide[]) : [];
    });
    const allSlides = dedupeSlidesByInsight([
      ...autoChartSlides,
      ...mainSectionUserSlides,
      ...runSlides,
    ] as any) as Slide[];

    const tabInsights = collectSlidesInsights(allSlides);

    let serializedSlides: unknown[] = [];
    try {
      serializedSlides = JSON.parse(JSON.stringify(allSlides));
    } catch {
      serializedSlides = [];
    }

    const clip: AnalysisClip = {
      marker: AVALOKA_CLIP_MARKER,
      v: 1,
      run: toClipRun(selectedRun),
      runs: sourceRuns.map(toClipRun),
      insights: [...tabInsights.summaries, ...tabInsights.insights],
      slides: serializedSlides,
      fidelity: analysisFidelity,
    };
    const text = JSON.stringify(clip);


    // Try the async clipboard API first.
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        toast.success("Analysis data copied — paste into another analysis");
        return;
      } catch {
        // Fall through to the execCommand path below.
      }
    }
    // Fallback: hidden textarea + execCommand (works where writeText is denied).
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.top = "0";
      ta.style.left = "0";
      ta.style.opacity = "0";
      ta.setAttribute("readonly", "");
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      ta.setSelectionRange(0, ta.value.length);
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      if (!ok) throw new Error("execCommand copy failed");
      toast.success("Analysis data copied — paste into another analysis");
      return;
    } catch {
      // Last resort: show the text so the user can copy it manually.
      setCopyFallbackText(text);
      toast.info("Clipboard is blocked here — copy the text from the dialog");
    }
  }, [
    selectedRun,
    analysisFidelity,
    tableRuns,
    resolvedTabId,
    activeTabId,
    currentMainSectionId,
    userInsightEntries,
    autoChartSlides,
    mainSectionUserSlides,
  ]);

  /** Paste modal state — user pastes the copied blob into a text area, then
   *  we parse it and render it here as a new result page. */
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");

  const applyPastedAnalysis = useCallback(() => {
    const clip = isAnalysisClip(pasteText);
    if (!clip) {
      toast.error("That doesn't look like copied analysis data");
      return;
    }
    const stamp = Date.now();
    const sectionId = currentMainSectionId || "main";
    const clipRuns = (clip.runs?.length ? clip.runs : [clip.run]).filter(
      (r) => r && Array.isArray(r.rows) && r.rows.length > 0,
    );
    if (!clipRuns.length) {
      toast.error("Copied analysis has no data");
      return;
    }
    const tabInsights = Array.isArray(clip.insights) ? clip.insights.filter(Boolean) : [];

    const groupId = `pasted-${stamp}`;
    const clipSlides = Array.isArray(clip.slides) ? clip.slides : [];

    const pastedRuns: PastedAnalysis[] = clipRuns.map((r, i) => ({
      id: `pasted-${stamp}-${i}`,
      groupId,
      sectionId,
      title: r.title ? `${r.title} (pasted)` : `Pasted result ${i + 1}`,
      rows: r.rows,
      code: r.code ?? null,
      promptTs: r.promptTs ?? null,
      file: r.file ?? null,
      vizConfig: r.vizConfig ?? null,
      // Charts + key insights of the source tab ride on the first run of the
      // group; the whole group renders as one carousel block.
      slides: i === 0 ? clipSlides : undefined,
      extraInsights: i === 0 ? tabInsights : undefined,
    }));


    setPastedAnalyses((prev) => [...prev, ...pastedRuns]);
    setPasteOpen(false);
    setPasteText("");
    toast.success(
      pastedRuns.length > 1
        ? `${pastedRuns.length} pasted results added below the current analysis`
        : "Pasted result added below the current analysis",
    );

    // Persist so the pasted results survive a reload — stored on the analysis
    // message stream as a `pasted_analyses` payload (not a chat bubble), and
    // merge the pasted rows into the analysis' `samples` column so the full
    // dataset (original + pasted) travels with the analysis row.
    const aid = resolvedAnalysisId;
    if (aid) {
      void persistMessage({
        analysis_id: aid,
        role: "assistant",
        content: "",
        output: { pasted_analyses: pastedRuns, section_id: sectionId },
      }).catch((err) => console.warn("Failed to save pasted analysis", err));

      void (async () => {
        try {
          const row = await persistLoadAnalysis(aid);
          const existing = Array.isArray(row?.samples)
            ? (row.samples as Record<string, unknown>[])
            : [];
          const merged = [...existing, ...pastedRuns.flatMap((r) => r.rows)];
          await persistUpdateAnalysisContext(aid, { samples: merged });
        } catch (err) {
          console.warn("Failed to merge pasted rows into samples", err);
        }
      })();
    }
  }, [pasteText, currentMainSectionId, resolvedAnalysisId]);

  /** Result action toolbar — rendered under the data table AND under the
   *  charts of the currently viewed result. Scoped to that result only. */
  const renderResultToolbar = () => (
    <TooltipProvider delayDuration={100}>
      <div className="mt-4 flex items-center gap-4 text-fg-quaternary">
        {[
          ...(resolvedProjectId ? [{ label: "Add to dashboard", Icon: ClipboardPlus }] : []),
          { label: "Copy", Icon: Copy01 },
          { label: "Paste result", Icon: ClipboardPlus },
          ...(resolvedProjectId ? [{ label: "Add to Comment", Icon: MessageCircle02 }] : []),
          { label: "Refresh", Icon: Repeat02 },
          { label: "Open code", Icon: CodeSnippet02 },
          { label: "Good Response", Icon: ThumbsUp },
          { label: "Bad Response", Icon: ThumbsDown },
          { label: "Add New Section", Icon: PlusSquareIcon },
        ].map(({ label, Icon }) => {
          const isComment = label === "Add to Comment";
          const isNewSection = label === "Add New Section";
          const isThumbUp = label === "Good Response";
          const isThumbDown = label === "Bad Response";
          const isRefresh = label === "Refresh";
          const isCopy = label === "Copy";
          const isPaste = label === "Paste result";
          const isActiveThumb =
            (isThumbUp && insightFeedback === "positive") ||
            (isThumbDown && insightFeedback === "negative");
          const btn = (
            <button
              aria-label={label}
              aria-pressed={isThumbUp || isThumbDown ? isActiveThumb : undefined}
              disabled={
                ((isThumbUp || isThumbDown) && feedbackSubmitting !== null) ||
                (isRefresh && refreshing)
              }
              onClick={
                label === "Add to dashboard"
                  ? () => openAddToDashboard(0)
                  : isComment
                    ? () => setInsightCommentOpen((v) => !v)
                    : isNewSection
                      ? () => addWorkspaceSection()
                      : label === "Open code"
                        ? () => {
                            void handleOpenCode();
                          }
                        : isRefresh
                          ? () => {
                              void handleRefresh();
                            }
                          : isThumbUp
                            ? () => {
                                void handleThumb("positive");
                              }
                            : isThumbDown
                              ? () => {
                                  void handleThumb("negative");
                                }
                              : isCopy
                                ? () => {
                                    void copyAnalysisData();
                                  }
                                : isPaste
                                  ? () => setPasteOpen(true)
                                  : undefined
              }
              className={cx(
                "cursor-pointer hover:text-fg-secondary disabled:opacity-50",
                isComment && insightCommentOpen && "text-[#1565ef]",
                isActiveThumb && (isThumbUp ? "text-[#1565ef]" : "text-[#d92d20]"),
              )}
            >
              <Icon className={cx("size-4", isRefresh && refreshing && "animate-spin")} />
            </button>
          );
          return (
            <Tooltip key={label}>
              <TooltipTrigger asChild>{btn}</TooltipTrigger>
              <TooltipContent side="top" className="bg-[#0a0d12] text-white">
                {label}
              </TooltipContent>
            </Tooltip>
          );
        })}
        <Popover
          open={versionsOpen}
          onOpenChange={(o) => {
            setVersionsOpen(o);
            if (o) void fetchVersions();
          }}
        >
          <Tooltip>
            <TooltipTrigger asChild>
              <PopoverTrigger asChild>
                <button
                  aria-label="Version history"
                  className="cursor-pointer hover:text-fg-secondary disabled:opacity-50"
                >
                  <ClockRewind className="size-4" />
                </button>
              </PopoverTrigger>
            </TooltipTrigger>
            <TooltipContent side="top" className="bg-[#0a0d12] text-white">
              Version history
            </TooltipContent>
          </Tooltip>
          <PopoverContent
            align="start"
            side="bottom"
            sideOffset={8}
            collisionPadding={12}
            className="z-50 w-[340px] overflow-hidden rounded-xl border border-secondary bg-primary p-0 shadow-xl"
          >
            <div className="border-b border-secondary bg-primary px-4 py-3">
              <p className="text-sm font-semibold text-primary">Version history</p>
              <p className="text-xs text-tertiary">Restore an earlier checkpoint</p>
            </div>

            <div className="max-h-[360px] overflow-y-auto">
              {versionsLoading ? (
                <p className="px-4 py-6 text-center text-xs text-tertiary">Loading…</p>
              ) : versions.length === 0 ? (
                <p className="px-4 py-6 text-center text-xs text-tertiary">No versions yet.</p>
              ) : (
                <ul className="divide-y divide-secondary">
                  {versions.map((v) => {
                    const isRestoring = restoringTs === v.prompt_ts;
                    return (
                      <li key={v.prompt_ts} className="flex items-start gap-3 px-4 py-3">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <p className="truncate text-sm font-medium text-primary">
                              {formatPromptTs(v.prompt_ts)}
                            </p>
                            {v.is_current && (
                              <span className="rounded-full bg-[#eff5ff] px-1.5 py-0.5 text-[10px] font-medium text-[#175cd3]">
                                Current
                              </span>
                            )}
                          </div>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {v.has_code && (
                              <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary">
                                Code
                              </span>
                            )}
                            {v.has_output && (
                              <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary">
                                Output
                              </span>
                            )}
                            {v.has_viz && (
                              <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary">
                                Chart
                              </span>
                            )}
                          </div>
                        </div>
                        {!v.is_current && (
                          <button
                            type="button"
                            disabled={restoringTs !== null}
                            onClick={() => void handleRestore(v.prompt_ts)}
                            className="shrink-0 rounded-md border border-secondary px-2 py-1 text-xs font-medium text-primary hover:bg-secondary disabled:opacity-50"
                          >
                            {isRestoring ? "Restoring…" : "Restore"}
                          </button>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </TooltipProvider>
  );

  const downloadSelectedRunCsv = useCallback(() => {
    if (!selectedRun?.rows?.length) return;
    const blob = new Blob([tableRunToCsv(selectedRun.rows)], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-result-${selectedRunIndex + 1}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [selectedRun, selectedRunIndex]);


  const tabVizInsights = useMemo(() => collectSlidesInsights(autoChartSlides), [autoChartSlides]);
  const hasTabInsights = tabVizInsights.summaries.length > 0 || tabVizInsights.insights.length > 0;

  const resolveThreadVizConfig = useCallback(
    (targetAnalysisId: string) => {
      const cached = threadInsightSources.find((s) => s.analysisId === targetAnalysisId)?.vizConfig;
      if (cached) return cached;
      if (targetAnalysisId === resolvedAnalysisId) return autoInsightsConfig ?? analysisVizConfig;
      return null;
    },
    [threadInsightSources, resolvedAnalysisId, autoInsightsConfig, analysisVizConfig],
  );

  useEffect(() => {
    if (!insightsOpen || !motherAnalysisId || threadInsightSources.length > 0) return;
    void refreshThreadInsightSources(motherAnalysisId, childAnalyses);
  }, [insightsOpen, motherAnalysisId, childAnalyses, threadInsightSources.length, refreshThreadInsightSources]);

  const historyLiveChat = useMemo(
    () => messages.map((m) => ({ id: m.id, role: m.role, content: m.content })),
    [messages],
  );

  const dashboardInsight = useMemo((): DashboardInsightPayload | null => {
    if (resolvedAutoViz) {
      const payload = insightPayloadFromVizConfig(resolvedAutoViz, 0);
      if (payload) return payload;
      return {
        title: workspaceHeading,
        chartType: "bar",
        config: resolvedAutoViz as Record<string, unknown>,
        graphKey: "chart:0",
      };
    }
    if (hasBackendInsights) {
      return {
        title: workspaceHeading,
        chartType: "bar",
        config: { insights: vizInsights },
        graphKey: "insights:0",
      };
    }
    return null;
  }, [resolvedAutoViz, workspaceHeading, hasBackendInsights, vizInsights]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setIsAdmin(localStorage.getItem("role") === "admin");
    try {
      const raw = sessionStorage.getItem("analysis:context");
      if (raw) setCtx(JSON.parse(raw));
    } catch {}
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, chatOpen]);

  // Hydrate workspace from Supabase whenever ?aid= (mother) changes.
  useEffect(() => {
    if (!aidFromUrl) {
      setHydrating(false);
      setMotherAnalysisId(null);
      setMotherAnalysisName("");
      setChildAnalyses([]);
      setActiveAnalysisId(null);
      return;
    }

    let cancelled = false;
    setHydrating(true);
    setAnalysisId(aidFromUrl);
    consumedUploadRef.current = `aid:${aidFromUrl}`;

    (async () => {
      try {
        const motherId = await resolveMotherAnalysisId(aidFromUrl);
        if (cancelled) return;

        const [motherRow, children] = await Promise.all([persistLoadAnalysis(motherId), loadChildAnalyses(motherId)]);
        if (cancelled) return;

        setMotherAnalysisId(motherId);
        setMotherAnalysisName(motherRow?.name ?? "Analysis");
        setChildAnalyses(children);
        void refreshThreadInsightSources(motherId, children);

        const initialActiveId =
          childIdFromUrl && (childIdFromUrl === motherId || children.some((c) => c.id === childIdFromUrl))
            ? childIdFromUrl
            : motherId;
        setActiveAnalysisId(initialActiveId);

        const [rows, row] = await Promise.all([
          persistLoadMessages(initialActiveId),
          persistLoadAnalysis(initialActiveId),
        ]);
        if (cancelled) return;
        await applyPersistedAnalysis(initialActiveId, rows, row);
      } catch (err) {
        console.warn("Failed to load saved analysis", err);
      } finally {
        if (!cancelled) setHydrating(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [aidFromUrl, applyPersistedAnalysis, refreshThreadInsightSources]);

  // Apply fresh upload data when a child analysis tab is created via +.
  useEffect(() => {
    if (!childIdFromUrl || !motherAnalysisId) return;
    const up = navState?.uploadResponse;
    if (!up) return;
    const childId = (navState?.childAnalysisId as string | undefined) ?? childIdFromUrl;
    if (childId !== childIdFromUrl) return;

    const key = up.dataset_id || up.thread_id || JSON.stringify(up).slice(0, 64);
    if (consumedUploadRef.current === `child:${childId}:${key}`) return;
    consumedUploadRef.current = `child:${childId}:${key}`;

    const cfg = up.visualization_config ?? up.visualization_configs ?? null;
    const status = up.visualization_status as string | undefined;
    setActiveAnalysisId(childId);
    setActiveTabId(analysisDashboardTabId(childId));
    setAutoInsightsConfig(cfg);
    setAutoInsightsStatus(status);
    setAutoInsightsDatasetName(navState?.filename);
    setAutoInsightsSamples(Array.isArray(up.samples) ? up.samples : undefined);
    if (cfg && (!status || status === "ready")) {
      setAnalysisVizConfig(cfg);
      setView("chart");
    }

    if (up.thread_id) setThreadId(up.thread_id);
    if (up.session_id) setSessionId(up.session_id);
    const chips: { id: string; name: string }[] = [];
    if (Array.isArray(up.datasets)) {
      up.datasets.forEach((d: any) => chips.push({ id: d.dataset_id, name: d.alias || d.filename || d.dataset_id }));
    } else if (up.dataset_id) {
      chips.push({ id: up.dataset_id, name: navState?.filename || "Dataset" });
    }
    if (chips.length) setDatasetChips(chips);

    setChildAnalyses((prev) => {
      const name = String(navState?.filename ?? "Analysis").replace(/\.[^.]+$/, "");
      if (prev.some((c) => c.id === childId)) return prev;
      return [...prev, { id: childId, name, created_at: new Date().toISOString() }];
    });
    if (motherAnalysisId) {
      void loadChildAnalyses(motherAnalysisId).then((children) => {
        setChildAnalyses(children);
        void refreshThreadInsightSources(motherAnalysisId, children);
      });
    }

    setMessages([]);
    setUserInsightsList([]);
    setUserInsightsViz(null);
  }, [childIdFromUrl, motherAnalysisId, navState]);

  // Ensure an analysis_id exists; create one on demand so the first user
  // prompt can be saved. Returns null if we can't (e.g. user not signed in
  // or no project available) — caller should treat persistence as best-effort.
  const ensureAnalysisId = async (): Promise<string | null> => {
    if (analysisId) return analysisId;
    try {
      const up = navState?.uploadResponse as any | undefined;
      const a = await persistCreateAnalysis({
        project_id: null,
        name: ctx?.name || navState?.filename || "New Analysis",
        dataset_id: up?.dataset_id ?? datasetChips[0]?.id ?? null,
        thread_id: threadId,
        session_id: sessionId,
        filename: navState?.filename ?? autoInsightsDatasetName ?? null,
        schema: up?.schema ?? null,
        samples: up?.samples ?? autoInsightsSamples ?? null,
        viz_config: analysisVizConfig ?? autoInsightsConfig ?? null,
      });

      setAnalysisId(a.id);
      if (typeof window !== "undefined") {
        const url = new URL(window.location.href);
        url.searchParams.set("aid", a.id);
        window.history.replaceState({}, "", url.toString());
      }
      qc.invalidateQueries({ queryKey: analysesKey(null) });
      return a.id;
    } catch (err) {
      console.warn("Failed to create analysis row", err);
      return null;
    }
  };

  const openAddToDashboard = (chartIndex = 0) => {
    void (async () => {
      const id = resolvedAnalysisId ?? (await ensureAnalysisId());
      if (!id) {
        toast.error("Could not save analysis. Open this analysis from a project first.");
        return;
      }
      setDashboardChartIndex(chartIndex);
      setAddToDashboardOpen(true);
    })();
  };

  /**
   * Render a training result that finished while this view was unmounted
   * (the user navigated away mid-run). Minimal: assistant text + table + chart.
   */
  const applyBackgroundTrainingResult = (res: any) => {
    if (!res || typeof res !== "object") return;
    updatePlannerGraphAvailability(res);
    const sectionId = activeSectionIdRef.current || mainSectionId(resolvedTabIdRef.current || activeTabId);
    const nowTime = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const allMsgs = (res.messages ?? []) as any[];
    const lastAssistant = [...allMsgs]
      .reverse()
      .find((m) => m && m.role === "assistant" && typeof m.content === "string" && m.content.trim());
    const extracted = lastAssistant
      ? extractTabularContent(lastAssistant.content as string)
      : { text: "", tables: [] as ParsedTable[] };

    setMessages((curr) => [
      ...curr.filter((m) => !m.thinking),
      {
        id: `bg-${Date.now()}`,
        role: "ai",
        content: extracted.text || ANALYSIS_COMPLETE_MESSAGE,
        time: nowTime,
        sectionId,
      },
    ]);

    const runId = `bg-run-${Date.now()}`;
    const rows: Record<string, unknown>[] | null = Array.isArray(res.output_json) && res.output_json.length
      ? (res.output_json as Record<string, unknown>[])
      : extracted.tables[0]?.rows ?? null;
    if (rows && rows.length) {
      setAnalysisRows(rows);
      setUserInsightsRows(rows);
      pushTableRun({
        id: runId,
        sectionId,
        title: "Training result",
        rows,
        code: typeof res?.coder_definition?.code === "string" ? res.coder_definition.code : null,
        promptTs: null,
        file: null,
      });
      setView("table");
      const viz = deriveVizFromRows(rows) ?? (res.visualization_config ?? null);
      if (viz) {
        setUserInsightsViz(viz);
        setUserInsightsList((prev) =>
          prev.some((e) => e.sectionId === sectionId && vizFingerprint(e.vizConfig) === vizFingerprint(viz))
            ? prev
            : [...prev, { id: runId, sectionId, vizConfig: viz }],
        );
      }
    }
    setIsSending(false);
  };

  // Background training results: claim anything that landed for this thread
  // while we were unmounted, and listen for completions from the global store.
  useEffect(() => {
    if (!threadId) return;
    const drain = () => {
      const claimed = claimTrainingResult(threadId);
      if (claimed?.result) applyBackgroundTrainingResult(claimed.result);
    };
    drain();
    const onEvent = (e: Event) => {
      const detail = (e as CustomEvent).detail as { threadId?: string } | undefined;
      if (detail?.threadId && detail.threadId !== threadId) return;
      drain();
    };
    window.addEventListener(TRAINING_RESULT_EVENT, onEvent);
    return () => window.removeEventListener(TRAINING_RESULT_EVENT, onEvent);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadId]);

  const send = async (content: string): Promise<boolean> => {
    content = content.trim();
    if (!content || isSending) return false;

    const tabId = resolvedTabIdRef.current || resolvedTabId || activeTabId;
    const targetSectionId = activeSectionIdRef.current || mainSectionId(tabId);


    const nowTime = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const userId = `u${Date.now()}`;
    const placeholderId = `p${Date.now()}`;
    const carriedReply = replyContext;

    setMessages((m) => [
      ...m,
      {
        id: userId,
        role: "user",
        content,
        time: nowTime,
        replyTo: carriedReply ?? undefined,
        sectionId: targetSectionId || undefined,
        authorId: myProfileId,
        isSelf: true,
      },
      { id: placeholderId, role: "ai", content: "Analyzing...", time: nowTime, thinking: true },
    ]);
    setReplyContext(null);

    if (!threadId || !sessionId) {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === placeholderId
            ? {
                ...msg,
                thinking: false,
                error: true,
                content: "⚠️ No active analysis session. Please upload a dataset first to start a chat.",
              }
            : msg,
        ),
      );
      return true;
    }

    // Persist the user prompt (best-effort, won't block the chat flow).
    const aid = resolvedAnalysisId ?? (await ensureAnalysisId());
    if (aid) {
      persistMessage({ analysis_id: aid, role: "user", content, author_id: myProfileId }).catch((err) =>
        console.warn("Failed to save user message", err),
      );
    }

    setIsSending(true);
    try {
      // Dataset routing: always carry the full group of dataset_ids so a
      // multi-file group keeps every file in scope on every turn. A single
      // explicit selection narrows the request to that one dataset.
      const groupIds = Array.from(new Set(datasetChips.map((c) => c.id).filter(Boolean)));
      const validSelected = Array.from(
        new Set(selectedDatasetIds.filter((id) => id && groupIds.includes(id))),
      );
      const datasetIdsToSend =
        validSelected.length === 1
          ? validSelected
          : groupIds.length > 0
            ? groupIds
            : undefined;
      const pinnedId =
        validSelected.length === 1
          ? validSelected[0]
          : groupIds.length === 1
            ? groupIds[0]
            : null;
      let res = await backendApi.sendMessage(
        threadId,
        sessionId,
        content,
        datasetIdsToSend,
        analysisFidelity,
        selectedSampleName,
        { dataset_id: pinnedId },
      );
      // eslint-disable-next-line no-console
      console.log("[analysis.sendMessage] raw response", res);

      // LONG-RUNNING TURN: the backend either accepted a training job and is
      // still running it, or the proxy stopped waiting for a slow response.
      // Keep the user's message, show a progress bubble and poll for the
      // real result instead of failing or faking a "Done." reply.
      if (isDeferredTurn(res)) {
        const deferredId = (res as any).analysis_task_id ?? null;
        setMessages((curr) =>
          curr.map((m) =>
            m.id === placeholderId
              ? {
                  ...m,
                  thinking: true,
                  error: false,
                  content:
                    "Training is running — this can take a few minutes. Results will appear here automatically.",
                }
              : m,
          ),
        );

        // Polling lives in the global training-jobs store so it survives
        // navigation between threads/views. Exactly one poll per thread.
        const pending = await startTrainingJob({
          threadId,
          deferredId,
          sessionId,
          analysisId: aid ?? resolvedAnalysisId ?? null,
          label: String(content ?? "").trim().slice(0, 80) || "Training run",
        });

        if (pending.status === "done" && pending.result) {
          // Claim it so the background listener doesn't render it twice.
          claimTrainingResult(threadId);
          res = pending.result;
          // eslint-disable-next-line no-console
          console.log("[analysis.sendMessage] deferred result", res);
        } else if (pending.status === "error") {
          const failMsg = pending.message || "Training failed. Please try again.";
          setMessages((curr) =>
            curr.map((m) =>
              m.id === placeholderId
                ? {
                    ...m,
                    thinking: false,
                    error: true,
                    content: `⚠️ ${failMsg}`,
                    time: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
                  }
                : m,
            ),
          );
          return false;
        } else {
          // superseded / none — a newer turn took over. Drop the bubble quietly.
          setMessages((curr) => curr.filter((m) => m.id !== placeholderId));
          return true;
        }
      }


      updatePlannerGraphAvailability(res);

      // Render the LAST assistant message's explanatory text. Any tabular
      // payload is stripped — the result table belongs in the Data table panel.
      const allMsgs = (res.messages ?? []) as any[];
      const lastAssistant = [...allMsgs]
        .reverse()
        .find(
          (m) =>
            m && m.role === "assistant" && typeof m.content === "string" && m.content.trim().length > 0,
        );
      const extracted = lastAssistant
        ? extractTabularContent(lastAssistant.content as string)
        : { text: "", tables: [] as ParsedTable[] };
      const lastAssistantText = extracted.text;
      const extractedTables = extracted.tables;

      const nowTime = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

      // Placeholder only when this turn produced an execution result AND the
      // assistant returned no usable text after stripping tables.
      const assistantMsgs = lastAssistantText
        ? [{ id: `${placeholderId}-a0`, role: "ai" as const, content: lastAssistantText, time: nowTime() }]
        : hasExecutionResult(res) || lastAssistant || extractedTables.length
          ? [{ id: `${placeholderId}-a0`, role: "ai" as const, content: ANALYSIS_COMPLETE_MESSAGE, time: nowTime() }]
          : [];

      // SCHEDULED TASK SIGNAL: if the backend returns task_info with a
      // task_id, this response belongs to the Scheduled Analysis phase.
      // Record it in the scheduled-task store and DO NOT render the output
      // on the normal Analysis page (charts/tables/insights are skipped).
      const scheduledTaskInfo = (res as any).task_info as { task_id?: string; next_due_at?: number } | undefined;
      const hasScheduledRows = Array.isArray(res.output_json) && res.output_json.length > 0;
      // Only treat as a scheduling-creation response when task_info is present
      // AND no output_json rows came back. A "result of task X" fetch also
      // returns task_info but includes output_json — we must render that.
      const isScheduledResponse = !!scheduledTaskInfo?.task_id && !hasScheduledRows;
      if (isScheduledResponse && scheduledTaskInfo?.task_id) {
        const schedule = extractSchedulePhrase(content) ?? "Scheduled";
        const nextRun = scheduledTaskInfo.next_due_at
          ? new Date(scheduledTaskInfo.next_due_at * 1000).toISOString()
          : null;
        currentUserLabel()
          .then((owner) => {
            scheduledTaskStore.upsert({
              task_id: scheduledTaskInfo.task_id!,
              name: content.split(/[.!?\n]/)[0]?.slice(0, 60) || "Scheduled analysis",
              prompt: content,
              schedule,
              status: "Active",
              next_run: nextRun,
              last_run: null,
              run_count: 0,
              created_at: new Date().toISOString(),
              owner,
              latest_output: Array.isArray(res.output_json) ? res.output_json : null,
              output_filename: (res as any).output_file_data?.filename ?? null,
            });
          })
          .catch(() => {});
        assistantMsgs.push({
          id: `${placeholderId}-sched`,
          role: "ai" as const,
          content: `📅 Scheduled task created — task_id: \`${scheduledTaskInfo.task_id}\` · ${schedule}. Open **Scheduled Analysis** to manage it.`,
          time: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
        });
      }

      // TRAINING RESULT: summarise the completed run under the assistant text.
      if ((res as any).training_completed === true) {
        const metrics = (res as any).training_metrics ?? (res as any).training_result?.metrics ?? null;
        const metricLines =
          metrics && typeof metrics === "object"
            ? Object.entries(metrics)
                .slice(0, 12)
                .map(([k, v]) => `- **${k}**: ${typeof v === "number" ? Number(v).toFixed(4) : String(v)}`)
                .join("\n")
            : "";
        const runId = (res as any).mlflow_run_id;
        const statusLabel = String((res as any).training_status ?? "completed");
        assistantMsgs.push({
          id: `${placeholderId}-train`,
          role: "ai" as const,
          content: [
            `**Training ${statusLabel}**`,
            runId ? `Run ID: \`${runId}\`` : "",
            metricLines,
          ]
            .filter(Boolean)
            .join("\n\n"),
          time: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
        });
      }

      setMessages((curr) => {
        const without = curr.filter((m) => m.id !== placeholderId);
        if (assistantMsgs.length) return [...without, ...assistantMsgs];
        return [
          ...without,
          {
            id: placeholderId,
            role: "ai",
            content: "Done.",
            time: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
          },
        ];
      });

      // If the assistant returned markdown tables with no structured output_json,
      // parse them into result runs so they render in the Data Table panel.
      const extractedTableRunIds: string[] = [];
      const hasStructuredRows = Array.isArray(res.output_json) && res.output_json.length > 0;
      // One request must map to exactly one result: when the backend already
      // returned structured rows, ignore markdown tables in the same reply.
      if (extractedTables.length && !isScheduledResponse && !hasStructuredRows) {
        const baseRunId = `chat-${Date.now()}`;
        extractedTables.forEach((table, idx) => {
          const runId = `${baseRunId}-table-${idx}`;
          extractedTableRunIds.push(runId);
          pushTableRun({
            id: runId,
            sectionId: targetSectionId,
            title: String(content ?? "").trim() || "Result",
            rows: table.rows,
            code: null,
            promptTs: null,
            file: null,
          });
        });
        setView("table");
      }

      // Backend visualization (if any). For prompt replies the backend
      // often returns only output_json; derive a small viz client-side so
      // every user prompt also gets a Chart view.
      const vc = (res as any).visualization_config;
      const vs = (res as any).visualization_status;
      const backendVizReady = vc && (!vs || vs === "ready");
      const hasPromptRows = Array.isArray(res.output_json) && res.output_json.length > 0;
      const hasExtractedRows = extractedTables.length > 0;
      const derivedViz = hasPromptRows ? deriveVizFromRows(res.output_json) : null;

      // User Insights must follow the latest prompt result. Some backend
      // message responses can echo the upload-time auto-insight config, so
      // when output_json exists we prefer a chart derived from those prompt
      // rows instead of reusing the stable Auto Insights chart.
      const effectiveViz = derivedViz ?? (backendVizReady ? vc : null);

      // The code that produced THIS result. Never reused across results.
      const coderDef: any = (res as any).coder_definition;
      const runCode: string | null =
        typeof coderDef?.code === "string" && coderDef.code.trim() ? coderDef.code : null;
      const runPromptTs: string | null =
        typeof (res as any).prompt_ts === "string"
          ? (res as any).prompt_ts
          : typeof coderDef?.prompt_ts === "string"
            ? coderDef.prompt_ts
            : null;
      const rawFile: any = (res as any).output_file_data;
      const runFile: OutputFile | null =
        rawFile && typeof rawFile.content === "string" && typeof rawFile.filename === "string"
          ? { filename: rawFile.filename, content: rawFile.content }
          : null;

      // Persist assistant reply(ies). Attach output_json + viz_config as
      // structured output so we can rehydrate both table and chart later.
      if (aid) {
        const output =
          (Array.isArray(res.output_json) && res.output_json.length > 0) ||
          effectiveViz ||
          extractedTables.length
            ? {
                output_json: Array.isArray(res.output_json)
                  ? res.output_json
                  : extractedTables[0]?.rows ?? undefined,
                extracted_tables: extractedTables.length ? extractedTables : undefined,
                viz_config: effectiveViz ?? undefined,
                section_id: targetSectionId,
                code: runCode ?? undefined,
                prompt_ts: runPromptTs ?? undefined,
                output_file: runFile ?? undefined,
              }
            : undefined;

        if (assistantMsgs.length) {
          for (const am of assistantMsgs) {
            persistMessage({
              analysis_id: aid,
              role: "assistant",
              content: am.content,
              output,
            }).catch((err) => console.warn("Failed to save assistant message", err));
          }
        } else {
          persistMessage({
            analysis_id: aid,
            role: "assistant",
            content: "Done.",
            output,
          }).catch((err) => console.warn("Failed to save assistant message", err));
        }

        // Analysis-completed notification intentionally removed — no popup.

      }

      // Skip all output rendering for scheduled tasks — those live on the
      // Scheduled Analysis page and must never mix into normal results.
      if (!isScheduledResponse) {
        // Shared id so the chart entry and the table run refer to the same run.
        const runId = `chat-${Date.now()}`;
        // output_json -> table (appended as its own run; never overwritten)
        if (Array.isArray(res.output_json) && res.output_json.length > 0) {
          setAnalysisRows(res.output_json as Record<string, unknown>[]);
          setUserInsightsRows(res.output_json as Record<string, unknown>[]);
          pushTableRun({
            id: runId,
            sectionId: targetSectionId,
            title: String(content ?? "").trim() || "Result",
            rows: res.output_json as Record<string, unknown>[],
            code: runCode,
            promptTs: runPromptTs,
            file: runFile,
          });
          setView("table");
        } else if (res.output_json == null) {
          // leave previous rows in place
        }

        // output_file_data -> csv download
        if (runFile) {
          setOutputFile(runFile);
        }


        // visualization (backend wins, else use client-derived)
        if (effectiveViz) {
          setUserInsightsViz(effectiveViz);
          setUserInsightsList((prev) => {
            const key = vizFingerprint(effectiveViz);
            if (prev.some((entry) => entry.sectionId === targetSectionId && vizFingerprint(entry.vizConfig) === key)) {
              return prev;
            }
            return [...prev, { id: runId, sectionId: targetSectionId, vizConfig: effectiveViz }];
          });
          setView(backendVizReady ? "chart" : hasPromptRows || extractedTables.length ? "table" : "chart");
          // Link the user message to the produced insight section for
          // navigation, but preserve the user's exact typed text verbatim
          // — do NOT overwrite `content` with the workspace heading or
          // dataset name.
          setMessages((curr) =>
            curr.map((m) =>
              m.id === userId
                ? {
                    ...m,
                    sectionId: targetSectionId,
                    insight: false,
                    content,
                  }
                : m,
            ),
          );
          activeSectionIdRef.current = targetSectionId;
          setActiveSectionId(targetSectionId);
          if (aid && backendVizReady && !hasPromptRows) {
            persistUpdateAnalysisContext(aid, { viz_config: vc }).catch((err) =>
              console.warn("Failed to persist viz_config", err),
            );
          }
        }
      }

      if ((res as any).analysis_fidelity) setAnalysisFidelity((res as any).analysis_fidelity);
      if ((res as any).selected_sample_name) setSelectedSampleName((res as any).selected_sample_name);

      const respDatasets = (res as any).datasets;
      const activeIds = (res as any).active_dataset_ids;
      if (Array.isArray(respDatasets) && respDatasets.length) {
        // Reconcile against the existing group: keep every uploaded dataset
        // (with its samples / visualization_config) and only append new ones.
        setDatasetChips((prev) => {
          const next = [...prev];
          respDatasets.forEach((d: any) => {
            if (!d?.dataset_id) return;
            const idx = next.findIndex((c) => c.id === d.dataset_id);
            const patch = {
              id: d.dataset_id,
              name: d.alias || d.filename || d.dataset_id,
              visualizationConfig: d.visualization_config ?? d.visualization_configs,
              visualizationStatus: d.visualization_status,
            };
            if (idx === -1) next.push(patch as DatasetChip);
            else
              next[idx] = {
                ...next[idx],
                name: next[idx].name || patch.name,
                visualizationConfig: next[idx].visualizationConfig ?? patch.visualizationConfig,
                visualizationStatus: next[idx].visualizationStatus ?? patch.visualizationStatus,
              };
          });
          return next;
        });
      }
      if (Array.isArray(activeIds) && activeIds.length) {
        // Highlight only — never prunes the uploaded group.
        setSelectedDatasetIds((prev) => {
          const ids = activeIds.filter((id: any) => typeof id === "string");
          return ids.length ? ids : prev;
        });
      }


      if (aid) {
        void refreshOneThreadInsightSource(aid);
      }
      return true;
    } catch (err: any) {
      const msg = err?.message || "Something went wrong sending your message.";
      const serviceIsRestarting =
        err instanceof BackendApiError && err.code === "SERVICE_RESTARTING";
      setMessages((curr) =>
        curr.map((m) =>
          m.id === placeholderId
            ? {
                ...m,
                thinking: false,
                error: true,
                content: serviceIsRestarting
                  ? `⚠️ **Service temporarily unavailable**\n\n${msg}`
                  : `⚠️ ${msg}`,
                time: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
              }
            : m,
        ),
      );
      return false;
    } finally {
      setIsSending(false);
    }
  };

  const projectContextLoading = !!analysisId && !ctx;

  return (
    <div className="flex h-screen w-full bg-primary text-primary">
      {projectContextLoading ? (
        <aside className="flex h-full w-[64px] shrink-0 flex-col items-center gap-3 border-r border-secondary bg-primary py-3">
          <div className="mb-1 size-9 animate-pulse rounded-lg bg-tertiary" />
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={`rail-skel-${i}`} className="size-10 animate-pulse rounded-lg bg-tertiary" />
          ))}
        </aside>
      ) : (
      <IconRail
        variant={resolvedProjectId ? "analysis-project" : "default"}
        active={
          resolvedProjectId
            ? pane === "project"
              ? "projects"
              : insightsOpen
                ? "reports"
                : restorePanelOpen
                  ? "restore"
                  : "chat"
            : undefined
        }
        skipNavIds={resolvedProjectId ? ["chat", "projects", "sources", "restore", "reports"] : undefined}
        onItemClick={
          resolvedProjectId
            ? (id: string) => {
                if (id === "chat") {
                  setPane("workspace");
                  setHistoryOpen(false);
                  setRestorePanelOpen(false);
                  setInsightsOpen(false);
                  setConnectCloudOpen(false);
                  setSourcesOpen(false);
                  setChatOpen(true);
                } else if (id === "projects") {
                  setPane("project");
                  setHistoryOpen(false);
                  setRestorePanelOpen(false);
                  setConnectCloudOpen(false);
                  setSourcesOpen(false);
                } else if (id === "sources") {
                  setPane("workspace");
                  setHistoryOpen(false);
                  setRestorePanelOpen(false);
                  setInsightsOpen(false);
                  setConnectCloudOpen(false);
                  setSourcesOpen(true);
                } else if (id === "restore") {
                  setPane("workspace");
                  setInsightsOpen(false);
                  setHistoryOpen(false);
                  setConnectCloudOpen(false);
                  setSourcesOpen(false);
                  setRestorePanelOpen((open) => {
                    const next = !open;
                    if (next) void fetchVersions();
                    return next;
                  });
                } else if (id === "reports") {
                  setInsightsOpen((open) => !open);
                  setHistoryOpen(false);
                  setRestorePanelOpen(false);
                  setConnectCloudOpen(false);
                  setSourcesOpen(false);
                }
              }
            : undefined
        }
      />
      )}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2 bg-secondary p-2">
        {resolvedProjectId && !hydrating ? (
          <div className="rounded-xl bg-primary">
            <div className="flex items-center gap-3 px-5 py-3">
              <div className="min-w-0 flex-1">
                <p className="truncate text-md font-semibold text-primary">{workspaceHeading || "New Analysis"}</p>
                {ctx?.project ? <p className="truncate text-sm text-tertiary">{ctx.project}</p> : null}
              </div>
              <div className="ml-auto flex items-center gap-2">
                <ShareCollaboratorsButton
                  analysisId={resolvedAnalysisId ?? aidFromUrl ?? null}
                  projectId={resolvedProjectId}
                  open={shareCollaboratorsOpen}
                  onOpenChange={setShareCollaboratorsOpen}
                  onCollaborationChange={handleCollaborationChange}
                />
                <ProjectNotificationsPopover projectId={resolvedProjectId} />
              </div>
            </div>
          </div>
        ) : null}
        <div className="flex min-h-0 flex-1 gap-2 overflow-hidden">
          {restorePanelOpen && pane === "workspace" && (
            <aside className="flex h-full w-[300px] shrink-0 flex-col overflow-hidden rounded-xl border border-secondary bg-primary">
              <div className="flex items-start justify-between gap-2 border-b border-secondary px-4 py-3">
                <div>
                  <p className="text-md font-semibold text-primary">Version History</p>
                  <p className="text-xs text-tertiary">Restore an earlier version</p>
                </div>
                <button
                  type="button"
                  aria-label="Close restore panel"
                  onClick={() => setRestorePanelOpen(false)}
                  className="rounded-md p-1 text-fg-quaternary hover:bg-primary_hover"
                >
                  <X className="size-5" />
                </button>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto">
                {versionsLoading ? (
                  <p className="px-4 py-6 text-center text-xs text-tertiary">Loading…</p>
                ) : versions.length === 0 ? (
                  <p className="px-4 py-6 text-center text-xs text-tertiary">No versions yet.</p>
                ) : (
                  <ul className="divide-y divide-secondary">
                    {versions.map((v) => (
                      <li key={v.prompt_ts} className="flex items-start gap-3 px-4 py-3">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <p className="truncate text-sm font-medium text-primary">{formatPromptTs(v.prompt_ts)}</p>
                            {v.is_current && (
                              <span className="rounded-full bg-[#eff5ff] px-1.5 py-0.5 text-[10px] font-medium text-[#175cd3]">
                                Current
                              </span>
                            )}
                          </div>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {v.has_code && (
                              <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary">Code</span>
                            )}
                            {v.has_output && (
                              <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary">Output</span>
                            )}
                            {v.has_viz && (
                              <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary">Chart</span>
                            )}
                          </div>
                        </div>
                        {!v.is_current && (
                          <button
                            type="button"
                            disabled={restoringTs !== null}
                            onClick={() => void handleRestore(v.prompt_ts)}
                            className="shrink-0 rounded-md border border-secondary px-2 py-1 text-xs font-medium text-primary hover:bg-secondary disabled:opacity-50"
                          >
                            {restoringTs === v.prompt_ts ? "Restoring…" : "Restore"}
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </aside>
          )}
          {historyOpen && pane === "workspace" && (
            <HistoryVersionPanel
              active={historyOpen}
              analysisId={resolvedAnalysisId}
              refreshKey={`${messages.length}:${collaborationRevision}`}
              liveChat={historyLiveChat}
              onClose={() => setHistoryOpen(false)}
              onView={(item) => {
                const payload = resolveHistoryViewPayload(item.output);
                if (!payload) {
                  toast.error("No detail available for this entry");
                  return;
                }

                setPane("workspace");
                setHistoryOpen(false);

                if (payload.kind === "viz") {
                  setAnalysisVizConfig(payload.config);
                  setUserInsightsViz(payload.config);
                  setView("chart");
                  toast.success("Restored chart view");
                } else if (payload.kind === "table") {
                  setAnalysisRows(payload.rows);
                  pushTableRun({
                    id: `history-${Date.now()}`,
                    sectionId: activeSectionId ?? "",
                    title: "Restored result",
                    rows: payload.rows as Record<string, unknown>[],
                  });
                  setView("table");
                  toast.success("Restored data table");
                } else if (payload.kind === "message") {
                  setChatOpen(true);
                  toast.success("Opened chat");
                } else if (payload.kind === "share" || payload.kind === "collaborator") {
                  setCollaborationDetail(payload);
                }
              }}
            />
          )}

          {insightsOpen && (
            <AllInsightsPanel
              sources={threadInsightSources}
              onClose={() => setInsightsOpen(false)}
              onSelectSlide={() => {
                setPane("workspace");
                setView("chart");
              }}
            />
          )}

          {/* Main canvas */}
          <main className="relative flex flex-1 flex-col overflow-hidden rounded-xl bg-primary">
            {hydrating ? (
              <AnalysisSkeleton />
            ) : (
              <>
                {resolvedProjectId && tabs.length > 0 ? (
                  <div className="border-b border-secondary bg-primary">
                    <div className="flex items-center gap-4 px-5">
                      {tabs.map((tab) => (
                        <button
                          key={tab.id}
                          type="button"
                          onClick={() => {
                            const targetId = tab.id.startsWith("analysis-") ? tab.id.slice("analysis-".length) : tab.id;
                            void switchAnalysisTab(targetId);
                          }}
                          className={cx(
                            "relative py-3 text-sm font-semibold transition",
                            activeTabId === tab.id ? "text-[#1565ef]" : "text-tertiary hover:text-secondary",
                          )}
                        >
                          {tab.title}
                          {activeTabId === tab.id ? (
                            <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-[#1565ef]" />
                          ) : null}
                        </button>
                      ))}
                      <button
                        type="button"
                        aria-label="Add analysis"
                        title="Upload a new dataset as a child analysis"
                        onClick={openChildUpload}
                        className="relative inline-flex items-center py-3 text-tertiary transition hover:text-[#1565ef]"
                      >
                        <Plus className="size-4" />
                      </button>
                    </div>
                  </div>
                ) : null}

                {pane === "workspace" && (
                  <>
                    {!resolvedProjectId ? (
                      <div className="border-b border-secondary bg-primary">
                        <div className="flex items-center gap-3 px-5 py-3">
                          <div className="flex min-w-0 flex-1 items-center gap-2">
                            <Folder className="size-5 text-fg-secondary" />
                            <p className="truncate text-md font-semibold text-primary">
                              {workspaceHeading || "New Analysis"}
                            </p>
                          </div>
                          <div className="ml-auto flex items-center gap-2">
                            <button
                              onClick={() => setAutoInsightsOpen(true)}
                              className="inline-flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
                            >
                              <Stars02 className="size-4 text-[#1565ef]" />
                              Auto Insights
                            </button>
                            <button
                              onClick={() => setUserInsightsOpen(true)}
                              disabled={!userInsightsViz && !userInsightsRows?.length}
                              title={
                                userInsightsViz || userInsightsRows?.length
                                  ? "View insights from your latest prompt"
                                  : "Send a prompt to generate insights"
                              }
                              className="inline-flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              <Stars02 className="size-4 text-[#7a5af8]" />
                              User Insights
                            </button>
                            <button
                              onClick={() => {
                                if (upgradeBlocked("Moving analyses into a project")) return;
                                setMoveOpen(true);
                              }}

                              className="inline-flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
                            >
                              <Folder className="size-4 text-fg-secondary" />
                              Move to project
                            </button>
                            <ShareCollaboratorsButton
                              analysisId={resolvedAnalysisId ?? aidFromUrl ?? null}
                              projectId={resolvedProjectId}
                              open={shareCollaboratorsOpen}
                              onOpenChange={setShareCollaboratorsOpen}
                              onCollaborationChange={handleCollaborationChange}
                            />
                          </div>
                        </div>
                      </div>
                    ) : null}

                    {/* Body */}
                    <div className="relative min-h-0 flex-1">
                      {workspaceSections.length > 1 ? (
                        <div className="pointer-events-none absolute inset-y-0 right-3 z-10 hidden items-center md:flex">
                          <div className="pointer-events-auto flex max-h-[min(50vh,320px)] flex-col items-center gap-2 overflow-y-auto py-2">
                            {workspaceSections.map((s, index) => (
                              <button
                                key={s.id}
                                type="button"
                                aria-label={`Go to section ${index + 1}`}
                                title={s.title ?? `Section ${index + 1}`}
                                onClick={() => focusSection(s.id)}
                                className="grid size-3 shrink-0 place-items-center"
                              >
                                <span
                                  className={cx(
                                    "size-2 rounded-full transition-colors",
                                    activeSectionId === s.id ? "bg-[#1565ef]" : "bg-[#d0d5dd]",
                                  )}
                                />
                              </button>
                            ))}
                          </div>
                        </div>
                      ) : null}

                      <div className="h-full overflow-y-auto px-4 py-4">
                        <div key={activeTab?.id} className="mx-auto flex w-full max-w-none flex-col">
                          <div
                            ref={(el) => {
                              sectionRefs.current[currentMainSectionId] = el;
                            }}
                            data-section-id={currentMainSectionId}
                            className={cx(
                              "scroll-mt-4 rounded-xl p-4 transition-[box-shadow,border-color]",
                              sectionHighlightClass(currentMainSectionId),
                            )}

                          >
                            <div className="mb-3 flex items-center justify-end gap-3">

                              <div className="flex items-center gap-2">
                                {view === "table" && selectedRun?.rows?.length ? (
                                  <button
                                    type="button"
                                    onClick={downloadSelectedRunCsv}
                                    className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                                  >
                                    Download CSV
                                  </button>
                                ) : outputFile ? (
                                  <a
                                    href={outputFile.content}
                                    download={outputFile.filename}
                                    className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                                  >
                                    Download CSV
                                  </a>
                                ) : null}
                                {plannerGraphAvailable && threadId ? (
                                  <button
                                    type="button"
                                    onClick={openPlannerGraph}
                                    disabled={plannerGraphLoading}
                                    className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover disabled:opacity-50"
                                  >
                                    {plannerGraphLoading ? "Loading…" : "Planner Graph"}
                                  </button>
                                ) : null}
                                {plannerGraphError ? (
                                  <span className="text-xs text-error-primary">{plannerGraphError}</span>
                                ) : null}
                                <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-1 shadow-xs">
                                  <button
                                    onClick={() => setView("table")}
                                    className={cx(
                                      "rounded-md px-3 py-1.5 text-sm font-semibold transition",
                                      view === "table" ? "bg-secondary text-primary" : "text-tertiary",
                                    )}
                                  >
                                    Data table
                                  </button>
                                  <button
                                    onClick={() => setView("chart")}
                                    className={cx(
                                      "rounded-md px-3 py-1.5 text-sm font-semibold transition",
                                      view === "chart" ? "bg-secondary text-primary" : "text-tertiary",
                                    )}
                                  >
                                    Chart
                                  </button>
                                  <button
                                    onClick={() => setView("preview")}
                                    className={cx(
                                      "rounded-md px-3 py-1.5 text-sm font-semibold transition",
                                      view === "preview" ? "bg-secondary text-primary" : "text-tertiary",
                                    )}
                                    title="View uploaded dataset preview"
                                  >
                                    Data Preview
                                  </button>
                                </div>
                              </div>
                            </div>

                            {view === "preview" ? (
                              <>
                                <DataPreviewPanel
                                  dataset={persistedDataset}
                                  filename={autoInsightsDatasetName ?? navState?.filename}
                                />
                                {renderResultToolbar()}
                              </>
                            ) : view === "chart" ? (
                              resolvedAutoViz || multiDatasetInsights || mainSectionUserSlides.length > 0 ? (
                                <div className="mt-4 space-y-6">
                                  {!chartSamples?.length && mainSectionUserSlides.length === 0 ? (
                                    <p className="text-sm text-tertiary">Loading chart data…</p>
                                  ) : autoChartSlides.length === 0 && mainSectionUserSlides.length === 0 ? (
                                    <>
                                      <WorkspaceSectionUserInsights
                                        sectionId={currentMainSectionId}
                                        entries={userInsightEntries}
                                        activeTab={undefined}
                                        chartSamples={chartSamples}
                                        excludeTitles={autoChartTitles}
                                        dismissedKeys={dismissedInsightCardKeys}
                                        onDismiss={dismissInsightCard}
                                        activeRunId={selectedRunId}
                                        onRunChange={setSelectedRunId}
                                      />
                                      <p className="text-sm text-tertiary">No chart available for this tab.</p>
                                    </>
                                  ) : (
                                    <>
                                      {/* User insights (from user prompts) — always rendered on top */}
                                      <WorkspaceSectionUserInsights
                                        sectionId={currentMainSectionId}
                                        entries={userInsightEntries}
                                        activeTab={undefined}
                                        chartSamples={chartSamples}
                                        excludeTitles={autoChartTitles}
                                        dismissedKeys={dismissedInsightCardKeys}
                                        onDismiss={dismissInsightCard}
                                        activeRunId={selectedRunId}
                                        onRunChange={setSelectedRunId}
                                      />
                                      {/* Auto insights (generated at upload) — always rendered below.
                                          With several datasets selected, render the union grouped per dataset. */}
                                      {multiDatasetInsights ? (
                                        <div className="space-y-6">
                                          {autoInsightGroups.map((g) => (
                                            <div key={g.id} className="space-y-3">
                                              <div className="flex items-center gap-2">
                                                <h3 className="truncate text-sm font-semibold text-primary">{g.name}</h3>
                                                <span className="rounded-full bg-[#1565ef]/10 px-2 py-0.5 text-[11px] font-semibold text-[#1565ef]">
                                                  {g.slides.length} chart{g.slides.length === 1 ? "" : "s"}
                                                </span>
                                              </div>
                                              {g.slides.length ? (
                                                <WorkspaceChartCards
                                                  slides={g.slides as any}
                                                  keyPrefix={`auto-${g.id}`}
                                                  dismissedKeys={dismissedInsightCardKeys}
                                                  onDismiss={dismissInsightCard}
                                                />
                                              ) : (
                                                <p className="text-sm text-tertiary">
                                                  No auto-insights available for this dataset
                                                </p>
                                              )}
                                            </div>
                                          ))}
                                        </div>
                                      ) : (
                                        <WorkspaceChartCards
                                          slides={autoChartSlides}
                                          keyPrefix="auto"
                                          dismissedKeys={dismissedInsightCardKeys}
                                          onDismiss={dismissInsightCard}
                                        />
                                      )}

                                    </>
                                  )}
                                </div>
                              ) : insightsGenerating ? (
                                <div className="mt-8 flex flex-col items-center gap-3 text-sm text-tertiary">
                                  <span className="size-4 animate-spin rounded-full border-2 border-[#1565ef]/30 border-t-[#1565ef]" />
                                  Generating auto insights…
                                </div>
                              ) : isPersistedAnalysis ? (
                                <p className="mt-8 text-center text-sm text-tertiary">
                                  No chart saved for this analysis yet.
                                </p>

                              ) : (
                                <>
                                  {/* Chart */}
                                  <div className="mt-4 h-[340px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                      <ComposedChart
                                        data={chartData}
                                        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                                      >
                                        <defs>
                                          <linearGradient id="fill2020" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#1565ef" stopOpacity={0.25} />
                                            <stop offset="100%" stopColor="#1565ef" stopOpacity={0.05} />
                                          </linearGradient>
                                          <linearGradient id="fill2021" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#f97066" stopOpacity={0.2} />
                                            <stop offset="100%" stopColor="#f97066" stopOpacity={0.05} />
                                          </linearGradient>
                                        </defs>
                                        <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 3" />
                                        <XAxis
                                          dataKey="month"
                                          tick={{ fill: "#667085", fontSize: 12 }}
                                          tickLine={false}
                                          axisLine={false}
                                        />
                                        <YAxis
                                          domain={[-100, 100]}
                                          ticks={[-100, -50, 0, 50, 100]}
                                          tick={{ fill: "#667085", fontSize: 12 }}
                                          tickLine={false}
                                          axisLine={false}
                                        />
                                        <ReferenceLine y={0} stroke="#1565ef" strokeWidth={1.5} />
                                        <Area
                                          type="linear"
                                          dataKey="y2021"
                                          stroke="#f97066"
                                          strokeWidth={1.5}
                                          fill="url(#fill2021)"
                                          dot={{ fill: "#f97066", stroke: "#fff", strokeWidth: 2, r: 5 }}
                                          activeDot={{ r: 6 }}
                                        />
                                        <Area
                                          type="linear"
                                          dataKey="y2020"
                                          stroke="#1565ef"
                                          strokeWidth={1.5}
                                          fill="url(#fill2020)"
                                          dot={{ fill: "#1565ef", stroke: "#fff", strokeWidth: 2, r: 5 }}
                                          activeDot={{ r: 6 }}
                                        />
                                      </ComposedChart>
                                    </ResponsiveContainer>
                                  </div>

                                  {/* Legend */}
                                  <div className="mt-2 flex items-center justify-center gap-6 text-xs text-secondary">
                                    <span className="inline-flex items-center gap-2">
                                      <span className="size-2.5 rounded-sm bg-[#1565ef]" /> 2020
                                    </span>
                                    <span className="inline-flex items-center gap-2">
                                      <span className="size-2.5 rounded-sm bg-[#f97066]" /> 2021
                                    </span>
                                  </div>
                                </>
                              )
                            ) : (
                              <div>
                                {tableRuns.length > 1 && (
                                  <div className="mt-4 flex items-center justify-between gap-3 rounded-lg border border-secondary bg-primary px-3 py-2">
                                    <p className="min-w-0 truncate text-xs text-tertiary" title={selectedRun?.title}>
                                      {selectedRun?.title || "Result"}
                                    </p>
                                    <div className="flex shrink-0 items-center gap-1">
                                      <button
                                        type="button"
                                        aria-label="Previous result"
                                        onClick={() => goToRun(-1)}
                                        className="rounded-md border border-secondary bg-primary p-1 text-fg-secondary hover:bg-primary_hover"
                                      >
                                        <ChevronLeft className="size-4" />
                                      </button>
                                      <span className="px-1 text-xs text-tertiary">
                                        {selectedRunIndex + 1} / {tableRuns.length}
                                      </span>
                                      <button
                                        type="button"
                                        aria-label="Next result"
                                        onClick={() => goToRun(1)}
                                        className="rounded-md border border-secondary bg-primary p-1 text-fg-secondary hover:bg-primary_hover"
                                      >
                                        <ChevronRight className="size-4" />
                                      </button>
                                    </div>
                                  </div>
                                )}
                                <DataTableView
                                  key={selectedRun?.id ?? "latest"}
                                  data={selectedRun?.rows ?? analysisRows ?? undefined}
                                  dataset={persistedDataset}
                                  disableMockFallback={isPersistedAnalysis}
                                />
                                {/* Toolbar for THIS result, directly under its data table */}
                                {renderResultToolbar()}
                              </div>
                            )}

                            {/* Same toolbar directly under the charts of this result */}
                            {view === "chart" ? renderResultToolbar() : null}

                            {/* Key Insights */}
                            <CollapsibleInsights>
                              <ul className="list-disc space-y-1.5 pl-5 text-sm text-secondary">
                                {resolvedAutoViz ? (
                                  hasTabInsights ? (
                                    <>
                                      {tabVizInsights.summaries.map((summary, i) => (
                                        <li key={`summary-${i}`}>
                                          <InlineMarkdown content={summary} />
                                        </li>
                                      ))}
                                      {tabVizInsights.insights.map((insight, i) => (
                                        <li key={`insight-${i}`}>
                                          <InlineMarkdown content={insight} />
                                        </li>
                                      ))}
                                    </>
                                  ) : (
                                    <li className="text-tertiary">
                                      Chart insights will appear here once analysis completes.
                                    </li>
                                  )
                                ) : insightsGenerating ? (
                                  <li className="text-tertiary">Generating key insights…</li>
                                ) : isPersistedAnalysis ? (
                                  <li className="text-tertiary">No insights saved for this analysis yet.</li>
                                ) : (
                                  <li className="text-tertiary">Insights will appear here once analysis completes.</li>
                                )}
                              </ul>

                            </CollapsibleInsights>

                            {pastedPreviews.map(({ groupId, runs, slides, insights }, pastedIndex) => (
                              <PastedAnalysisSection
                                key={groupId}
                                runs={runs}
                                slides={slides as RunSlide[]}
                                insights={insights}
                                index={pastedIndex}
                              />
                            ))}




                            {codeOpen && (
                              <InlineCodeEditor
                                key={codeRunId ?? "latest"}
                                loading={codeLoading}
                                code={codeText}
                                originalCode={originalCodeText}
                                downloadUrl={codeDownloadUrl}
                                onChange={handleCodeChange}
                                onReset={() => handleCodeChange(originalCodeText)}
                                onClose={() => setCodeOpen(false)}
                                onSaveAndExecute={handleSaveAndExecute}
                                saving={savingExec}
                                plannerNotes={plannerNotes}
                                validationFeedback={validationFeedback}
                                onDismissPlanner={() => setPlannerNotes(null)}
                                onDismissValidation={() => setValidationFeedback(null)}
                              />
                            )}


                            {insightCommentOpen && (
                              <div className="mt-4">
                                <InsightCommentCard analysisId={resolvedAnalysisId} />
                              </div>
                            )}
                          </div>

                          {(extraSectionsByTab[resolvedTabId] ?? []).map((s) => (
                            <div
                              key={s.id}
                              ref={(el) => {
                                sectionRefs.current[s.id] = el;
                              }}
                              data-section-id={s.id}
                              className={cx(
                                "mt-8 min-h-[120px] scroll-mt-4 rounded-xl p-4 transition-[box-shadow,border-color]",
                                sectionHighlightClass(s.id),
                              )}
                            >
                              <WorkspaceSectionUserInsights
                                sectionId={s.id}
                                entries={userInsightEntries}
                                activeTab={undefined}
                                chartSamples={chartSamples}
                                dismissedKeys={dismissedInsightCardKeys}
                                onDismiss={dismissInsightCard}
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </>
                )}
                {pane === "project" && (
                  <ProjectDashboardView
                    projectId={resolvedProjectId}
                    analysisId={resolvedAnalysisId}
                    activeTabId={activeTabId || resolvedTabId}
                    vizConfig={analysisVizConfig ?? autoInsightsConfig}
                    resolveVizConfig={resolveThreadVizConfig}
                    samples={chartSamples ?? null}
                    insightsOpen={insightsOpen}
                    onToggleInsights={() => {
                      setInsightsOpen((open) => !open);
                      setHistoryOpen(false);
                      setConnectCloudOpen(false);
                      setSourcesOpen(false);
                    }}
                    onVizConfigUpdated={(nextViz, updatedAnalysisId) => {
                      setAnalysisVizConfig(nextViz);
                      if (updatedAnalysisId === resolvedAnalysisId) {
                        setAutoInsightsConfig(nextViz);
                      }
                      if (motherAnalysisId) {
                        void refreshThreadInsightSources(motherAnalysisId, childAnalyses);
                      }
                    }}
                  />
                )}
              </>
            )}
          </main>

          {/* Chat panel */}
          {chatOpen && pane === "workspace" && (() => {
            const isFloat = chatDock === "float";
            const isLeft = chatDock === "left";

            const dockButton = (target: ChatDock, label: string, children: ReactNode) => (
              <button
                type="button"
                onClick={() => setChatDock(target)}
                aria-label={label}
                title={label}
                className={cx(
                  "grid size-7 place-items-center rounded-md text-fg-quaternary hover:bg-secondary hover:text-fg-secondary",
                  chatDock === target && "bg-[#eff8ff] text-[#1565ef]",
                )}
              >
                {children}
              </button>
            );

            const header = (
              <div
                className={cx(
                  "flex items-center gap-2 border-b border-secondary px-4 py-2.5",
                  isFloat && "cursor-move select-none",
                )}
                onMouseDown={
                  isFloat
                    ? (e) => {
                        if ((e.target as HTMLElement).closest("button")) return;
                        e.preventDefault();
                        floatDraggingRef.current = true;
                        const startX = e.clientX;
                        const startY = e.clientY;
                        const startPos = { ...chatFloatPos };
                        let latest = { ...startPos };
                        let raf = 0;
                        const applyStyle = () => {
                          raf = 0;
                          const el = floatAsideRef.current;
                          if (el) {
                            el.style.left = `${latest.x}px`;
                            el.style.top = `${latest.y}px`;
                          }
                        };
                        const onMove = (ev: MouseEvent) => {
                          if (!floatDraggingRef.current) return;
                          const nx = Math.max(8, Math.min(window.innerWidth - chatFloatSize.w - 8, startPos.x + (ev.clientX - startX)));
                          const ny = Math.max(8, Math.min(window.innerHeight - 60, startPos.y + (ev.clientY - startY)));
                          latest = { x: nx, y: ny };
                          if (!raf) raf = requestAnimationFrame(applyStyle);
                        };
                        const onUp = () => {
                          floatDraggingRef.current = false;
                          window.removeEventListener("mousemove", onMove);
                          window.removeEventListener("mouseup", onUp);
                          if (raf) cancelAnimationFrame(raf);
                          // Snap to dock if dropped near edges
                          const snapThreshold = 60;
                          if (latest.x <= snapThreshold) {
                            setChatDock("left");
                          } else if (latest.x + chatFloatSize.w >= window.innerWidth - snapThreshold) {
                            setChatDock("right");
                          } else {
                            setChatFloatPos(latest);
                          }
                        };
                        window.addEventListener("mousemove", onMove);
                        window.addEventListener("mouseup", onUp);
                      }
                    : undefined
                }
              >
                <h2 className="text-md font-semibold text-primary flex-1">Chat</h2>
                <div className="flex items-center gap-0.5">
                  {dockButton("left", "Dock left", <ArrowLeft className="size-4" />)}
                  {dockButton(
                    "float",
                    "Pop out",
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="size-4">
                      <path d="M14 3h7v7" />
                      <path d="M10 14 21 3" />
                      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
                    </svg>,
                  )}
                  {dockButton("right", "Dock right", <ArrowRight className="size-4" />)}
                </div>
              </div>
            );

            const body = (
              <>
                {avatarMode ? (
                  <VoicePanel
                    onClose={() => {
                      stopSpeaking();
                      setAvatarMode(false);
                    }}
                    messages={messages}
                    isSending={isSending}
                    listening={voiceListening}
                  />

                ) : (
                  <div ref={scrollRef} className="flex flex-1 flex-col overflow-y-auto px-4 py-4">
                    <div className="mt-auto flex flex-col">
                      <div className="my-3 flex items-center gap-3">
                        <div className="h-px flex-1 bg-border-secondary" />
                        <span className="text-xs text-tertiary">Today</span>
                        <div className="h-px flex-1 bg-border-secondary" />
                      </div>
                      {messages.map((m) => (
                        <MessageRow
                          key={m.id}
                          message={m}
                          activeSectionId={activeSectionId}
                          onSectionSelect={focusSection}
                          onReply={() =>
                            setReplyContext({
                              label: m.role === "ai" ? "Avaloka AI" : "You",
                              content: m.content,
                            })
                          }
                          onRefresh={() => {
                            void handleRefresh();
                          }}
                          onThumb={(t) => {
                            void handleThumb(t);
                          }}
                          onCopy={() => {
                            try {
                              void navigator.clipboard?.writeText(m.content);
                              toast.success("Copied");
                            } catch {
                              toast.error("Failed to copy");
                            }
                          }}
                          feedback={insightFeedback}
                          feedbackSubmitting={feedbackSubmitting}
                          refreshing={refreshing}
                        />
                      ))}
                    </div>
                  </div>
                )}

                <div className="border-t border-secondary bg-secondary/40 p-3">
                  {replyContext && (
                    <div className="mb-2 flex items-stretch overflow-hidden rounded-lg border border-secondary bg-primary">
                      <span className="w-1 shrink-0 bg-[#1565ef]" />
                      <div className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2">
                        <MessageChatCircle className="size-4 shrink-0 text-fg-quaternary" />
                        <div className="min-w-0 flex-1">
                          <p className="text-xs font-semibold text-[#1565ef]">{replyContext.label}</p>
                          <p className="truncate text-xs text-secondary">{replyContext.content}</p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => setReplyContext(null)}
                        aria-label="Cancel reply"
                        className="grid w-9 place-items-center text-fg-quaternary hover:text-fg-secondary"
                      >
                        <X className="size-4" />
                      </button>
                    </div>
                  )}
                  <TypingIndicator users={typingUsers} />
                  <ChatComposer
                    isSending={isSending}
                    avatarMode={avatarMode}
                    onAvatarModeChange={() => setAvatarMode((value) => !value)}
                    onListeningChange={setVoiceListening}
                    onSend={send}
                    onTyping={notifyTyping}
                    onStopTyping={stopTyping}
                  />
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {(datasetChips.length
                        ? datasetChips
                        : isPersistedAnalysis
                          ? []
                          : [{ id: "ds1", name: "Data Set 1" }]
                      ).map((d) => {
                        const active =
                          datasetChips.length <= 1
                            ? selectedDatasetIds.length === 0 || selectedDatasetIds.includes(d.id)
                            : selectedDatasetIds.includes(d.id);
                        return (
                          <button
                            type="button"
                            key={d.id}
                            onClick={() => selectDatasetChip(d as DatasetChip)}
                            title={`${d.name}\nDataset ID: ${d.id}`}
                            className={cx(
                              "inline-flex max-w-[240px] items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition",
                              active
                                ? "border-[#1565ef] bg-[#eff8ff] text-[#175cd3] shadow-xs"
                                : "border-secondary bg-primary text-tertiary hover:border-tertiary",
                            )}
                          >
                            <Database01
                              className={cx("size-3.5 shrink-0", active ? "text-[#1565ef]" : "text-fg-quaternary")}
                            />
                            <span className="truncate">{datasetChipLabel(d.name)}</span>
                          </button>
                        );
                      })}
                    </div>
                    <div className="flex items-center gap-2 text-xs font-semibold text-secondary">
                      <span className="text-primary">Selected Dataset</span>
                      <button
                        type="button"
                        onClick={() => setSourcesOpen(true)}
                        className="inline-flex items-center gap-1 hover:text-primary"
                      >
                        <Plus className="size-3.5" /> Add
                      </button>
                    </div>
                  </div>
                </div>
              </>
            );

            if (isFloat) {
              return (
                <aside
                  ref={floatAsideRef}
                  style={{
                    position: "fixed",
                    left: chatFloatPos.x,
                    top: chatFloatPos.y,
                    width: chatFloatSize.w,
                    height: chatFloatSize.h,
                    zIndex: 60,
                  }}
                  className="flex flex-col overflow-hidden rounded-xl border border-secondary bg-primary shadow-2xl"
                >
                  {header}
                  {body}
                </aside>
              );
            }

            const separator = (
              <div
                role="separator"
                aria-orientation="vertical"
                aria-label="Resize chat"
                onMouseDown={(e) => {
                  e.preventDefault();
                  chatResizingRef.current = true;
                  const startX = e.clientX;
                  const startW = chatWidth;
                  let latest = startW;
                  let raf = 0;
                  const apply = () => {
                    raf = 0;
                    if (chatAsideRef.current) chatAsideRef.current.style.width = `${latest}px`;
                  };
                  const onMove = (ev: MouseEvent) => {
                    if (!chatResizingRef.current) return;
                    const dx = isLeft ? ev.clientX - startX : startX - ev.clientX;
                    latest = Math.min(900, Math.max(300, startW + dx));
                    if (!raf) raf = requestAnimationFrame(apply);
                  };
                  const onUp = () => {
                    chatResizingRef.current = false;
                    if (raf) cancelAnimationFrame(raf);
                    window.removeEventListener("mousemove", onMove);
                    window.removeEventListener("mouseup", onUp);
                    setChatWidth(latest);
                  };
                  window.addEventListener("mousemove", onMove);
                  window.addEventListener("mouseup", onUp);
                }}
                style={{ order: isLeft ? -1 : 0 }}
                className="w-1 shrink-0 cursor-col-resize bg-transparent hover:bg-[#1565ef]/30 active:bg-[#1565ef]/50"
              />
            );

            return (
              <>
                {separator}
                <aside
                  ref={chatAsideRef}
                  style={{ width: chatWidth, order: isLeft ? -2 : 0 }}
                  className="flex shrink-0 flex-col overflow-hidden rounded-xl border border-secondary bg-primary"
                >
                  {header}
                  {body}
                </aside>
              </>
            );
          })()}
        </div>

        {upgradeDialog}
        <MoveProjectModal

          open={moveOpen}
          onOpenChange={setMoveOpen}
          analysisId={resolvedAnalysisId}
          onMoved={(name) => {
            setMoveOpen(false);
            setMoveSuccess(name);
          }}
        />
        <UploadModal
          open={childUploadOpen}
          onOpenChange={setChildUploadOpen}
          projectId={resolvedProjectId}
          projectName={ctx?.project ?? null}
          parentAnalysisId={motherAnalysisId}
          motherAnalysisId={motherAnalysisId}
        />
        <Dialog open={childPickerOpen} onOpenChange={setChildPickerOpen}>
          <DialogContent className="max-w-2xl bg-white p-6">
            <DialogTitle>New Analysis</DialogTitle>
            <button
              type="button"
              onClick={() => {
                setChildPickerOpen(false);
                setChildUploadOpen(true);
              }}
              className="group relative mt-2 block w-full cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed border-secondary bg-primary/60 px-6 py-10 text-center transition hover:border-brand"
            >
              <img src={bgPattern.url} alt="" aria-hidden className="pointer-events-none absolute left-1/2 top-1/2 w-[420px] -translate-x-1/2 -translate-y-1/2 select-none" />
              <div className="relative flex flex-col items-center">
                <img src={illustration.url} alt="" aria-hidden className="h-auto w-auto max-w-[96px] select-none object-contain" />
                <h3 className="mt-4 text-md font-semibold text-primary">Select or drag &amp; drop file to analyze data</h3>
                <p className="mt-1 text-sm text-tertiary">CSV, Excel or JSON (100MB)</p>
              </div>
            </button>
            <div className="my-6 flex items-center gap-4">
              <div className="h-px flex-1 bg-border-secondary" />
              <span className="text-sm text-tertiary">Or</span>
              <div className="h-px flex-1 bg-border-secondary" />
            </div>
            <DataSourceGrid />
          </DialogContent>
        </Dialog>
        <MoveSuccessModal
          open={!!moveSuccess}
          projectName={moveSuccess ?? ""}
          onOpenChange={(v) => {
            if (!v) setMoveSuccess(null);
          }}
        />
        <CurrentDatasetModal
          open={!!currentDataset}
          onOpenChange={(v) => {
            if (!v) setCurrentDataset(null);
          }}
          datasetName={currentDataset?.title ?? ""}
        />
        <DataSetBrowserModal open={sourcesOpen} onOpenChange={setSourcesOpen} />
        <ConnectCloudModal open={connectCloudOpen} onOpenChange={setConnectCloudOpen} />
        <AutoInsightsModal
          open={autoInsightsOpen}
          onOpenChange={setAutoInsightsOpen}
          config={autoInsightsConfig}
          status={autoInsightsStatus}
          datasetName={autoInsightsDatasetName}
          samples={chartSamples}
          datasets={autoInsightsDatasets}
          primaryDatasetId={selectedDatasetIds[selectedDatasetIds.length - 1] ?? datasetChips[0]?.id ?? null}
          aid={aidFromUrl}
        />

        {/* {plannerGraphUrl ? (
          <div
            className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 p-6"
            onClick={closePlannerGraph}
          >
            <div
              className="flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl bg-primary shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-secondary px-4 py-3">
                <h3 className="text-sm font-semibold text-primary">Planner Graph</h3>
                <div className="flex items-center gap-2">
                  <a
                    href={plannerGraphUrl}
                    download="planner-graph.png"
                    className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                  >
                    Download PNG
                  </a>
                  <button
                    type="button"
                    onClick={closePlannerGraph}
                    className="inline-flex items-center rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                  >
                    Close
                  </button>
                </div>
              </div>
              <div className="min-h-0 flex-1 overflow-auto p-4">
                <img src={plannerGraphUrl} alt="Planner graph for this analysis thread" className="mx-auto max-w-full" />
              </div>
            </div>
          </div>
        ) : null} */}

      {plannerGraphUrl ? (
          <div
            className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 p-2 sm:p-4"
            onClick={closePlannerGraph}
          >
            <div
              className="flex h-[94vh] w-[96vw] max-w-none flex-col overflow-hidden rounded-xl bg-primary shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-secondary px-4 py-3">
                <h3 className="text-sm font-semibold text-primary">Planner Graph</h3>
                <div className="flex items-center gap-2">
                  <a
                    href={plannerGraphUrl}
                    download="planner-graph.png"
                    className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                  >
                    Download PNG
                  </a>
                  <button
                    type="button"
                    onClick={closePlannerGraph}
                    className="inline-flex items-center rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                  >
                    Close
                  </button>
                </div>
              </div>
              <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-4">
                <img
                  src={plannerGraphUrl}
                  alt="Planner graph for this analysis thread"
                  className="block h-auto max-h-full w-auto max-w-full object-contain"
                />
              </div>
            </div>
          </div>
        ) : null}


        {pasteOpen ? (
          <div
            className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 p-6"
            onClick={() => {
              setPasteOpen(false);
              setPasteText("");
            }}
          >
            <div
              className="flex w-full max-w-xl flex-col overflow-hidden rounded-xl bg-primary shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-secondary px-4 py-3">
                <h3 className="text-sm font-semibold text-primary">Paste analysis data</h3>
                <button
                  type="button"
                  onClick={() => {
                    setPasteOpen(false);
                    setPasteText("");
                  }}
                  className="inline-flex items-center rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                >
                  Close
                </button>
              </div>
              <div className="flex flex-col gap-3 p-4">
                <p className="text-xs text-tertiary">
                  Paste the data you copied from another analysis below, then click Paste.
                </p>
                <textarea
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  placeholder="Paste copied analysis data here…"
                  rows={8}
                  autoFocus
                  className="w-full resize-none rounded-lg border border-secondary bg-primary px-3 py-2 font-mono text-xs text-primary shadow-xs outline-none placeholder:text-placeholder focus:border-brand"
                />
                <div className="flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setPasteOpen(false);
                      setPasteText("");
                    }}
                    className="inline-flex items-center rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={applyPastedAnalysis}
                    disabled={!pasteText.trim()}
                    className="inline-flex items-center rounded-lg bg-brand-solid px-3 py-2 text-sm font-semibold text-white shadow-xs hover:bg-brand-solid_hover disabled:opacity-50"
                  >
                    Paste
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {copyFallbackText !== null ? (
          <div
            className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 p-6"
            onClick={() => setCopyFallbackText(null)}
          >
            <div
              className="flex w-full max-w-xl flex-col overflow-hidden rounded-xl bg-primary shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-secondary px-4 py-3">
                <h3 className="text-sm font-semibold text-primary">Copy analysis data</h3>
                <button
                  type="button"
                  onClick={() => setCopyFallbackText(null)}
                  className="inline-flex items-center rounded-lg border border-secondary bg-primary px-2.5 py-1.5 text-xs font-semibold text-primary shadow-xs hover:bg-primary_hover"
                >
                  Close
                </button>
              </div>
              <div className="flex flex-col gap-3 p-4">
                <p className="text-xs text-tertiary">
                  Your browser blocked automatic clipboard access. Click in the box, press Ctrl+A then Ctrl+C (Cmd+A / Cmd+C on Mac) to copy.
                </p>
                <textarea
                  value={copyFallbackText}
                  readOnly
                  rows={8}
                  autoFocus
                  onFocus={(e) => e.target.select()}
                  className="w-full resize-none rounded-lg border border-secondary bg-primary px-3 py-2 font-mono text-xs text-primary shadow-xs outline-none focus:border-brand"
                />
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => setCopyFallbackText(null)}
                    className="inline-flex items-center rounded-lg bg-brand-solid px-3 py-2 text-sm font-semibold text-white shadow-xs hover:bg-brand-solid_hover"
                  >
                    Done
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : null}

        <AutoInsightsModal
          open={userInsightsOpen}
          onOpenChange={setUserInsightsOpen}
          config={userInsightsViz}
          status="ready"
          datasetName={autoInsightsDatasetName}
          samples={chartSamples}
          outputRows={userInsightsRows}
          defaultView={userInsightsRows?.length ? "table" : "chart"}
          title="User Insights"
          subtitle="Insights generated from your latest prompt"
          aid={aidFromUrl}
        />


        <AddToDashboardModal
          open={addToDashboardOpen}
          onOpenChange={setAddToDashboardOpen}
          projectId={resolvedProjectId}
          analysisId={resolvedAnalysisId}
          analysisTitle={workspaceHeading}
          vizConfig={analysisVizConfig}
          chartIndex={dashboardChartIndex}
          insight={dashboardInsight}
        />


        <HistoryCollaborationDetailModal
          payload={collaborationDetail}
          onClose={() => setCollaborationDetail(null)}
          onOpenSharing={() => setShareCollaboratorsOpen(true)}
        />
      </div>
    </div>
  );
}

const PY_KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
  "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
  "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
]);
const PY_CONSTANTS = new Set(["True", "False", "None", "self", "cls"]);
const PY_BUILTINS = new Set([
  "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len", "list",
  "map", "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "tuple", "type", "zip",
]);

const PY_TOKEN_RE =
  /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)/g;

/** Lightweight VS Code-ish Python highlighter (display only). */
function highlightPython(code: string) {
  const out: React.ReactNode[] = [];
  let last = 0;
  let key = 0;
  const push = (text: string, cls?: string) => {
    if (!text) return;
    out.push(cls ? <span key={key++} className={cls}>{text}</span> : text);
  };
  for (const m of code.matchAll(PY_TOKEN_RE)) {
    const idx = m.index ?? 0;
    push(code.slice(last, idx));
    last = idx + m[0].length;
    if (m[1]) push(m[1], "text-[#008000] italic dark:text-[#6a9955]");
    else if (m[2]) push(m[2], "text-[#a31515] dark:text-[#ce9178]");
    else if (m[3]) push(m[3], "text-[#098658] dark:text-[#b5cea8]");
    else {
      const word = m[4]!;
      const after = code.slice(last);
      if (PY_KEYWORDS.has(word)) push(word, "text-[#0000ff] dark:text-[#569cd6]");
      else if (PY_CONSTANTS.has(word)) push(word, "text-[#0070c1] dark:text-[#4fc1ff]");
      else if (PY_BUILTINS.has(word)) push(word, "text-[#267f99] dark:text-[#4ec9b0]");
      else if (/^\s*\(/.test(after)) push(word, "text-[#795e26] dark:text-[#dcdcaa]");
      else push(word, "text-[#001080] dark:text-[#9cdcfe]");

    }
  }
  push(code.slice(last));
  return out;
}

function InlineCodeEditor({
  loading,
  code,
  originalCode,
  downloadUrl,
  onChange,
  onReset,
  onClose,
  onSaveAndExecute,
  saving,
  plannerNotes,
  validationFeedback,
  onDismissPlanner,
  onDismissValidation,
}: {
  loading: boolean;
  code: string;
  originalCode: string;
  downloadUrl?: string | null;
  onChange: (value: string) => void;
  onReset: () => void;
  onClose: () => void;
  onSaveAndExecute?: () => void | Promise<void>;
  saving?: boolean;
  plannerNotes?: string | null;
  validationFeedback?: string | null;
  onDismissPlanner?: () => void;
  onDismissValidation?: () => void;
}) {
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      toast.success("Copied");
    } catch {
      toast.error("Copy failed");
    }
  };
  const download = () => {
    try {
      const blob = new Blob([code], { type: "text/x-python" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "analysis.py";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch {
      toast.error("Download failed");
    }
  };
  const dirty = code !== originalCode;
  const [expanded, setExpanded] = useState(false);
  const [editing, setEditing] = useState(false);
  return (
    <div className="mt-4 overflow-hidden rounded-2xl border border-[#e4e7ec] bg-white shadow-[0_1px_2px_rgba(16,24,40,0.05)] dark:border-[#1f2937] dark:bg-[#0d1117]">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#e4e7ec] bg-[#fcfcfd] px-4 py-2.5 dark:border-[#1f2937] dark:bg-[#161b22]">
        <div className="flex items-center gap-2">
          <h3 className="text-[13px] font-semibold text-[#344054] dark:text-[#c9d1d9]">Python</h3>
          {dirty && (
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-[#b54708]">
              <span className="inline-block size-1.5 rounded-full bg-[#f79009]" />
              Unsaved changes
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {onSaveAndExecute && (
            <button
              type="button"
              onClick={() => void onSaveAndExecute()}
              disabled={loading || saving || !code}
              className="relative inline-flex items-center gap-1.5 rounded-lg border border-[#1570ef] bg-[#1570ef] px-2.5 py-1 text-[12px] font-semibold text-white shadow-sm hover:bg-[#175cd3] disabled:opacity-60"
            >
              {saving && (
                <span className="size-3 animate-spin rounded-full border-2 border-white/50 border-t-white" />
              )}
              {saving ? "Verifying & Running…" : "Save & Execute"}
              {!saving && dirty && (
                <span className="ml-0.5 inline-block size-1.5 rounded-full bg-white" />
              )}
            </button>
          )}
          <button
            type="button"
            onClick={copy}
            disabled={loading || !code}
            aria-label="Copy code"
            title="Copy"
            className="inline-flex size-7 items-center justify-center rounded-md text-[#667085] hover:bg-[#f2f4f7] hover:text-[#344054] disabled:opacity-50 dark:text-[#8b949e] dark:hover:bg-[#21262d] dark:hover:text-[#c9d1d9]"
          >
            <Copy01 className="size-4" />
          </button>
          <button
            type="button"
            onClick={download}
            disabled={loading || !code}
            className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] font-medium text-[#475467] hover:bg-[#f2f4f7] dark:text-[#8b949e] dark:hover:bg-[#21262d] dark:hover:text-[#c9d1d9] disabled:opacity-50"
          >
            Download
          </button>
          {downloadUrl ? (
            <a
              href={downloadUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] font-medium text-[#475467] hover:bg-[#f2f4f7] dark:text-[#8b949e] dark:hover:bg-[#21262d] dark:hover:text-[#c9d1d9]"
            >
              Original
            </a>
          ) : null}
          <button
            type="button"
            onClick={onReset}
            disabled={loading || !dirty}
            className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] font-medium text-[#475467] hover:bg-[#f2f4f7] dark:text-[#8b949e] dark:hover:bg-[#21262d] dark:hover:text-[#c9d1d9] disabled:opacity-50"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Collapse code"
            className="inline-flex items-center rounded-lg px-2 py-1 text-[12px] font-medium text-[#475467] hover:bg-[#f2f4f7] dark:text-[#8b949e] dark:hover:bg-[#21262d] dark:hover:text-[#c9d1d9]"
          >
            Collapse
          </button>
        </div>
      </div>
      {plannerNotes && (
        <div className="flex items-start justify-between gap-3 border-b border-[#fedf89] bg-[#fffaeb] px-4 py-2 text-[12px] text-[#93370d]">
          <p>
            <span className="font-semibold">The planner corrected your code before running:</span>{" "}
            {plannerNotes}
          </p>
          {onDismissPlanner && (
            <button
              type="button"
              onClick={onDismissPlanner}
              className="shrink-0 rounded px-1.5 py-0.5 text-[11px] font-semibold text-[#93370d] hover:bg-[#fedf89]"
            >
              Dismiss
            </button>
          )}
        </div>
      )}
      {validationFeedback && (
        <div className="flex items-start justify-between gap-3 border-b border-[#fecdca] bg-[#fef3f2] px-4 py-2 text-[12px] text-[#b42318]">
          <pre className="whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed">
            {validationFeedback}
          </pre>
          {onDismissValidation && (
            <button
              type="button"
              onClick={onDismissValidation}
              className="shrink-0 rounded px-1.5 py-0.5 text-[11px] font-semibold text-[#b42318] hover:bg-[#fecdca]"
            >
              Dismiss
            </button>
          )}
        </div>
      )}
      <div className="relative bg-white dark:bg-[#0d1117]">
        {loading ? (
          <p className="p-4 text-sm text-[#667085] dark:text-[#8b949e]">Loading…</p>
        ) : (
          <>
            <div
              className={cx(
                "relative w-full",
                expanded ? "max-h-[640px] overflow-auto" : "overflow-hidden",
              )}
              style={{ height: expanded ? undefined : 320 }}
            >
              <div className="relative flex w-max min-w-full">
                {editing && (
                  <div className="sticky left-0 z-10 shrink-0 select-none border-r border-[#eaecf0] bg-[#fcfcfd] py-5 pl-4 pr-3 text-right font-mono text-[12.5px] leading-[1.7] text-[#98a2b3] dark:border-[#21262d] dark:bg-[#161b22] dark:text-[#6e7681]">
                    {code.split("\n").map((_, i) => (
                      <div key={i}>{i + 1}</div>
                    ))}
                  </div>
                )}
                <div className="relative min-w-0 flex-1">
                  <pre
                    aria-hidden
                    className="pointer-events-none m-0 whitespace-pre p-5 font-mono text-[12.5px] leading-[1.7] text-[#1d2939] dark:text-[#d4d4d4]"
                  >
                    {highlightPython(code)}
                    {"\n"}
                  </pre>
                  <textarea
                    value={code}
                    onChange={(e) => onChange(e.target.value)}
                    onFocus={() => setEditing(true)}
                    onBlur={() => setEditing(false)}
                    spellCheck={false}
                    wrap="off"
                    className="absolute inset-0 h-full w-full resize-none overflow-hidden whitespace-pre bg-transparent p-5 font-mono text-[12.5px] leading-[1.7] text-transparent caret-[#1d2939] outline-none dark:caret-[#d4d4d4]"
                  />
                </div>
              </div>

            </div>
            {!expanded && (
              <div className="pointer-events-none absolute inset-x-0 bottom-0 flex h-24 items-end justify-center bg-gradient-to-t from-white via-white/90 to-transparent pb-3 dark:from-[#0d1117] dark:via-[#0d1117]/90">
                <button
                  type="button"
                  onClick={() => setExpanded(true)}
                  className="pointer-events-auto rounded-lg border border-[#d0d5dd] bg-white px-3.5 py-2 text-[13px] font-semibold text-[#344054] shadow-sm hover:bg-[#f9fafb] dark:border-[#30363d] dark:bg-[#161b22] dark:text-[#c9d1d9] dark:hover:bg-[#21262d]"
                >
                  Show more
                </button>
              </div>
            )}
            {expanded && (
              <div className="flex justify-center border-t border-[#f2f4f7] py-2 dark:border-[#21262d]">
                <button
                  type="button"
                  onClick={() => setExpanded(false)}
                  className="rounded-lg border border-[#d0d5dd] bg-white px-3.5 py-1.5 text-[13px] font-semibold text-[#344054] shadow-sm hover:bg-[#f9fafb] dark:border-[#30363d] dark:bg-[#161b22] dark:text-[#c9d1d9] dark:hover:bg-[#21262d]"
                >
                  Show less
                </button>
              </div>
            )}
          </>
        )}
      </div>


    </div>
  );
}

function AnalysisSkeleton() {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-secondary bg-primary px-5 py-3">
        <div className="size-5 animate-pulse rounded bg-secondary" />
        <div className="h-4 w-40 animate-pulse rounded bg-secondary" />
        <div className="ml-auto flex items-center gap-2">
          <div className="h-8 w-24 animate-pulse rounded-md bg-secondary" />
          <div className="h-8 w-24 animate-pulse rounded-md bg-secondary" />
        </div>
      </div>
      {/* Body */}
      <div className="flex flex-1 flex-col gap-4 overflow-hidden p-6">
        <div className="flex gap-3">
          <div className="h-6 w-32 animate-pulse rounded bg-secondary" />
          <div className="h-6 w-24 animate-pulse rounded bg-secondary" />
        </div>
        <div className="h-8 w-2/3 animate-pulse rounded bg-secondary" />
        <div className="h-4 w-1/2 animate-pulse rounded bg-secondary" />
        <div className="mt-2 flex-1 animate-pulse rounded-xl bg-secondary" />
        <div className="grid grid-cols-3 gap-3">
          <div className="h-20 animate-pulse rounded-lg bg-secondary" />
          <div className="h-20 animate-pulse rounded-lg bg-secondary" />
          <div className="h-20 animate-pulse rounded-lg bg-secondary" />
        </div>
      </div>
    </div>
  );
}

function ShareChatModal({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[460px] gap-0 rounded-2xl border-secondary bg-primary p-6 shadow-xl">
        <div className="flex size-11 items-center justify-center rounded-lg border border-secondary bg-primary shadow-xs">
          <MessageChatSquare className="size-5 text-fg-secondary" />
        </div>

        <DialogTitle className="mt-4 text-lg font-semibold text-primary">Share Chat</DialogTitle>

        <div className="mt-4 flex items-start gap-2 rounded-lg border border-[#fedf89] bg-[#fffaeb] px-3 py-2.5">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-[#dc6803]" />
          <p className="text-xs text-[#b54708]">
            This conversation may include personal information. Take a moment to check the content before sharing the
            link.
          </p>
        </div>

        <h3 className="mt-5 text-sm font-semibold text-primary">Chat Name</h3>
        <p className="mt-3 text-sm font-medium text-secondary">Preview</p>
        <div className="mt-1.5 rounded-lg border border-secondary bg-primary px-3.5 py-2.5">
          <p className="text-sm text-secondary">
            Here are the top 5 branches based on revenue for the most recent quarter. These branches contribute 62% of
            total revenue, with Branch A leading by a clear margin.
          </p>
        </div>

        <p className="mt-5 text-sm font-medium text-secondary">Select User to share</p>
        <UserSelect />

        <div className="mt-6 grid grid-cols-2 gap-3">
          <button
            onClick={() => onOpenChange(false)}
            className="rounded-lg border border-secondary bg-primary px-4 py-2.5 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
          >
            Copy Link
          </button>
          <button
            onClick={() => onOpenChange(false)}
            className="rounded-lg bg-[#1565ef] px-4 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-[#1257d6]"
          >
            Share
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

const DUMMY_USERS = [
  { id: "u1", name: "Olivia Rhye", email: "olivia@untitled.com" },
  { id: "u2", name: "Phoenix Baker", email: "phoenix@untitled.com" },
  { id: "u3", name: "Lana Steiner", email: "lana@untitled.com" },
  { id: "u4", name: "Demi Wilkinson", email: "demi@untitled.com" },
  { id: "u5", name: "Candice Wu", email: "candice@untitled.com" },
];

function UserSelect() {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<(typeof DUMMY_USERS)[number] | null>(null);
  return (
    <div className="relative mt-1.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between rounded-lg border border-secondary bg-primary px-3.5 py-2.5 text-left shadow-xs"
      >
        <span className="inline-flex items-center gap-2 text-sm">
          <User01 className="size-4 text-fg-quaternary" />
          {selected ? (
            <span className="text-primary">{selected.name}</span>
          ) : (
            <span className="text-tertiary">Select User</span>
          )}
        </span>
        <ChevronDown className={cx("size-4 text-fg-quaternary transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="absolute z-20 mt-1 max-h-60 w-full overflow-auto rounded-lg border border-secondary bg-primary py-1 shadow-lg">
          {DUMMY_USERS.map((u) => (
            <button
              key={u.id}
              type="button"
              onClick={() => {
                setSelected(u);
                setOpen(false);
              }}
              className="flex w-full items-center gap-2.5 px-3 py-2 text-left hover:bg-secondary"
            >
              <span className="grid size-8 place-items-center rounded-full bg-[#1565ef]/10 text-xs font-semibold text-[#1565ef]">
                {u.name
                  .split(" ")
                  .map((n) => n[0])
                  .join("")}
              </span>
              <span className="flex flex-col">
                <span className="text-sm font-medium text-primary">{u.name}</span>
                <span className="text-xs text-tertiary">{u.email}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

type MoveProjectOption = { id: string; name: string; desc: string };

function MoveProjectModal({
  open,
  onOpenChange,
  onMoved,
  analysisId,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  onMoved: (projectName: string) => void;
  analysisId: string | null;
}) {
  const { data: projectsData, isLoading } = useProjects();
  const { moveToProject } = useAnalysisMutations(null);
  const projects: MoveProjectOption[] = useMemo(
    () =>
      (projectsData ?? [])
        .filter((p) => !p.archived)
        .map((p) => ({ id: p.id, name: p.name, desc: p.pinned ? "Pinned" : p.bookmarked ? "Bookmarked" : "Project" })),
    [projectsData],
  );
  const [selectedId, setSelectedId] = useState<string>("");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [dropdownChoice, setDropdownChoice] = useState<MoveProjectOption | null>(null);

  useEffect(() => {
    if (!selectedId && projects.length > 0) setSelectedId(projects[0].id);
  }, [projects, selectedId]);

  const handleMove = async () => {
    const chosen = dropdownChoice ?? projects.find((p) => p.id === selectedId) ?? projects[0];
    if (!chosen || !analysisId) return;
    try {
      await moveToProject.mutateAsync({ id: analysisId, project_id: chosen.id });
      onMoved(chosen.name);
    } catch (e) {
      console.error("move to project failed", e);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[460px] gap-0 rounded-2xl border-secondary bg-primary p-6 shadow-xl">
        <div className="flex size-11 items-center justify-center rounded-lg bg-[#1565ef] shadow-xs">
          <Folder className="size-5 text-white" />
        </div>

        <DialogTitle className="mt-4 text-lg font-semibold text-primary">Move Project</DialogTitle>
        <p className="mt-1 text-sm text-tertiary">Move this chat to any of your projects</p>

        <p className="mt-5 text-sm font-medium text-secondary">Project</p>
        <div className="relative mt-1.5">
          <button
            type="button"
            onClick={() => setDropdownOpen((v) => !v)}
            className="flex w-full items-center justify-between rounded-lg border border-secondary bg-primary px-3.5 py-2.5 text-left shadow-xs"
          >
            <span className={cx("text-sm", dropdownChoice ? "text-primary" : "text-tertiary")}>
              {dropdownChoice ? dropdownChoice.name : "Select Project"}
            </span>
            <ChevronDown className={cx("size-4 text-fg-quaternary transition", dropdownOpen && "rotate-180")} />
          </button>
          {dropdownOpen && (
            <div className="absolute z-20 mt-1 max-h-56 w-full overflow-auto rounded-lg border border-secondary bg-primary py-1 shadow-lg">
              {isLoading && <div className="px-3 py-2 text-xs text-tertiary">Loading…</div>}
              {!isLoading && projects.length === 0 && (
                <div className="px-3 py-2 text-xs text-tertiary">No projects yet</div>
              )}
              {projects.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => {
                    setDropdownChoice(p);
                    setDropdownOpen(false);
                  }}
                  className="flex w-full flex-col px-3 py-2 text-left hover:bg-secondary"
                >
                  <span className="text-sm font-medium text-primary">{p.name}</span>
                  <span className="text-xs text-tertiary">{p.desc}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="mt-5 border-t border-secondary pt-4">
          <h3 className="text-sm font-semibold text-primary">Recent projects</h3>
          <div className="mt-3 flex flex-col gap-2.5">
            {projects.slice(0, 3).map((p) => {
              const active = selectedId === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setSelectedId(p.id)}
                  className={cx(
                    "flex w-full items-center gap-3 rounded-lg border bg-primary px-3.5 py-3 text-left transition",
                    active ? "border-[#1565ef] ring-1 ring-[#1565ef]" : "border-secondary hover:bg-primary_hover",
                  )}
                >
                  <span className="grid size-9 place-items-center rounded-md border border-secondary bg-primary">
                    <File02 className="size-4 text-fg-secondary" />
                  </span>
                  <span className="flex flex-1 flex-col">
                    <span className="text-sm font-semibold text-primary">{p.name}</span>
                    <span className="text-xs text-tertiary">{p.desc}</span>
                  </span>
                  <span
                    className={cx(
                      "grid size-5 place-items-center rounded-full border-2",
                      active ? "border-[#1565ef] bg-[#1565ef]" : "border-secondary bg-primary",
                    )}
                  >
                    {active && <span className="size-1.5 rounded-full bg-white" />}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-3">
          <button
            onClick={() => onOpenChange(false)}
            className="rounded-lg border border-secondary bg-primary px-4 py-2.5 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
          >
            Cancel
          </button>
          <button
            onClick={handleMove}
            className="rounded-lg bg-[#1565ef] px-4 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-[#1257d6]"
          >
            Move
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function VoicePanel({
  onClose,
  messages,
  isSending,
  listening,
}: {
  onClose: () => void;
  messages: ChatMessage[];
  isSending: boolean;
  listening: boolean;
}) {
  const [userName, setUserName] = useState("");
  const [spokenText, setSpokenText] = useState("");
  const [charIndex, setCharIndex] = useState(-1);
  const [speaking, setSpeaking] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // Ignore whatever reply already existed when voice mode opened: only replies
  // that arrive while the avatar is open are shown/spoken below it.
  const spokenMessageIdRef = useRef<string | null | undefined>(undefined);
  if (spokenMessageIdRef.current === undefined) {
    spokenMessageIdRef.current =
      [...messages].reverse().find((m) => m.role === "ai" && !m.thinking && !m.error)?.id ?? null;
  }

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const profileId = await getCurrentProfileId();
        if (!profileId) return;
        const { data } = await supabase
          .from("profiles")
          .select("full_name")
          .eq("id", profileId)
          .maybeSingle();
        if (!cancelled) setUserName((data as any)?.full_name ?? "");
      } catch {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Read the assistant's newest reply aloud and highlight along with it.
  useEffect(() => {
    const last = [...messages].reverse().find((m) => m.role === "ai" && !m.thinking && !m.error);
    if (!last || last.id === spokenMessageIdRef.current) return;
    spokenMessageIdRef.current = last.id;
    const clean = speakText(stripTabularContent(last.content ?? ""), {
      onStart: () => setSpeaking(true),
      onBoundary: (index) => setCharIndex(index),
      onEnd: () => {
        setCharIndex(-1);
        setSpeaking(false);
      },
    });
    setCharIndex(-1);
    setSpokenText(clean);
  }, [messages]);

  useEffect(() => () => stopSpeaking(), []);

  // Drive the avatar video: loop while TTS is speaking, pause + show the
  // static avatar image when speech stops.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (speaking) {
      video.play().catch(() => {
        /* autoplay may be blocked until a user gesture */
      });
    } else {
      video.pause();
      try {
        video.currentTime = 0;
      } catch {
        /* ignore */
      }
    }
  }, [speaking]);

  const isThinking = isSending && !listening;

  // Split the reply into word tokens so the spoken word can be highlighted.
  const tokens = useMemo(() => {
    const out: { text: string; start: number }[] = [];
    const regex = /\S+\s*/g;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(spokenText))) out.push({ text: match[0], start: match.index });
    return out;
  }, [spokenText]);

  const activeToken = useMemo(() => {
    if (charIndex < 0) return -1;
    let index = -1;
    for (let i = 0; i < tokens.length; i += 1) {
      if (tokens[i]!.start <= charIndex) index = i;
      else break;
    }
    return index;
  }, [tokens, charIndex]);

  return (
    <div className="relative flex flex-1 flex-col items-center overflow-y-auto px-6 py-8 text-center">
      <button
        type="button"
        aria-label="Exit voice mode"
        onClick={onClose}
        className="absolute right-3 top-3 rounded-lg p-1.5 text-fg-quaternary transition hover:bg-primary_hover hover:text-fg-secondary"
      >
        <X className="size-4" />
      </button>

      <div className="my-auto flex w-full flex-col items-center gap-5">
        <div className="relative flex items-center justify-center">
          {listening && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <span className="absolute size-28 rounded-full bg-brand-600/25 animate-[voice-listening-ring_1.8s_ease-out_infinite]" />
              <span className="absolute size-28 rounded-full bg-brand-600/20 animate-[voice-listening-ring_1.8s_ease-out_infinite_0.6s]" />
              <span className="absolute size-28 rounded-full bg-brand-600/15 animate-[voice-listening-ring_1.8s_ease-out_infinite_1.2s]" />
            </div>
          )}
          {isThinking && (
            <span className="pointer-events-none absolute size-32 rounded-full bg-brand-600/20 animate-[voice-thinking-pulse_1.5s_ease-in-out_infinite]" />
          )}
          <div className="relative z-10 size-28 overflow-hidden rounded-full">
            <img
              src={avatarTtsIdle}
              alt="Avaloka AI"
              className={cx("absolute inset-0 size-full object-cover transition-opacity", speaking ? "opacity-0" : "opacity-100")}
            />
            <video
              ref={videoRef}
              src={avatarTtsVideo}
              loop
              muted
              playsInline
              preload="auto"
              aria-hidden
              style={{ top: "-14%", height: "auto", width: "92%", left: "4%" }}
              className={cx("absolute object-cover transition-opacity", speaking ? "opacity-100" : "opacity-0")}
            />

          </div>
        </div>

        <div className="space-y-1">
          <p className="text-lg font-semibold text-primary">
            {userName ? `Welcome, ${userName}` : "Welcome"}
          </p>
          <p className="text-sm text-tertiary">
            {isThinking
              ? "Thinking..."
              : listening
                ? "Listening... tap stop when you're done."
                : "Tap the mic below and start speaking."}
          </p>
        </div>

        {isThinking ? (
          <div className="flex items-center gap-1.5 text-sm text-tertiary">
            <span className="size-2 animate-bounce rounded-full bg-brand-600 [animation-delay:-0.3s]" />
            <span className="size-2 animate-bounce rounded-full bg-brand-600 [animation-delay:-0.15s]" />
            <span className="size-2 animate-bounce rounded-full bg-brand-600" />
          </div>
        ) : (
          spokenText && (
            <p className="max-w-[92%] text-left text-sm leading-relaxed text-secondary">
              {tokens.map((token, index) => (
                <span
                  key={`${token.start}-${index}`}
                  className={cx(
                    "transition-colors",
                    index === activeToken ? "rounded bg-brand-600/15 font-bold text-brand-700" : "",
                  )}
                >
                  {token.text}
                </span>
              ))}
            </p>
          )
        )}
      </div>
    </div>
  );
}


function ChatComposer({
  isSending,
  avatarMode,
  onAvatarModeChange,
  onSend,
  onTyping,
  onStopTyping,
  onListeningChange,
}: {
  isSending: boolean;
  avatarMode: boolean;
  onAvatarModeChange: () => void;
  onSend: (content: string) => Promise<boolean>;
  onTyping?: () => void;
  onStopTyping?: () => void;
  onListeningChange?: (listening: boolean) => void;
}) {
  const [draft, setDraft] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const disabled = isSending || isSubmitting;

  const sendText = async (raw: string) => {
    const content = raw.trim();
    if (!content || disabled) return;

    setDraft("");
    onStopTyping?.();
    setIsSubmitting(true);
    try {
      const accepted = await onSend(content);
      if (!accepted) setDraft((current) => current || content);
    } finally {
      setIsSubmitting(false);

    }
  };

  const submitDraft = () => sendText(draft);

  const voice = useVoiceInput((text) => {
    void sendText(text);
  });

  useEffect(() => {
    onListeningChange?.(voice.listening);
  }, [voice.listening, onListeningChange]);

  useEffect(() => {
    if (voice.listening) setDraft(voice.transcript);
  }, [voice.listening, voice.transcript]);

  const holdStart = () => {
    if (!voice.supported) {
      toast.error("Voice input isn't supported in this browser.");
      return;
    }
    stopSpeaking();
    voice.start();
  };

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void submitDraft();
      }}
      className="rounded-xl border border-secondary bg-primary p-3 shadow-xs"
    >
      <div className="flex items-start gap-2">
        <img
          src={avatarBot}
          alt=""
          aria-hidden
          width={28}
          height={28}
          loading="lazy"
          className="mt-0.5 size-7 shrink-0 rounded-full object-contain"
        />
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
            if (event.target.value.trim()) onTyping?.();
            else onStopTyping?.();
          }}
          onBlur={() => onStopTyping?.()}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              void submitDraft();
            }
          }}
          placeholder={voice.listening ? "Listening..." : "Ask me anything... "}
          rows={1}
          className="field-sizing-content min-h-5 max-h-[240px] flex-1 resize-none overflow-y-auto bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
        />
        {avatarMode ? (
          <button
            type="button"
            aria-label={voice.listening ? "Stop listening" : "Start listening"}
            title={voice.listening ? "Stop listening" : "Start listening"}
            onClick={() => {
              if (voice.listening) voice.stop();
              else holdStart();
            }}
            className={cx(
              "mt-0.5 shrink-0 rounded-full p-1 transition",
              voice.listening ? "bg-[#1565ef] text-white ring-4 ring-[#1565ef]/20" : "text-[#1565ef] hover:bg-[#1565ef]/10",
            )}
          >
            {voice.listening ? <StopListeningIcon className="size-5" /> : <MicIcon className="size-5" />}
          </button>
        ) : (

          <button
            type="button"
            aria-label="Voice input"
            onClick={onAvatarModeChange}
            className="mt-0.5 shrink-0 text-fg-quaternary hover:text-fg-secondary"
          >
            <AudioWaveIcon className="size-5" />
          </button>
        )}
      </div>

      <div className="mt-1 flex justify-end">
        <button
          type="submit"
          disabled={disabled || !draft.trim()}
          className={cx(
            "text-sm font-semibold",
            disabled || !draft.trim() ? "cursor-not-allowed text-tertiary" : "text-[#1565ef] hover:underline",
          )}
        >
          {disabled ? "Sending..." : "Send"}
        </button>
      </div>
    </form>
  );
}

function MessageRow({
  message,
  onReply,
  onSectionSelect,
  activeSectionId,
  onRefresh,
  onCopy,
  onThumb,
  feedback,
  feedbackSubmitting,
  refreshing,
}: {
  message: ChatMessage;
  onReply?: () => void;
  onSectionSelect?: (sectionId: string) => void;
  activeSectionId?: string;
  onRefresh?: () => void;
  onCopy?: () => void;
  onThumb?: (type: "positive" | "negative") => void;
  feedback?: "positive" | "negative" | null;
  feedbackSubmitting?: "positive" | "negative" | null;
  refreshing?: boolean;
}) {
  const isAi = message.role === "ai";
  const [dragX, setDragX] = useState(0);
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const triggeredRef = useRef(false);

  const handlePointerDown = (e: ReactPointerEvent) => {
    if (!onReply || message.thinking) return;
    startRef.current = { x: e.clientX, y: e.clientY };
    triggeredRef.current = false;
  };
  const handlePointerMove = (e: ReactPointerEvent) => {
    if (!startRef.current) return;
    const dx = e.clientX - startRef.current.x;
    const dy = e.clientY - startRef.current.y;
    if (Math.abs(dx) > Math.abs(dy)) {
      const clamped = Math.max(-90, Math.min(90, dx));
      setDragX(clamped);
      if (!triggeredRef.current && Math.abs(clamped) > 55) {
        triggeredRef.current = true;
        onReply?.();
      }
    }
  };
  const handlePointerUp = () => {
    startRef.current = null;
    setDragX(0);
  };

  const swipeProps = {
    onPointerDown: handlePointerDown,
    onPointerMove: handlePointerMove,
    onPointerUp: handlePointerUp,
    onPointerCancel: handlePointerUp,
    style: {
      transform: `translateX(${dragX}px)`,
      transition: dragX === 0 ? "transform 0.18s ease-out" : "none",
      touchAction: "pan-y" as const,
    },
  };

  if (message.thinking) {
    return (
      <div className="mb-4 flex items-start gap-2">
        <img
          src={avatarBot}
          alt=""
          aria-hidden
          width={28}
          height={28}
          className="size-7 shrink-0 rounded-full object-contain"
        />
        <div className="inline-flex items-center gap-2 rounded-md border border-secondary bg-primary px-2.5 py-1.5">
          <span className="text-xs text-tertiary">Thinking</span>
          <div className="h-1 w-24 overflow-hidden rounded-full bg-[#e4e4e7]">
            <div className="h-full w-2/3 animate-pulse rounded-full bg-[#1565ef]" />
          </div>
        </div>
      </div>
    );
  }

  if (isAi) {
    return (
      <div className="relative mb-5">
        {dragX !== 0 && (
          <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-[#1565ef]">
            <MessageChatCircle className="size-4" />
          </div>
        )}
        <div {...swipeProps} className="flex items-start gap-2">
          <img
            src={avatarBot}
            alt="Avaloka AI"
            width={28}
            height={28}
            className="mt-0.5 size-7 shrink-0 rounded-full object-cover"
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-primary">Avaloka AI</span>
              <span className="text-[11px] text-tertiary">{message.time}</span>
            </div>
            <div
              className={cx(
                "mt-1 rounded-lg border px-3 py-2 text-sm cursor-pointer",
                message.error
                  ? "border-[#fda29b] bg-[#fef3f2] text-[#b42318]"
                  : "border-secondary bg-primary text-primary hover:border-[#b2ddff]",
              )}
              onDoubleClick={() => onReply?.()}
            >
              {message.replyTo && (
                <div className="mb-1.5 rounded-md border-l-[3px] border-[#1565ef] bg-secondary/50 px-2 py-1">
                  <p className="text-[11px] font-semibold text-[#1565ef]">{message.replyTo.label}</p>
                  <p className="truncate text-xs text-secondary">{message.replyTo.content}</p>
                </div>
              )}
              <MarkdownContent content={message.content} />
            </div>
            <div className="mt-2 flex items-center gap-3 text-fg-quaternary">
              <button aria-label="Edit" className="hover:text-fg-secondary">
                <Edit01Icon className="size-3.5" />
              </button>
              <button
                aria-label="Refresh"
                onClick={onRefresh}
                disabled={!!refreshing}
                className="hover:text-fg-secondary disabled:opacity-50"
              >
                <Repeat02 className={cx("size-3.5", refreshing && "animate-spin")} />
              </button>
              <button aria-label="Copy" onClick={onCopy} className="hover:text-fg-secondary">
                <Copy01 className="size-3.5" />
              </button>
              <button
                aria-label="Good Response"
                aria-pressed={feedback === "positive"}
                onClick={() => onThumb?.("positive")}
                disabled={feedbackSubmitting !== null && feedbackSubmitting !== undefined}
                className={cx(
                  "hover:text-fg-secondary disabled:opacity-50",
                  feedback === "positive" && "text-[#1565ef]",
                )}
              >
                <ThumbsUp className="size-3.5" />
              </button>
              <button
                aria-label="Bad Response"
                aria-pressed={feedback === "negative"}
                onClick={() => onThumb?.("negative")}
                disabled={feedbackSubmitting !== null && feedbackSubmitting !== undefined}
                className={cx(
                  "hover:text-fg-secondary disabled:opacity-50",
                  feedback === "negative" && "text-[#d92d20]",
                )}
              >
                <ThumbsDown className="size-3.5" />
              </button>

              <button
                type="button"
                aria-label="Reply"
                onClick={() => onReply?.()}
                className="ml-auto text-[11px] font-semibold text-[#1565ef] hover:underline"
              >
                Reply
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // user
  return (
    <div className="relative mb-5">
      {dragX !== 0 && (
        <div className="pointer-events-none absolute inset-y-0 left-2 flex items-center text-[#1565ef]">
          <MessageChatCircle className="size-4" />
        </div>
      )}
      <div {...swipeProps} className="flex flex-col items-end">
        <div className="flex items-center gap-1.5">
          {!message.isSelf && message.authorAvatar ? (
            <img
              src={message.authorAvatar}
              alt=""
              width={20}
              height={20}
              className="size-5 rounded-full object-cover"
            />
          ) : !message.isSelf ? (
            <span className="grid size-5 place-items-center rounded-full bg-[#1565ef]/10 text-[9px] font-semibold text-[#1565ef]">
              {(message.authorName ?? "C").slice(0, 1).toUpperCase()}
            </span>
          ) : null}
          <span className="text-xs font-semibold text-primary">
            {message.isSelf === false ? (message.authorName ?? "Collaborator") : "You"}
          </span>
          <span className="text-[11px] text-tertiary">{message.time}</span>
          <CheckCheckIcon className="size-3.5 text-[#1565ef]" />
        </div>
        <div
          onDoubleClick={() => onReply?.()}
          onClick={() => {
            if (message.sectionId) onSectionSelect?.(message.sectionId);
          }}
          className={cx(
            "mt-1 max-w-[85%] cursor-pointer rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary hover:border-[#b2ddff]",
            message.sectionId && activeSectionId === message.sectionId && "border-[#1565ef] ring-2 ring-[#1565ef]/20",
          )}
        >
          {message.replyTo && (
            <div className="mb-1.5 rounded-md border-l-[3px] border-[#1565ef] bg-secondary/50 px-2 py-1 text-left">
              <p className="text-[11px] font-semibold text-[#1565ef]">{message.replyTo.label}</p>
              <p className="truncate text-xs text-secondary">{message.replyTo.content}</p>
            </div>
          )}
          {message.content}
        </div>
        <div className="mt-1.5 flex items-center gap-3 text-fg-quaternary">
          <button aria-label="Copy" onClick={onCopy} className="hover:text-fg-secondary">
            <Copy01 className="size-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="prose-sm max-w-none break-words text-sm leading-relaxed [&_a]:text-[#1565ef] [&_a]:underline [&_code]:rounded [&_code]:bg-secondary/60 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[12px] [&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-[#0f172a] [&_pre]:p-3 [&_pre]:text-[12px] [&_pre]:text-white [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-white [&_h1]:mt-2 [&_h1]:mb-1 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:mt-2 [&_h2]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h3]:mt-2 [&_h3]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold [&_p]:my-1 [&_ul]:my-1 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5 [&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-secondary [&_blockquote]:pl-3 [&_blockquote]:text-secondary [&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-secondary [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_td]:border [&_td]:border-secondary [&_td]:px-2 [&_td]:py-1 [&_hr]:my-2 [&_hr]:border-secondary [&_strong]:font-semibold">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function InlineMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <>{children}</>,
        a: ({ children, href }) => (
          <a href={href} className="text-[#1565ef] underline" target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
        code: ({ children }) => <code className="rounded bg-secondary/60 px-1 py-0.5 text-[12px]">{children}</code>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function workspaceChartCardKey(keyPrefix: string, slide: Slide, index: number) {
  return `${keyPrefix}::${slideFingerprint(slide)}::${index}`;
}

function WorkspaceChartCards({
  slides,
  keyPrefix,
  dismissedKeys,
  onDismiss,
  activeRunId,
  onRunChange,
}: {
  slides: RunSlide[];
  keyPrefix: string;
  dismissedKeys?: Set<string>;
  onDismiss?: (cardKey: string) => void;
  activeRunId?: string | null;
  onRunChange?: (runId: string) => void;
}) {
  const visible = useMemo(
    () =>
      dedupeSlidesByInsight(slides)
        .map((s, i) => ({
          slide: s,
          index: i,
          cardKey: workspaceChartCardKey(keyPrefix, s, i),
          runId: (s as RunSlide).runId,
        }))
        .filter(({ cardKey }) => !dismissedKeys?.has(cardKey)),
    [slides, keyPrefix, dismissedKeys],
  );
  if (!visible.length) return null;
  return (
    <ChartCarousel items={visible} onDismiss={onDismiss} activeRunId={activeRunId} onRunChange={onRunChange} />
  );
}

function ChartCarousel({
  items,
  onDismiss,
  activeRunId,
  onRunChange,
}: {
  items: { slide: Slide; index: number; cardKey: string; runId?: string }[];
  onDismiss?: (cardKey: string) => void;
  activeRunId?: string | null;
  onRunChange?: (runId: string) => void;
}) {
  const [active, setActive] = useState(0);
  const total = items.length;
  const safeActive = Math.min(active, total - 1);
  const runIdsKey = items.map((item) => item.runId ?? "").join("\u001f");
  const selectIndex = useCallback(
    (i: number) => {
      setActive((current) => (current === i ? current : i));
      const rid = items[i]?.runId;
      if (rid && rid !== activeRunId) onRunChange?.(rid);
    },
    [items, activeRunId, onRunChange],
  );
  const go = (dir: -1 | 1) => selectIndex((safeActive + dir + total) % total);
  const activeItem = items[safeActive]!;
  const s = activeItem.slide;
  const cardKey = activeItem.cardKey;

  // Keep the carousel on the run selected elsewhere (e.g. the Data table pager),
  // without fighting the user when they page within the same run's charts.
  useEffect(() => {
    if (!activeRunId) return;
    if (items[safeActive]?.runId === activeRunId) return;
    const i = items.findIndex((it) => it.runId === activeRunId);
    if (i >= 0) setActive((current) => (current === i ? current : i));
  }, [activeRunId, runIdsKey, safeActive]);



  // Offset from center: -1 = prev, 0 = active, 1 = next. Only render 3 neighbours.
  const getOffset = (i: number) => {
    if (total <= 1) return 0;
    let d = i - safeActive;
    if (d > total / 2) d -= total;
    if (d < -total / 2) d += total;
    return d;
  };

  const CHART_H = 320;
  const STAGE_H = CHART_H + 140; // header + chart + padding

  return (
    <div className="space-y-3 rounded-2xl bg-secondary/50 p-4">
      <div className="relative">

        {total > 1 ? (
          <button
            type="button"
            aria-label="Previous chart"
            onClick={() => go(-1)}
            className="absolute left-2 top-1/2 z-30 -translate-y-1/2 rounded-full bg-brand-600 p-2.5 text-white shadow-md hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            <ChevronLeft className="size-5" strokeWidth={2.5} />
          </button>
        ) : null}
        {total > 1 ? (
          <button
            type="button"
            aria-label="Next chart"
            onClick={() => go(1)}
            className="absolute right-2 top-1/2 z-30 -translate-y-1/2 rounded-full bg-brand-600 p-2.5 text-white shadow-md hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            <ChevronRight className="size-5" strokeWidth={2.5} />
          </button>
        ) : null}

        <div
          className="relative mx-auto w-full overflow-hidden"
          style={{ height: STAGE_H, perspective: "1600px" }}
        >
          {items.map((it, i) => {
            const off = getOffset(i);
            if (Math.abs(off) > 1) return null; // only render 3 nearest
            const isActive = off === 0;
            const translateX = off * 52; // percent
            const translateY = isActive ? 0 : -10;
            const scale = isActive ? 1 : 0.86;
            const rotY = off * -10; // subtle 3D tilt toward center
            const opacity = isActive ? 1 : 0.55;
            const z = isActive ? 20 : 10;
            const filter = isActive ? "none" : "grayscale(1) blur(0.3px)";

            return (
              <div
                key={it.cardKey}
                aria-hidden={!isActive}
                onClick={() => {
                  if (!isActive) selectIndex(i);
                }}
                className="absolute left-1/2 top-0 w-[88%] will-change-transform"
                style={{
                  transform: isActive
                    ? `translate3d(-50%, 0, 0)`
                    : `translate3d(calc(-50% + ${translateX}%), ${translateY}px, 0) scale(${scale}) rotateY(${rotY}deg)`,
                  transformStyle: "preserve-3d",
                  opacity,
                  zIndex: z,
                  filter,
                  transition:
                    "transform 550ms cubic-bezier(0.22, 1, 0.36, 1), opacity 400ms ease, filter 400ms ease",
                  pointerEvents: isActive ? "auto" : "auto",
                  cursor: isActive ? "default" : "pointer",
                }}

              >
                <div className={cx(
                  "relative rounded-xl p-4",
                  isActive
                    ? "border border-secondary bg-primary shadow-sm"
                    : "border border-primary bg-muted/90"
                )}>
                  {isActive && onDismiss ? (
                    <button
                      type="button"
                      aria-label="Close chart"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDismiss(cardKey);
                      }}
                      className="absolute right-3 top-3 z-10 rounded-md p-1 text-fg-quaternary hover:bg-primary_hover hover:text-fg-secondary"
                    >
                      <X className="size-4" />
                    </button>
                  ) : null}
                  <div className="flex flex-wrap items-center gap-2 pr-8">
                    <h3 className="text-sm font-semibold text-primary">{it.slide.title}</h3>
                    {isActive ? <TagPicker size="sm" /> : null}
                    {isActive && total > 1 ? (
                      <span className="ml-auto text-xs text-tertiary">
                        {safeActive + 1} / {total}
                      </span>
                    ) : null}
                  </div>
                  {isActive && it.slide.subtitle ? (
                    <p className="text-xs text-tertiary">
                      <InlineMarkdown content={it.slide.subtitle} />
                    </p>
                  ) : null}
                  <div className="mt-3 h-[320px] w-full min-w-0 shrink-0">
                    <DynamicChart slide={it.slide} height={CHART_H} chartKey={it.cardKey} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {s.insights.length > 0 ? (
          <div className="mx-auto mt-4 w-[86%]">
            <CollapsibleInsights className="rounded-xl border border-secondary bg-primary p-4">
              <ul className="list-disc space-y-1.5 pl-5 text-sm text-secondary">
                {s.insights.map((insight, j) => (
                  <li key={j}>
                    <InlineMarkdown content={insight} />
                  </li>
                ))}
              </ul>
            </CollapsibleInsights>
          </div>
        ) : null}
      </div>
      {total > 1 ? (
        <div className="flex items-center justify-center gap-1.5">
          {items.map((it, i) => (
            <button
              key={it.cardKey}
              type="button"
              aria-label={`Go to chart ${i + 1}`}
              onClick={() => selectIndex(i)}
              className={
                i === safeActive
                  ? "h-2 w-6 rounded-full bg-brand-solid transition-all"
                  : "h-2 w-2 rounded-full bg-quaternary hover:bg-tertiary transition-all"
              }
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function CollapsibleInsights({
  title = "Key Insights",
  defaultOpen = true,
  className = "mt-6 rounded-xl border border-secondary bg-primary p-4",
  children,
}: {
  title?: string;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={className}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <h3 className="text-sm font-semibold text-primary">{title}</h3>
        <ChevronDown
          className={`size-4 text-fg-secondary transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open ? <div className="mt-3">{children}</div> : null}
    </div>
  );
}


function WorkspaceSectionUserInsights({
  sectionId,
  entries,
  activeTab,
  chartSamples,
  excludeTitles,
  dismissedKeys,
  onDismiss,
  activeRunId,
  onRunChange,
}: {
  sectionId: string;
  entries: UserInsightEntry[];
  activeTab: DashboardTab | undefined;
  chartSamples: unknown[] | undefined;
  excludeTitles?: Set<string>;
  dismissedKeys?: Set<string>;
  onDismiss?: (cardKey: string) => void;
  activeRunId?: string | null;
  onRunChange?: (runId: string) => void;
}) {
  const slides = useMemo(
    () =>
      dedupeSlidesByInsight(
        userChartSlidesForSection(entries, sectionId, activeTab, chartSamples),
        excludeTitles,
      ) as RunSlide[],
    [entries, sectionId, activeTab, chartSamples, excludeTitles],
  );
  const sectionInsights = useMemo(() => collectSlidesInsights(slides), [slides]);
  const hasSectionInsights = sectionInsights.summaries.length > 0 || sectionInsights.insights.length > 0;

  if (!slides.length) return null;

  return (
    <div className="p-1">
      <WorkspaceChartCards
        slides={slides}
        keyPrefix={`user-${sectionId}`}
        dismissedKeys={dismissedKeys}
        onDismiss={onDismiss}
        activeRunId={activeRunId}
        onRunChange={onRunChange}
      />

      {hasSectionInsights ? (
        <CollapsibleInsights>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-secondary">
            {sectionInsights.summaries.map((summary, i) => (
              <li key={`user-summary-${i}`}>
                <InlineMarkdown content={summary} />
              </li>
            ))}
            {sectionInsights.insights.map((insight, i) => (
              <li key={`user-insight-${i}`}>
                <InlineMarkdown content={insight} />
              </li>
            ))}
          </ul>
        </CollapsibleInsights>
      ) : null}
    </div>
  );
}

// Small inline icons used in MessageRow (kept local to avoid extra imports)
function PlusSquareIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.33" className={className} aria-hidden>
      <rect x="2" y="2" width="12" height="12" rx="2" />
      <path d="M8 5v6M5 8h6" strokeLinecap="round" />
    </svg>
  );
}

function Edit01Icon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
    </svg>
  );
}

function CheckCheckIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <path d="M2 13l4 4L16 7" />
      <path d="M9 17l4 4L24 8" />
    </svg>
  );
}

function DotMark() {
  return (
    <div className="grid grid-cols-2 gap-0.5">
      <span className="size-1 rounded-full bg-[#1565ef]" />
      <span className="size-1 rounded-full bg-[#1565ef]" />
      <span className="size-1 rounded-full bg-[#1565ef]" />
      <span className="size-1 rounded-full bg-[#1565ef]" />
    </div>
  );
}

function DotCluster() {
  return (
    <div className="relative size-32">
      <div className="absolute inset-0 rounded-full bg-[#f4f4f5]" />
      <div className="absolute inset-0 grid place-items-center">
        <div className="grid grid-cols-4 gap-1.5">
          {Array.from({ length: 16 }).map((_, i) => (
            <span
              key={i}
              className="size-1.5 rounded-full bg-[#1565ef]"
              style={{ opacity: 0.4 + ((i * 37) % 60) / 100 }}
            />
          ))}
        </div>
      </div>
      <span className="absolute -top-1 left-8 size-1.5 rounded-full bg-[#1565ef]" />
      <span className="absolute top-6 -right-2 size-2 rounded-full bg-[#1565ef]" />
      <span className="absolute -bottom-2 left-4 size-1.5 rounded-full bg-[#d0d5dd]" />
      <span className="absolute top-2 right-4 size-1 rounded-full bg-[#d0d5dd]" />
    </div>
  );
}

type TableRow = {
  id: number;
  month: string;
  year: number;
  branch: string;
  category: string;
  revenue: number;
  growth: number;
};

const tableRows: TableRow[] = (() => {
  const months = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"];
  const branches = ["North", "South", "East", "West"];
  const categories = ["Electronics", "Apparel", "Home", "Beauty", "Sports"];
  const rows: TableRow[] = [];
  for (let i = 0; i < 36; i++) {
    const seed = (i * 9301 + 49297) % 233280;
    const r = seed / 233280;
    rows.push({
      id: i + 1,
      month: months[i % 12],
      year: 2020 + (i % 3),
      branch: branches[i % branches.length],
      category: categories[i % categories.length],
      revenue: Math.round(20000 + r * 80000),
      growth: Math.round((r * 40 - 15) * 10) / 10,
    });
  }
  return rows;
})();

type UploadedDataset = {
  filename: string;
  schema: string[];
  samples: Record<string, unknown>[];
  rows_sampled: number;
  size_mb?: number;

};

function DataPreviewPanel({
  dataset: datasetProp,
  filename,
}: {
  dataset?: UploadedDataset | null;
  filename?: string;
}) {
  const sessionDataset = useUploadedDataset();
  const dataset = datasetProp ?? sessionDataset;
  const columns: string[] = dataset?.schema ?? [];
  const samples: any[] = dataset?.samples ?? [];
  const rows = samples.slice(0, 50);
  const cellOf = (row: any, col: string) => {
    if (Array.isArray(row)) return row[columns.indexOf(col)];
    return row?.[col];
  };
  const title = filename || dataset?.filename || "Dataset preview";

  return (
    <div className="mt-4 flex flex-col">
      <div className="mb-3 min-w-0">
        <h3 className="truncate text-sm font-semibold text-primary">{title}</h3>
        <p className="mt-0.5 text-xs text-tertiary">
          {columns.length} columns · preview of {dataset?.rows_sampled ?? samples.length} rows — not full analysis results
          {formatDatasetSize(dataset?.size_mb) ? ` · ${formatDatasetSize(dataset?.size_mb)}` : ""}
        </p>
      </div>
      <div className="flex flex-col overflow-hidden">

          {rows.length === 0 || columns.length === 0 ? (
            <p className="text-sm text-tertiary">
              No dataset preview is available. Try re-uploading the file to view its preview.
            </p>
          ) : (
            <div className="max-h-[60vh] overflow-auto rounded-lg border border-secondary">

              <table className="w-full border-collapse text-sm">
                <thead className="sticky top-0 z-10 bg-secondary">
                  <tr>
                    <th className="w-10 px-3 py-2 text-left text-xs font-medium text-tertiary">#</th>
                    {columns.map((c) => (
                      <th key={c} className="whitespace-nowrap px-3 py-2 text-left text-xs font-medium text-secondary">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => (
                    <tr key={i} className="border-t border-secondary hover:bg-secondary/60">
                      <td className="px-3 py-2 text-xs text-tertiary">{i + 1}</td>
                      {columns.map((c) => {
                        const v = cellOf(row, c);
                        const s = v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
                        return (
                          <td key={c} className="px-3 py-2 text-primary">
                            <div className="max-w-[200px] truncate" title={s}>
                              {s}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
  );
}

function useUploadedDataset(): UploadedDataset | null {
  const [data, setData] = useState<UploadedDataset | null>(null);
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = sessionStorage.getItem("analysis:dataset");
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.schema) && Array.isArray(parsed.samples)) {
        setData(parsed as UploadedDataset);
      }
    } catch {}
  }, []);
  return data;
}

function formatCell(v: unknown): { text: string; isNumber: boolean } {
  if (v === null || v === undefined || v === "") return { text: "—", isNumber: false };
  if (typeof v === "number" && Number.isFinite(v)) {
    return { text: v.toLocaleString(undefined, { maximumFractionDigits: 4 }), isNumber: true };
  }
  if (typeof v === "object") return { text: JSON.stringify(v), isNumber: false };
  return { text: String(v), isNumber: false };
}

function DataTableView({
  data,
  dataset: datasetProp,
  disableMockFallback = false,
}: {
  data?: Record<string, unknown>[];
  dataset?: UploadedDataset | null;
  disableMockFallback?: boolean;
} = {}) {
  const sessionDataset = useUploadedDataset();
  const dataset = datasetProp ?? sessionDataset;
  const pageSize = 8;
  const [page, setPage] = useState(1);

  const usingProvided = Array.isArray(data) && data.length > 0;
  const usingReal = !usingProvided && !!dataset && dataset.samples.length > 0;

  let columns: string[];
  let rows: Record<string, unknown>[];
  if (usingProvided) {
    const keys = new Set<string>();
    data!.slice(0, 50).forEach((r) => Object.keys(r ?? {}).forEach((k) => keys.add(k)));
    columns = Array.from(keys);
    rows = data!;
  } else if (usingReal) {
    columns = dataset!.schema;
    rows = dataset!.samples;
  } else if (disableMockFallback) {
    columns = ["—"];
    rows = [];
  } else {
    columns = ["Month", "Year", "Branch", "Category", "Revenue", "Growth %"];
    rows = tableRows.map((r) => ({
      Month: r.month,
      Year: r.year,
      Branch: r.branch,
      Category: r.category,
      Revenue: r.revenue,
      "Growth %": r.growth,
    }));
  }

  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * pageSize;
  const pageRows = rows.slice(start, start + pageSize);
  const totalRows = usingReal ? dataset!.rows_sampled : rows.length;

  return (
    <div className="mt-4 flex max-h-[min(640px,calc(100vh-260px))] flex-col overflow-hidden rounded-xl border border-secondary bg-primary">
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-secondary text-tertiary">
            <tr>
              <th className="px-4 py-2.5 text-left font-medium">#</th>
              {columns.map((c) => (
                <th key={c} className="px-4 py-2.5 text-left font-medium whitespace-nowrap">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length + 1} className="px-4 py-10 text-center text-sm text-tertiary">
                  No table data saved for this analysis yet.
                </td>
              </tr>
            ) : (
              pageRows.map((row, i) => (
                <tr key={start + i} className="border-t border-secondary">
                  <td className="px-4 py-2.5 text-tertiary">{start + i + 1}</td>
                  {columns.map((c) => {
                    const { text, isNumber } = formatCell(row[c]);
                    return (
                      <td
                        key={c}
                        className={cx(
                          "px-4 py-2.5 whitespace-nowrap",
                          isNumber ? "text-right font-medium text-primary" : "text-secondary",
                        )}
                      >
                        {text}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="flex shrink-0 items-center justify-between border-t border-secondary px-4 py-3">
        <p className="text-xs text-tertiary">
          Showing {start + 1}-{Math.min(start + pageSize, rows.length)} of {rows.length.toLocaleString()}
          {usingReal && totalRows > rows.length ? ` (preview · ${totalRows.toLocaleString()} rows total)` : ""}
        </p>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={safePage === 1}
            className="rounded-md border border-secondary bg-primary px-3 py-1.5 text-xs font-semibold text-primary disabled:cursor-not-allowed disabled:opacity-50"
          >
            Previous
          </button>
          {(() => {
            const visiblePages: (number | "ellipsis")[] = [];
            const startWindow = Math.max(2, safePage - 2);
            const endWindow = Math.min(totalPages - 1, safePage + 2);
            visiblePages.push(1);
            if (startWindow > 2) visiblePages.push("ellipsis");
            for (let n = startWindow; n <= endWindow; n++) visiblePages.push(n);
            if (endWindow < totalPages - 1) visiblePages.push("ellipsis");
            if (totalPages > 1) visiblePages.push(totalPages);

            return visiblePages.map((item, index) => {
              if (item === "ellipsis") {
                return (
                  <span key={`ellipsis-${index}`} className="px-1 text-xs text-tertiary">
                    ...
                  </span>
                );
              }
              const n = item;
              return (
                <button
                  key={n}
                  onClick={() => setPage(n)}
                  className={cx(
                    "size-8 rounded-md text-xs font-semibold",
                    n === safePage ? "bg-[#1565ef] text-white" : "text-secondary hover:bg-secondary",
                  )}
                >
                  {n}
                </button>
              );
            });
          })()}
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={safePage === totalPages}
            className="rounded-md border border-secondary bg-primary px-3 py-1.5 text-xs font-semibold text-primary disabled:cursor-not-allowed disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

function MoveSuccessModal({
  open,
  projectName,
  onOpenChange,
}: {
  open: boolean;
  projectName: string;
  onOpenChange: (v: boolean) => void;
}) {
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => onOpenChange(false), 2200);
    return () => clearTimeout(t);
  }, [open, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[420px] gap-0 rounded-2xl border-secondary bg-primary p-8 shadow-xl">
        <div className="flex flex-col items-center text-center">
          <div className="grid size-14 place-items-center rounded-full bg-[#dcfae6]">
            <div className="grid size-10 place-items-center rounded-full bg-[#a9efc5]">
              <Check className="size-5 text-[#067647]" />
            </div>
          </div>
          <DialogTitle className="mt-5 text-lg font-semibold text-primary">Successfully Moved to Projects</DialogTitle>
          <p className="mt-1 text-sm text-tertiary">{projectName}</p>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ----- Reports Modal (Dataset.01) -----

type ReportNode = {
  id: string;
  label: string;
  type: "set" | "folder" | "analysis" | "table" | "year";
  badge?: number;
  children?: ReportNode[];
};

const reportTree: ReportNode[] = [
  { id: "ds1", label: "Data Set 1", type: "set" },
  { id: "ds3", label: "Data Set 3", type: "set" },
  { id: "ds4", label: "Data Set 4", type: "set" },
  { id: "ds2", label: "Data Set 2", type: "set" },
  {
    id: "dataset",
    label: "Dataset",
    type: "set",
    badge: 20,
    children: [
      {
        id: "f1",
        label: "Folder 01",
        type: "folder",
        badge: 20,
        children: [],
      },
      {
        id: "f2",
        label: "Folder 02",
        type: "folder",
        badge: 20,
        children: [
          { id: "a1", label: "Analysis 1", type: "analysis" },
          { id: "a2", label: "Analysis 2", type: "analysis" },
          { id: "a3", label: "Analysis 3", type: "analysis" },
          { id: "a4", label: "Analysis 4", type: "analysis" },
          {
            id: "a5",
            label: "Analysis 5",
            type: "analysis",
            children: [
              { id: "t1", label: "Table 1", type: "table" },
              { id: "t2", label: "Table 2", type: "table" },
              { id: "t3", label: "Table 3", type: "table" },
            ],
          },
        ],
      },
    ],
  },
  {
    id: "fn",
    label: "Folder Name",
    type: "folder",
    badge: 20,
    children: [
      { id: "fa1", label: "Analysis 1", type: "analysis" },
      {
        id: "fa2",
        label: "Analysis 2",
        type: "analysis",
        children: [{ id: "y2025", label: "2025", type: "year" }],
      },
    ],
  },
];

const reportRows = [
  {
    org: "Ai Insights",
    isHeader: true,
    r24: [43000, 54000],
    r25: [55500, 47600, 49100],
    yoy: [72, -68],
    net: [59, -47, 63, -78],
  },
  { org: "LT Inc", r24: 43000, r25: 55500, yoy: 72, net: 59 },
  { org: "Nova Inc Networks Inc", r24: 54000, r25: 47600, yoy: -68, net: -47 },
  { org: "Zenith Inc Zones Inc", r24: 24000, r25: 49100, yoy: 86, net: 63 },
  { org: "Pinnacle Inc Products Inc", r24: 32000, r25: 61900, yoy: -95, net: -78 },
  { org: "Radiant Inc Realms Inc", r24: 42000, r25: 52400, yoy: 81, net: -52 },
  { org: "Acorn Inc Enterprises Inc", r24: 23000, r25: 59300, yoy: -52, net: 81 },
  { org: "Summit Inc Systems Inc", r24: 21000, r25: 38700, yoy: -78, net: -95 },
];

function MiniBars({ values }: { values: number[] }) {
  const max = Math.max(...values.map((v) => Math.abs(v)));
  return (
    <div className="flex h-16 items-end gap-1">
      {values.map((v, i) => (
        <div
          key={i}
          className="w-5 rounded-sm bg-[#1565ef]"
          style={{ height: `${Math.max(8, (Math.abs(v) / max) * 100)}%` }}
        />
      ))}
    </div>
  );
}

export function ReportsModal({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  return <CurrentDatasetModal open={open} onOpenChange={onOpenChange} datasetName="Dataset.01" />;
}

export function CurrentDatasetModal({
  open,
  onOpenChange,
  datasetName,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  datasetName: string;
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    dataset: true,
    f2: true,
    a5: true,
    fn: true,
    fa2: true,
  });
  const [activeId, setActiveId] = useState("f2");

  const toggle = (id: string) => setExpanded((e) => ({ ...e, [id]: !e[id] }));

  const renderNode = (n: ReportNode, depth: number): ReactNode => {
    const hasChildren = n.children !== undefined;
    const isOpen = !!expanded[n.id];
    const isActive = activeId === n.id;
    const Icon =
      n.type === "table"
        ? Table
        : n.type === "analysis" || n.type === "year"
          ? Folder
          : n.type === "set"
            ? Database01
            : Folder;
    const iconColor = n.type === "table" ? "text-[#079455]" : "text-fg-quaternary";

    return (
      <div key={n.id}>
        <button
          type="button"
          onClick={() => {
            setActiveId(n.id);
            if (hasChildren) toggle(n.id);
          }}
          className={cx(
            "flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-sm",
            isActive ? "bg-secondary text-primary" : "text-secondary hover:bg-secondary",
          )}
          style={{ paddingLeft: 8 + depth * 14 }}
        >
          {hasChildren ? (
            <ChevronDown className={cx("size-3.5 shrink-0 text-fg-quaternary transition", !isOpen && "-rotate-90")} />
          ) : (
            <span className="w-3.5" />
          )}
          <Icon className={cx("size-4 shrink-0", iconColor)} />
          <span className="flex-1 truncate">{n.label}</span>
          {n.badge !== undefined && (
            <span className="rounded-full bg-[#e0f2fe] px-1.5 py-0.5 text-[10px] font-semibold text-[#026aa2]">
              {n.badge}
            </span>
          )}
        </button>
        {hasChildren && isOpen && <div>{n.children!.map((c) => renderNode(c, depth + 1))}</div>}
      </div>
    );
  };

  const tableCols = [
    "Organization",
    "Revenue_2024_MUSD",
    "Revenue_2025_MUSD",
    "YoY_Sales_Percent",
    "Net Revenue or Loss 20...",
  ];
  const tableRows = [
    { org: "LT Inc", a: "43000", b: "55500", c: "72", d: "59" },
    { org: "Nova Inc Networks Inc", a: "54000", b: "47600", c: "-68", d: "-47" },
    { org: "Zenith Inc Zones Inc", a: "24000", b: "49100", c: "86", d: "63" },
    { org: "Pinnacle Inc Products Inc", a: "32000", b: "61900", c: "-95", d: "-78" },
    { org: "Radiant Inc Realms Inc", a: "42000", b: "52400", c: "81", d: "-52" },
    { org: "Acorn Inc Enterprises Inc", a: "23000", b: "59300", c: "-52", d: "81" },
    { org: "Summit Inc Systems Inc", a: "21000", b: "38700", c: "-78", d: "-95" },
  ];
  const insightBars = [
    [
      { a: 43, b: 79 },
      { a: 54, b: 21 },
    ],
    [
      { a: 55, b: 79 },
      { a: 47, b: 21 },
      { a: 49, b: 18 },
    ],
    [
      { a: 72, b: 79 },
      { a: -68, b: 21 },
      { a: 86, b: 18 },
    ],
    [
      { a: 59, b: 79 },
      { a: -47, b: 21 },
      { a: 63, b: 18 },
      { a: -78, b: 18 },
    ],
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="grid h-[min(720px,calc(100vh-64px))] w-[min(1080px,calc(100vw-48px))] max-w-none grid-rows-[auto_1fr] gap-0 overflow-hidden rounded-2xl border-secondary bg-primary p-0 shadow-2xl">
        <div className="flex items-center justify-between gap-2 border-b border-secondary px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              className="grid size-7 place-items-center rounded-md text-fg-quaternary hover:bg-secondary"
              aria-label="Back"
            >
              <ArrowLeft className="size-4" />
            </button>
            <Database01 className="size-4 text-fg-secondary" />
            <DialogTitle className="text-sm font-semibold text-primary">{datasetName || "Dataset.01"}</DialogTitle>
          </div>
          <button className="inline-flex items-center gap-1.5 rounded-lg bg-[#1565ef] px-3 py-1.5 text-sm font-semibold text-white shadow-xs hover:bg-[#1255cf]">
            <Check className="size-4" /> Select Dataset
          </button>
        </div>

        <div className="grid min-h-0 grid-cols-[260px_1fr] overflow-hidden">
          <aside className="flex min-h-0 flex-col border-r border-secondary">
            <div className="px-3 pt-3">
              <div className="flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 shadow-xs">
                <SearchLg className="size-4 text-fg-quaternary" />
                <input
                  placeholder="Search for trades"
                  className="flex-1 bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
                />
                <span className="rounded border border-secondary px-1 text-[10px] text-tertiary">⌘K</span>
              </div>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">{reportTree.map((n) => renderNode(n, 0))}</div>
          </aside>

          <section className="flex min-h-0 flex-col bg-secondary/30 p-4">
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-secondary bg-primary">
              <div className="flex items-center gap-2 border-b border-secondary px-4 py-3">
                <Table className="size-4 text-[#079455]" />
                <h3 className="text-sm font-semibold text-primary">Table 1</h3>
              </div>
              <div className="min-h-0 flex-1 overflow-auto">
                <table className="w-full min-w-[720px] border-collapse text-sm">
                  <thead className="sticky top-0 bg-primary">
                    <tr className="border-b border-secondary text-left text-xs font-semibold text-tertiary">
                      {tableCols.map((c) => (
                        <th key={c} className="whitespace-nowrap px-4 py-2.5">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-secondary align-middle">
                      <td className="px-4 py-3 text-sm font-medium text-primary">AI Insights</td>
                      {insightBars.map((bars, i) => (
                        <td key={i} className="px-4 py-2">
                          <div className="flex h-20 items-end gap-1.5">
                            {bars.map((b, j) => (
                              <div key={j} className="flex flex-col items-center gap-1">
                                <span className="text-[9px] font-medium text-tertiary">{b.a}</span>
                                <div
                                  className="w-3.5 rounded-t bg-[#1565ef]"
                                  style={{ height: `${Math.min(60, Math.abs(b.a) / 1.2)}px` }}
                                />
                                <span className="text-[9px] text-tertiary">{b.b}%</span>
                              </div>
                            ))}
                          </div>
                        </td>
                      ))}
                    </tr>
                    {tableRows.map((r, idx) => (
                      <tr key={idx} className="border-b border-secondary last:border-b-0">
                        <td className="whitespace-nowrap px-4 py-3 text-sm text-primary">{r.org}</td>
                        <td className="px-4 py-3 text-sm text-secondary">{r.a}</td>
                        <td className="px-4 py-3 text-sm text-secondary">{r.b}</td>
                        <td className="px-4 py-3 text-sm text-secondary">{r.c}</td>
                        <td className="px-4 py-3 text-sm text-secondary">{r.d}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}

type CloudDataset = {
  id: string;
  title: string;
  useCase: string;
  provider: string;
  providerColor: string;
  author: string;
  authorAvatar: string;
  updated: string;
  icon: "folder" | "dollar" | "image";
  fileCount?: number;
};

function StopListeningIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden>
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function MicIcon({ className }: { className?: string }) {

  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="18" x2="12" y2="22" />
    </svg>
  );
}

function AudioWaveIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      className={className}
      aria-hidden
    >
      <line x1="3" y1="9" x2="3" y2="11" />
      <line x1="6" y1="7" x2="6" y2="13" />
      <line x1="9" y1="4" x2="9" y2="16" />
      <line x1="12" y1="6" x2="12" y2="14" />
      <line x1="15" y1="3" x2="15" y2="17" />
      <line x1="18" y1="8" x2="18" y2="12" />
    </svg>
  );
}

const revenueBarData = [
  { month: "Jan", y2024: 2700, y2025: 3000 },
  { month: "Feb", y2024: -500, y2025: 6500 },
  { month: "March", y2024: 1100, y2025: -600 },
  { month: "April", y2024: -400, y2025: 1750 },
  { month: "May", y2024: -700, y2025: 1850 },
  { month: "June", y2024: 1050, y2025: -500 },
  { month: "July", y2024: -350, y2025: 1700 },
  { month: "Aug", y2024: 4100, y2025: -400 },
  { month: "Sept", y2024: 100, y2025: 900 },
  { month: "Oct", y2024: -150, y2025: -700 },
];

type PastedAnalysisSectionProps = {
  runs: TableRun[];
  slides: RunSlide[];
  insights: { summaries: string[]; insights: string[] };
  index: number;
};

function PastedAnalysisSection({ runs, slides, insights, index }: PastedAnalysisSectionProps) {
  const [pastedView, setPastedView] = useState<"table" | "chart">("chart");
  const [runIdx, setRunIdx] = useState(0);
  const safeIdx = Math.min(runIdx, Math.max(runs.length - 1, 0));
  const run = runs[safeIdx];
  const groupKey = runs[0]?.id ?? `pasted-${index}`;
  if (!run) return null;
  return (
    <section className="mt-8 border-t border-secondary pt-6" aria-labelledby={`${groupKey}-title`}>
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase text-brand-secondary">
            Pasted analysis {index + 1}
          </p>
          <h3 id={`${groupKey}-title`} className="mt-1 truncate text-base font-semibold text-primary">
            {run.title}
          </h3>
          <p className="mt-0.5 text-xs text-tertiary">
            {runs.length} result{runs.length === 1 ? "" : "s"} · {slides.length} chart
            {slides.length === 1 ? "" : "s"}
          </p>
        </div>
        <div className="inline-flex shrink-0 items-center rounded-lg border border-secondary bg-primary p-1 shadow-xs">
          <button
            onClick={() => setPastedView("table")}
            className={cx(
              "rounded-md px-3 py-1.5 text-sm font-semibold transition",
              pastedView === "table" ? "bg-secondary text-primary" : "text-tertiary",
            )}
          >
            Data table
          </button>
          <button
            onClick={() => setPastedView("chart")}
            className={cx(
              "rounded-md px-3 py-1.5 text-sm font-semibold transition",
              pastedView === "chart" ? "bg-secondary text-primary" : "text-tertiary",
            )}
          >
            Chart
          </button>
        </div>
      </div>

      {pastedView === "chart" ? (
        slides.length ? (
          <WorkspaceChartCards slides={slides} keyPrefix={groupKey} />
        ) : (
          <p className="rounded-lg border border-secondary bg-secondary/40 px-4 py-3 text-sm text-tertiary">
            No chart could be derived from the pasted data.
          </p>
        )
      ) : (
        <>
          {runs.length > 1 && (
            <div className="mb-3 flex items-center gap-2 text-sm">
              <button
                type="button"
                onClick={() => setRunIdx((i) => (i - 1 + runs.length) % runs.length)}
                className="rounded-md border border-secondary px-2 py-1 text-tertiary hover:text-primary"
              >
                Prev
              </button>
              <span className="text-tertiary">
                Result {safeIdx + 1} of {runs.length}
              </span>
              <button
                type="button"
                onClick={() => setRunIdx((i) => (i + 1) % runs.length)}
                className="rounded-md border border-secondary px-2 py-1 text-tertiary hover:text-primary"
              >
                Next
              </button>
            </div>
          )}
          <DataTableView data={run.rows} disableMockFallback />
        </>
      )}


      <div className="mt-5">
        <CollapsibleInsights title="Pasted result insights">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-secondary">
            {insights.summaries.length > 0 || insights.insights.length > 0 ? (
              <>
                {insights.summaries.map((summary, i) => (
                  <li key={`${run.id}-summary-${i}`}>
                    <InlineMarkdown content={summary} />
                  </li>
                ))}
                {insights.insights.map((insight, i) => (
                  <li key={`${run.id}-insight-${i}`}>
                    <InlineMarkdown content={insight} />
                  </li>
                ))}
              </>
            ) : (
              <li className="text-tertiary">No insights could be derived from the pasted data.</li>
            )}
          </ul>
        </CollapsibleInsights>
      </div>
    </section>
  );
}

function RevenueBarSection({ title = "Revenue 2026 graph" }: { title?: string }) {
  const [view, setView] = useState<"table" | "chart">("chart");
  return (
    <div className="mt-0">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-md font-semibold text-primary">{title}</h3>
        <TagPicker />
      </div>

      <div className="mt-4 rounded-xl border border-secondary bg-primary p-4">
        <div className="flex items-center justify-end gap-2">
          <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-1 shadow-xs">
            <button
              onClick={() => setView("table")}
              className={cx(
                "rounded-md px-3 py-1.5 text-sm font-semibold transition",
                view === "table" ? "bg-secondary text-primary" : "text-tertiary",
              )}
            >
              Data table
            </button>
            <button
              onClick={() => setView("chart")}
              className={cx(
                "rounded-md px-3 py-1.5 text-sm font-semibold transition",
                view === "chart" ? "bg-secondary text-primary" : "text-tertiary",
              )}
            >
              Chart
            </button>
          </div>
          </div>

        {view === "chart" ? (
          <>
            <div className="mt-4 h-[340px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={revenueBarData} stackOffset="sign" margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="month" tick={{ fill: "#667085", fontSize: 12 }} tickLine={false} axisLine={false} />
                  <YAxis
                    domain={[-2000, 7000]}
                    ticks={[-2000, -500, 1000, 2500, 4000, 5500, 7000]}
                    tick={{ fill: "#667085", fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <ReferenceLine y={0} stroke="#d0d5dd" />
                  <Bar dataKey="y2024" stackId="rev" fill="#1565ef" radius={[2, 2, 0, 0]} barSize={28} />
                  <Bar dataKey="y2025" stackId="rev" fill="#b2ddff" radius={[2, 2, 0, 0]} barSize={28} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 flex items-center justify-center gap-6 text-xs text-secondary">
              <span className="inline-flex items-center gap-2">
                <span className="size-2.5 rounded-sm bg-[#1565ef]" /> 2024
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="size-2.5 rounded-sm bg-[#b2ddff]" /> 2025
              </span>
            </div>
          </>
        ) : (
          <DataTableView />
        )}
      </div>

      <ul className="mt-4 list-disc space-y-1.5 pl-5 text-sm text-secondary">
        <li>
          The chart compares revenue performance across brands for 2024 and 2025, highlighting year-over-year changes.
          Overall, revenue trends vary significantly by brand. Several brands show improved performance in 2025, while
          others experience a decline or volatility.
        </li>
      </ul>

      <h4 className="mt-4 text-sm font-semibold text-primary">Key Insights</h4>
      <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-secondary">
        <li>
          Zara and Uniqlo continue to generate high revenue, with Uniqlo showing strong growth in 2025 compared to 2024.
        </li>
        <li>
          Shein records a notable increase in 2024 but shows a pullback in 2025, indicating possible short-term
          volatility.
        </li>
        <li>
          Nike, Adidas, and Puma display relatively stable performance, with moderate fluctuations year-over-year.
        </li>
      </ul>
    </div>
  );
}
