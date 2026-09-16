// Analysis workspace activity history — aggregated from existing tables.
// Not the chat thread UI; a read-only activity feed for the History panel.

import { loadMessages, type AnalysisMessage } from "@/lib/analysis-messages";
import { dashboardWidgetsForHistory } from "@/lib/api/project-dashboard";
import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";

export type AnalysisHistoryActorKind = "user" | "ai";

export type AnalysisHistoryEntry = {
  id: string;
  analysis_id: string;
  actor_name: string;
  actor_initials: string | null;
  actor_kind: AnalysisHistoryActorKind;
  description: string;
  created_at: string;
  output: unknown | null;
};

type AnalysisRow = {
  id: string;
  name: string;
  filename: string | null;
  dataset_id: string | null;
  project_id: string | null;
  viz_config: unknown;
  samples?: unknown[] | null;
  created_at: string;
  updated_at: string;
  created_by_name: string | null;
  created_by_initials: string | null;
};

type DashboardGraphRow = {
  id: string;
  title: string;
  config: unknown;
  created_at: string;
};

type CollaboratorRow = {
  id: string;
  created_at: string;
  department: string | null;
  access_level: string;
  app_user_id: string;
  app_users: { name: string; email: string } | null;
};

type ShareLinkRow = {
  id: string;
  created_at: string;
  updated_at: string;
  link_access: string;
  share_token: string;
};

const LINK_ACCESS_LABEL: Record<string, string> = {
  view: "View",
  edit: "Edit",
  comment: "Comment",
  restricted: "Restricted",
};

const MEMBER_ACCESS_LABEL: Record<string, string> = {
  view: "View",
  edit: "Edit",
  comment: "Comment",
};

type InsightCommentRow = {
  id: string;
  content: string;
  created_at: string;
  parent_id: string | null;
  author: { full_name: string | null; avatar_url: string | null } | { full_name: string | null; avatar_url: string | null }[] | null;
};

function initialsFrom(name: string | null | undefined): string | null {
  if (!name) return null;
  if (name.includes("@")) {
    const local = name.split("@")[0] ?? "";
    const parts = local.split(/[._-]+/).filter(Boolean);
    if (parts.length >= 2) return `${parts[0][0] ?? ""}${parts[1][0] ?? ""}`.toUpperCase();
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return null;
  }
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() ?? "").join("") || null;
}

function formatDisplayName(name: string | null | undefined): string {
  if (!name) return "You";
  const trimmed = name.trim();
  if (!trimmed) return "You";
  if (!trimmed.includes("@")) return trimmed;

  const local = trimmed.split("@")[0] ?? "";
  const parts = local.split(/[._-]+/).filter(Boolean);
  if (parts.length >= 2) {
    return parts
      .slice(0, 2)
      .map((p) => p.charAt(0).toUpperCase() + p.slice(1).toLowerCase())
      .join(" ");
  }
  if (parts.length === 1) {
    return parts[0].charAt(0).toUpperCase() + parts[0].slice(1).toLowerCase();
  }
  return "You";
}

/** Parse viz_config when Supabase returns a JSON string. */
function parseVizConfig(raw: unknown): unknown {
  if (raw == null) return null;
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw) as unknown;
    } catch {
      return null;
    }
  }
  return raw;
}

export type HistoryViewPayload =
  | { kind: "viz"; config: unknown }
  | { kind: "table"; rows: Record<string, unknown>[] }
  | { kind: "message"; content: string }
  | { kind: "share"; token: string; access: string }
  | {
      kind: "collaborator";
      name: string;
      email: string | null;
      department: string | null;
      access: string;
    };

export function resolveHistoryViewPayload(output: unknown): HistoryViewPayload | null {
  if (!output || typeof output !== "object") return null;
  const o = output as Record<string, unknown>;

  const nested = parseVizConfig(o.viz_config);
  if (nested && typeof nested === "object") {
    return { kind: "viz", config: nested };
  }

  if (Array.isArray(o.charts) && o.charts.length) {
    return { kind: "viz", config: o };
  }

  if (o.encodings || o.type || o.data) {
    return { kind: "viz", config: o };
  }

  if (Array.isArray(o.output_json) && o.output_json.length) {
    return { kind: "table", rows: o.output_json as Record<string, unknown>[] };
  }

  if (typeof o.content === "string" && o.content.trim()) {
    return { kind: "message", content: o.content.trim() };
  }

  const collab = o.collaborator;
  if (collab && typeof collab === "object" && !Array.isArray(collab)) {
    const c = collab as Record<string, unknown>;
    return {
      kind: "collaborator",
      name: typeof c.name === "string" ? c.name : "Collaborator",
      email: typeof c.email === "string" ? c.email : null,
      department: typeof c.department === "string" ? c.department : null,
      access: typeof c.access === "string" ? c.access : "View",
    };
  }

  const share = o.share;
  if (share && typeof share === "object" && !Array.isArray(share)) {
    const s = share as Record<string, unknown>;
    const token = typeof s.token === "string" ? s.token : "";
    if (!token) return null;
    return {
      kind: "share",
      token,
      access: typeof s.access === "string" ? s.access : "View",
    };
  }

  return null;
}

export function isHistoryEntryViewable(output: unknown): boolean {
  return resolveHistoryViewPayload(output) !== null;
}

function truncate(text: string, max = 72): string {
  const normalized = text.trim().replace(/\s+/g, " ");
  if (!normalized) return "";
  if (normalized.length <= max) return normalized;
  return `${normalized.slice(0, max - 3)}...`;
}

function messageOutput(msg: AnalysisMessage): Record<string, unknown> | null {
  if (!msg.output || typeof msg.output !== "object") return null;
  return msg.output as Record<string, unknown>;
}

function hasViz(out: Record<string, unknown> | null | undefined): boolean {
  const cfg = parseVizConfig(out?.viz_config);
  if (cfg && typeof cfg === "object") {
    const c = cfg as Record<string, unknown>;
    return Boolean(Array.isArray(c.charts) && c.charts.length) || Boolean(c.encodings) || Boolean(c.type);
  }
  return Boolean(out?.viz_config);
}

function hasTable(out: Record<string, unknown> | null | undefined): boolean {
  const rows = out?.output_json;
  return Array.isArray(rows) && rows.length > 0;
}

function isChatTurn(msg: AnalysisMessage): boolean {
  return msg.role === "user" || msg.role === "assistant";
}

function userActorFor(author: { name: string; initials: string | null }) {
  return {
    actor_name: "You",
    actor_initials: author.initials ?? initialsFrom(author.name) ?? "YO",
    actor_kind: "user" as const,
  };
}

function userActor(row: AnalysisRow) {
  return userActorFor({
    name: row.created_by_name ?? "You",
    initials: row.created_by_initials,
  });
}

function aiActor() {
  return {
    actor_name: "Avaloka AI",
    actor_initials: "AI",
    actor_kind: "ai" as const,
  };
}

async function resolveCurrentAuthor(): Promise<{ name: string; initials: string | null }> {
  const data = { user: await getCachedAuthUser() };
  const user = data.user;
  if (!user) return { name: "You", initials: null };
  const { data: prof } = await supabase.from("profiles").select("full_name").eq("id", user.id).maybeSingle();
  const raw = (prof?.full_name as string | null) ?? user.email ?? null;
  return { name: raw ?? "You", initials: initialsFrom(raw) };
}

/** Match columns used elsewhere (e.g. loadAnalysis) — avoid optional columns missing on live DB. */
async function loadAnalysisRowForHistory(analysisId: string): Promise<AnalysisRow | null> {
  const primarySelect = "id,name,filename,dataset_id,project_id,viz_config,samples,created_at,updated_at";
  const { data, error } = await supabase.from("analyses").select(primarySelect).eq("id", analysisId).maybeSingle();

  if (!error && data) {
    const row = data as Omit<AnalysisRow, "created_by_name" | "created_by_initials">;
    return {
      ...row,
      viz_config: parseVizConfig(row.viz_config),
      samples: Array.isArray(row.samples) ? row.samples : null,
      created_by_name: null,
      created_by_initials: null,
    };
  }

  if (error) console.warn("[analysis-history] analyses", error.message);

  const { data: minimal, error: minimalError } = await supabase
    .from("analyses")
    .select("id,name,created_at,updated_at")
    .eq("id", analysisId)
    .maybeSingle();

  if (!minimalError && minimal) {
    return {
      id: minimal.id,
      name: minimal.name,
      filename: null,
      dataset_id: null,
      project_id: null,
      viz_config: null,
      samples: null,
      created_at: minimal.created_at,
      updated_at: minimal.updated_at,
      created_by_name: null,
      created_by_initials: null,
    };
  }

  if (minimalError) console.warn("[analysis-history] analyses (minimal)", minimalError.message);
  return null;
}

function withAuthor(row: AnalysisRow, author: { name: string; initials: string | null }): AnalysisRow {
  return {
    ...row,
    created_by_name: row.created_by_name ?? author.name,
    created_by_initials: row.created_by_initials ?? author.initials,
  };
}

function fallbackAnalysisRow(analysisId: string, author: { name: string; initials: string | null }): AnalysisRow {
  const now = new Date().toISOString();
  return {
    id: analysisId,
    name: "Analysis",
    filename: null,
    dataset_id: null,
    project_id: null,
    viz_config: null,
    samples: null,
    created_at: now,
    updated_at: now,
    created_by_name: author.name,
    created_by_initials: author.initials,
  };
}

function entryFromAssistantMessage(
  msg: AnalysisMessage,
  userPrompted: boolean,
  filename: string,
  row: AnalysisRow,
): AnalysisHistoryEntry | null {
  const out = messageOutput(msg);
  if (!hasViz(out) && !hasTable(out)) return null;

  if (hasViz(out)) {
    const actor = userPrompted ? userActor(row) : aiActor();
    return {
      id: msg.id,
      analysis_id: msg.analysis_id,
      ...actor,
      description: userPrompted ? `Created a visualisation cell for ${filename}.` : "Created a New visualisation cell.",
      created_at: msg.created_at,
      output: msg.output,
    };
  }

  const actor = userPrompted ? userActor(row) : aiActor();
  return {
    id: msg.id,
    analysis_id: msg.analysis_id,
    ...actor,
    description: "Generated analysis results.",
    created_at: msg.created_at,
    output: msg.output,
  };
}

function entriesFromChatMessages(chatMessages: AnalysisMessage[], row: AnalysisRow): AnalysisHistoryEntry[] {
  const filename = row.filename ?? "data_set";
  const entries: AnalysisHistoryEntry[] = [];

  for (let i = 0; i < chatMessages.length; i++) {
    const msg = chatMessages[i];
    if (msg.role === "user") {
      entries.push({
        id: `chat-user-${msg.id}`,
        analysis_id: msg.analysis_id,
        ...userActor(row),
        description: `Sent a message: "${truncate(msg.content)}"`,
        created_at: msg.created_at,
        output: { content: msg.content },
      });
      continue;
    }

    const prev = chatMessages[i - 1];
    const userPrompted = prev?.role === "user";
    const structured = entryFromAssistantMessage(msg, userPrompted, filename, row);
    if (structured) {
      entries.push(structured);
      continue;
    }

    const preview = truncate(msg.content);
    entries.push({
      id: `chat-ai-${msg.id}`,
      analysis_id: msg.analysis_id,
      ...aiActor(),
      description: preview ? `Replied: "${preview}"` : "Avaloka AI replied in chat.",
      created_at: msg.created_at,
      output: { content: msg.content },
    });
  }

  return entries;
}

function entriesFromAnalysisRow(row: AnalysisRow, hasMessageViz: boolean): AnalysisHistoryEntry[] {
  const entries: AnalysisHistoryEntry[] = [];
  const actor = userActor(row);
  const filename = row.filename ?? "data_set";
  const sampleRows = Array.isArray(row.samples) ? (row.samples as Record<string, unknown>[]) : [];

  if (row.filename || row.dataset_id) {
    entries.push({
      id: `analysis-upload-${row.id}`,
      analysis_id: row.id,
      ...actor,
      description: `Uploaded ${filename} for analysis.`,
      created_at: row.created_at,
      output: sampleRows.length ? { output_json: sampleRows, filename } : null,
    });
  } else {
    entries.push({
      id: `analysis-created-${row.id}`,
      analysis_id: row.id,
      ...actor,
      description: `Started analysis "${row.name}".`,
      created_at: row.created_at,
      output: null,
    });
  }

  const vizConfig = parseVizConfig(row.viz_config);
  if (vizConfig && typeof vizConfig === "object" && !hasMessageViz) {
    const vizTime = row.updated_at && row.updated_at !== row.created_at ? row.updated_at : row.created_at;
    entries.push({
      id: `analysis-auto-viz-${row.id}`,
      analysis_id: row.id,
      ...aiActor(),
      description: "Created a New visualisation cell.",
      created_at: vizTime,
      output: { viz_config: vizConfig },
    });
  }

  return entries;
}

function entriesFromDashboardGraphs(
  analysisId: string,
  graphs: DashboardGraphRow[],
  row: AnalysisRow,
): AnalysisHistoryEntry[] {
  const actor = userActor(row);
  return graphs.map((graph) => ({
    id: `graph-${graph.id}`,
    analysis_id: analysisId,
    ...actor,
    description: `Added "${graph.title}" chart to dashboard.`,
    created_at: graph.created_at,
    output: { viz_config: parseVizConfig(graph.config) ?? graph.config },
  }));
}

function entriesFromCollaborators(
  analysisId: string,
  collaborators: CollaboratorRow[],
  row: AnalysisRow,
): AnalysisHistoryEntry[] {
  const actor = userActor(row);
  return collaborators.map((collab) => {
    const invitee = collab.app_users?.name ?? collab.app_users?.email ?? collab.department ?? "a collaborator";
    const access = MEMBER_ACCESS_LABEL[collab.access_level] ?? "View";
    return {
      id: `collab-${collab.id}`,
      analysis_id: analysisId,
      ...actor,
      description: `Invited ${invitee} to collaborate (${access} access).`,
      created_at: collab.created_at,
      output: {
        collaborator: {
          name: invitee,
          email: collab.app_users?.email ?? null,
          department: collab.department,
          access,
        },
      },
    };
  });
}

function entriesFromShareLink(analysisId: string, share: ShareLinkRow, row: AnalysisRow): AnalysisHistoryEntry[] {
  const actor = userActor(row);
  const access = LINK_ACCESS_LABEL[share.link_access] ?? "View";
  const shareOutput = {
    share: {
      token: share.share_token,
      access,
    },
  };
  const entries: AnalysisHistoryEntry[] = [
    {
      id: `share-${share.id}`,
      analysis_id: analysisId,
      ...actor,
      description: `Enabled link sharing (${access} access).`,
      created_at: share.created_at,
      output: shareOutput,
    },
  ];
  if (share.updated_at && share.updated_at !== share.created_at) {
    entries.push({
      id: `share-update-${share.id}`,
      analysis_id: analysisId,
      ...actor,
      description: `Updated share link access to ${access}.`,
      created_at: share.updated_at,
      output: shareOutput,
    });
  }
  return entries;
}

function entriesFromInsightComments(
  analysisId: string,
  comments: InsightCommentRow[],
  row: AnalysisRow,
): AnalysisHistoryEntry[] {
  return comments.map((comment) => {
    const authorObj = Array.isArray(comment.author) ? comment.author[0] ?? null : comment.author;
    const rawName = authorObj?.full_name ?? row.created_by_name;
    const isOwn =
      !rawName || !row.created_by_name || rawName.trim().toLowerCase() === row.created_by_name.trim().toLowerCase();
    const isReply = Boolean(comment.parent_id);
    return {
      id: `insight-comment-${comment.id}`,
      analysis_id: analysisId,
      actor_name: isOwn ? "You" : formatDisplayName(rawName),
      actor_initials: initialsFrom(rawName) ?? initialsFrom(formatDisplayName(rawName)),
      actor_kind: "user" as const,
      description: isReply
        ? `Replied to an insight comment: "${truncate(comment.content)}"`
        : `Added an insight comment: "${truncate(comment.content)}"`,
      created_at: comment.created_at,
      output: { content: comment.content },
    };
  });
}

export function formatHistoryRelativeTime(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  if (diffMs < 60_000) return "Just now";
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 60) return `${String(mins).padStart(2, "0")} min Ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} hr${hours === 1 ? "" : "s"} Ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} Ago`;
}

/** Local calendar date key (YYYY-MM-DD) for grouping history entries. */
export function historyDateKey(iso: string): string {
  const d = new Date(iso);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Group header: Today, Yesterday, or full date. */
export function formatHistoryDateGroupLabel(dateKey: string): string {
  const today = historyDateKey(new Date().toISOString());
  if (dateKey === today) return "Today";

  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  if (dateKey === historyDateKey(yesterday.toISOString())) return "Yesterday";

  const [y, m, d] = dateKey.split("-").map(Number);
  if (!y || !m || !d) return dateKey;
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

/** Clock time shown on each row within a date group. */
export function formatHistoryClockTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

export type HistoryDateGroup = {
  dateKey: string;
  label: string;
  entries: AnalysisHistoryEntry[];
};

export function groupHistoryEntriesByDate(entries: AnalysisHistoryEntry[]): HistoryDateGroup[] {
  const map = new Map<string, AnalysisHistoryEntry[]>();
  for (const entry of entries) {
    const key = historyDateKey(entry.created_at);
    const list = map.get(key) ?? [];
    list.push(entry);
    map.set(key, list);
  }

  return Array.from(map.entries())
    .sort(([a], [b]) => b.localeCompare(a))
    .map(([dateKey, groupEntries]) => ({
      dateKey,
      label: formatHistoryDateGroupLabel(dateKey),
      entries: groupEntries.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    }));
}

async function loadOptional<T>(
  label: string,
  fetcher: () => PromiseLike<{ data: T | null; error: { message: string } | null }>,
): Promise<T | null> {
  try {
    const { data, error } = await fetcher();
    if (error) {
      console.warn(`[analysis-history] ${label}`, error.message);
      return null;
    }
    return data;
  } catch (err) {
    console.warn(`[analysis-history] ${label}`, err);
    return null;
  }
}

export function liveChatToHistoryItems(
  messages: Array<{ id: string; role: string; content: string }>,
  author?: { name: string; initials: string | null },
): AnalysisHistoryEntry[] {
  const now = Date.now();
  const userAct = author ? userActorFor(author) : userActorFor({ name: "You", initials: null });
  return messages
    .filter((m) => m.role === "user" || m.role === "ai" || m.role === "assistant")
    .map((m, index) => {
      const isUser = m.role === "user";
      const actor = isUser ? userAct : aiActor();
      const preview = truncate(m.content);
      return {
        id: `live-${m.id}`,
        analysis_id: "",
        ...actor,
        description: isUser
          ? `Sent a message: "${preview}"`
          : preview
            ? `Replied: "${preview}"`
            : "Avaloka AI replied in chat.",
        created_at: new Date(now - (messages.length - index) * 1000).toISOString(),
        output: { content: m.content },
      };
    });
}

function mergeHistoryEntries(persisted: AnalysisHistoryEntry[], live: AnalysisHistoryEntry[]): AnalysisHistoryEntry[] {
  if (!live.length) return persisted;
  if (!persisted.length) return live;

  const persistedContents = new Set(
    persisted
      .map((e) => {
        const out = e.output as { content?: string } | null;
        return out?.content ?? e.description;
      })
      .filter(Boolean),
  );

  const supplemental = live.filter((e) => {
    const out = e.output as { content?: string } | null;
    const key = out?.content ?? e.description;
    return key && !persistedContents.has(key);
  });

  return [...persisted, ...supplemental];
}

export async function loadAnalysisHistory(
  analysisId: string,
  liveChat?: Array<{ id: string; role: string; content: string }>,
): Promise<AnalysisHistoryEntry[]> {
  const author = await resolveCurrentAuthor();

  let messages: AnalysisMessage[] = [];
  try {
    messages = await loadMessages(analysisId);
  } catch (err) {
    console.warn("[analysis-history] messages", err);
  }

  const loadedRow = await loadAnalysisRowForHistory(analysisId);
  const row = withAuthor(loadedRow ?? fallbackAnalysisRow(analysisId, author), author);
  const chatMessages = messages.filter(isChatTurn);

  const [collaborators, shareLink, insightComments] = await Promise.all([
    loadOptional("resource_collaborators", () => {
      const resourceType = row.project_id ? "project" : "analysis";
      const resourceId = row.project_id ?? analysisId;
      return supabase
        .from("resource_collaborators")
        .select("id,created_at,department,access_level,app_user_id,app_users(name,email)")
        .eq("resource_type", resourceType)
        .eq("resource_id", resourceId)
        .order("created_at", { ascending: true });
    }),
    loadOptional("resource_share_links", () => {
      const resourceType = row.project_id ? "project" : "analysis";
      const resourceId = row.project_id ?? analysisId;
      return supabase
        .from("resource_share_links")
        .select("id,created_at,updated_at,link_access,share_token")
        .eq("resource_type", resourceType)
        .eq("resource_id", resourceId)
        .maybeSingle();
    }),
    loadOptional("analysis_comments", () =>
      supabase
        .from("analysis_comments" as never)
        .select("id,content,created_at,parent_id,author:profiles!author_id(full_name,avatar_url)")
        .eq("analysis_id", analysisId)
        .order("created_at", { ascending: true }),
    ),
  ]);

  const hasMessageViz = chatMessages.some((m) => m.role === "assistant" && hasViz(messageOutput(m)));

  const dashboardWidgets = dashboardWidgetsForHistory(row.viz_config) as unknown as import("@/lib/api/project-dashboard").DashboardWidget[] & DashboardGraphRow[];

  const entries: AnalysisHistoryEntry[] = [
    ...entriesFromAnalysisRow(row, hasMessageViz),
    ...entriesFromChatMessages(chatMessages, row),
    ...entriesFromInsightComments(analysisId, (insightComments ?? []) as unknown as InsightCommentRow[], row),
    ...entriesFromDashboardGraphs(analysisId, dashboardWidgets, row),
    ...entriesFromCollaborators(analysisId, (collaborators ?? []) as unknown as CollaboratorRow[], row),
  ];

  const shareLinkRow = shareLink as ShareLinkRow | null;
  if (shareLinkRow) {
    entries.push(...entriesFromShareLink(analysisId, shareLinkRow, row));
  }

  const merged = mergeHistoryEntries(entries, liveChat ? liveChatToHistoryItems(liveChat, author) : []);
  merged.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  return merged;
}
