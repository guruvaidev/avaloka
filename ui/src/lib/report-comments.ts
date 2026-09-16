// Report comments (public.report_comments) — mirrors analysis-insight-comments
// but scoped to a report. author_id → profiles.id.

import { supabase } from "@/integrations/supabase/client";
import { requireCurrentProfileId } from "@/lib/current-profile";
import { getCachedAuthUser } from "@/lib/auth-user";
import {
  INSIGHT_COMMENT_ATTACHMENT_BUCKET,
  MAX_INSIGHT_COMMENT_ATTACHMENT_BYTES,
  type InsightCommentAttachment,
  type InsightCommentAuthor,
  type AnalysisInsightComment,
  type AnalysisInsightCommentReply,
} from "@/lib/analysis-insight-comments";

export type ReportComment = AnalysisInsightComment;
export type ReportCommentReply = AnalysisInsightCommentReply;

function parseAttachments(raw: unknown): InsightCommentAttachment[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((item): item is InsightCommentAttachment => {
      if (!item || typeof item !== "object") return false;
      const a = item as InsightCommentAttachment;
      return typeof a.name === "string" && typeof a.path === "string";
    })
    .map((a) => ({
      name: a.name,
      path: a.path,
      mime_type: a.mime_type ?? "application/octet-stream",
      size: typeof a.size === "number" ? a.size : 0,
    }));
}

type Row = {
  id: string;
  body: string;
  created_at: string;
  parent_id: string | null;
  attachments: unknown;
  report_id: string;
  author_id: string;
  author: InsightCommentAuthor | InsightCommentAuthor[] | null;
};

const SELECT =
  "id,body,created_at,parent_id,attachments,report_id,author_id,author:profiles!author_id(id,full_name,avatar_url)";

function normalizeAuthor(a: Row["author"]): InsightCommentAuthor | null {
  if (!a) return null;
  const obj = Array.isArray(a) ? (a[0] ?? null) : a;
  if (!obj) return null;
  return { id: obj.id, full_name: obj.full_name ?? null, avatar_url: obj.avatar_url ?? null };
}

async function resolveAvatarUrl(value: string | null): Promise<string | null> {
  if (!value) return null;
  const marker = "/storage/v1/object/public/avatars/";
  if (/^https?:\/\//i.test(value) && !value.includes(marker)) return value;
  const rawPath = value.includes(marker) ? value.split(marker)[1] : value.replace(/^avatars\//, "");
  const path = decodeURIComponent(rawPath.split("?")[0] ?? "");
  if (!path) return null;
  const { data, error } = await supabase.storage.from("avatars").createSignedUrl(path, 60 * 60);
  if (error) return null;
  return data.signedUrl;
}

async function resolveRowAvatars(rows: Row[]): Promise<Row[]> {
  const resolved = new Map<string, string | null>();
  const values = new Set<string>();
  for (const row of rows) {
    const a = normalizeAuthor(row.author);
    if (a?.avatar_url) values.add(a.avatar_url);
  }
  await Promise.all(
    [...values].map(async (v) => resolved.set(v, await resolveAvatarUrl(v))),
  );
  return rows.map((row) => {
    const a = normalizeAuthor(row.author);
    if (!a?.avatar_url) return row;
    return { ...row, author: { ...a, avatar_url: resolved.get(a.avatar_url) ?? null } };
  });
}

function toReply(r: Row): ReportCommentReply {
  return {
    id: r.id,
    text: r.body,
    created_at: r.created_at,
    author: normalizeAuthor(r.author),
    attachments: parseAttachments(r.attachments),
  };
}

export async function loadReportComments(reportId: string): Promise<ReportComment[]> {
  const { data, error } = await supabase
    .from("report_comments" as never)
    .select(SELECT)
    .eq("report_id", reportId)
    .order("created_at", { ascending: true });
  if (error) throw error;
  const rows = await resolveRowAvatars((data ?? []) as unknown as Row[]);

  const map = new Map<string, ReportComment>();
  const topLevel: ReportComment[] = [];
  for (const r of rows) {
    if (!r.parent_id) {
      const entry = { ...toReply(r), replies: [] as ReportCommentReply[] };
      map.set(r.id, entry);
      topLevel.push(entry);
    }
  }
  for (const r of rows) {
    if (r.parent_id && map.has(r.parent_id)) {
      map.get(r.parent_id)!.replies.push(toReply(r));
    }
  }
  return topLevel.reverse();
}

export async function uploadReportCommentAttachment(
  reportId: string,
  file: File,
): Promise<InsightCommentAttachment> {
  if (file.size > MAX_INSIGHT_COMMENT_ATTACHMENT_BYTES) {
    throw new Error("File is too large. Maximum size is 25MB.");
  }
  const data = { user: await getCachedAuthUser() };
  const user = data.user;
  if (!user) throw new Error("Not signed in");
  const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_") || "file";
  const path = `${user.id}/reports/${reportId}/${Date.now()}-${safeName}`;
  const { error } = await supabase.storage
    .from(INSIGHT_COMMENT_ATTACHMENT_BUCKET)
    .upload(path, file, { upsert: false, contentType: file.type || undefined });
  if (error) throw error;
  return {
    name: file.name,
    path,
    mime_type: file.type || "application/octet-stream",
    size: file.size,
  };
}

async function insertComment(payload: {
  reportId: string;
  parentId: string | null;
  content: string;
  attachments: InsightCommentAttachment[];
}, mentionedProfileIds: string[] = []): Promise<Row> {
  const { createReportComment } = await import("@/lib/report-comment-create.functions");
  const data = await createReportComment({ data: payload });
  const [row] = await resolveRowAvatars([data as unknown as Row]);
  (async () => {
    try {
      const { fanOutReportCommentNotifications } = await import(
        "@/lib/report-comment-notify.functions"
      );
      await fanOutReportCommentNotifications({ data: { commentId: row.id, mentionedProfileIds } });
    } catch (e) {
      console.warn("[report-comments] fan-out failed", e);
    }
  })();
  return row;
}

export async function addReportComment(
  reportId: string,
  text: string,
  attachments: InsightCommentAttachment[] = [],
  mentionedProfileIds: string[] = [],
): Promise<ReportComment> {
  await requireCurrentProfileId();
  const row = await insertComment({
    reportId,
    content: text,
    parentId: null,
    attachments,
  }, mentionedProfileIds);
  return { ...toReply(row), replies: [] };
}

export async function addReportCommentReply(
  reportId: string,
  parentId: string,
  text: string,
  attachments: InsightCommentAttachment[] = [],
  mentionedProfileIds: string[] = [],
): Promise<ReportCommentReply> {
  await requireCurrentProfileId();
  const row = await insertComment({
    reportId,
    parentId,
    content: text,
    attachments,
  }, mentionedProfileIds);
  return toReply(row);
}
