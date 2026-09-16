// Key-insight comments for the analysis page comment card + sidebar panel.
// Backed by public.analysis_comments (author_id → profiles.id).

import { supabase } from "@/integrations/supabase/client";
import { requireCurrentProfileId } from "@/lib/current-profile";
import { getCachedAuthUser } from "@/lib/auth-user";

export const INSIGHT_COMMENT_ATTACHMENT_BUCKET = "analysis-comment-attachments";
export const MAX_INSIGHT_COMMENT_ATTACHMENT_BYTES = 25 * 1024 * 1024;
export const MAX_INSIGHT_COMMENT_ATTACHMENTS = 5;

export type InsightCommentAttachment = {
  name: string;
  path: string;
  mime_type: string;
  size: number;
};

export type InsightCommentAuthor = {
  id: string;
  full_name: string | null;
  avatar_url: string | null;
};

export type AnalysisInsightCommentReply = {
  id: string;
  text: string;
  created_at: string;
  author: InsightCommentAuthor | null;
  attachments: InsightCommentAttachment[];
};

export type AnalysisInsightComment = AnalysisInsightCommentReply & {
  replies: AnalysisInsightCommentReply[];
};

export function initialsFromName(name: string | null | undefined): string | null {
  if (!name) return null;
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() ?? "").join("") || null;
}

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
  content: string;
  created_at: string;
  parent_id: string | null;
  attachments: unknown;
  author: InsightCommentAuthor | InsightCommentAuthor[] | null;
};

const SELECT_WITH_AUTHOR =
  "id,content,created_at,parent_id,attachments,analysis_id,author_id,author:profiles!author_id(id,full_name,avatar_url)";

function normalizeAuthor(a: Row["author"]): InsightCommentAuthor | null {
  if (!a) return null;
  const obj = Array.isArray(a) ? (a[0] ?? null) : a;
  if (!obj) return null;
  return { id: obj.id, full_name: obj.full_name ?? null, avatar_url: obj.avatar_url ?? null };
}

async function resolveAvatarUrl(value: string | null): Promise<string | null> {
  if (!value) return null;

  const publicMarker = "/storage/v1/object/public/avatars/";
  if (/^https?:\/\//i.test(value) && !value.includes(publicMarker)) return value;

  const rawPath = value.includes(publicMarker)
    ? value.split(publicMarker)[1]
    : value.replace(/^avatars\//, "");
  const path = decodeURIComponent(rawPath.split("?")[0] ?? "");
  if (!path) return null;

  const { data, error } = await supabase.storage.from("avatars").createSignedUrl(path, 60 * 60);
  if (error) {
    console.warn("[insight-comments] avatar signed url failed", error.message);
    return null;
  }
  return data.signedUrl;
}

async function resolveRowAvatars(rows: Row[]): Promise<Row[]> {
  const resolved = new Map<string, string | null>();
  const avatarValues = new Set<string>();

  for (const row of rows) {
    const author = normalizeAuthor(row.author);
    if (author?.avatar_url) avatarValues.add(author.avatar_url);
  }

  await Promise.all(
    [...avatarValues].map(async (value) => {
      resolved.set(value, await resolveAvatarUrl(value));
    }),
  );

  return rows.map((row) => {
    const author = normalizeAuthor(row.author);
    if (!author?.avatar_url) return row;
    return { ...row, author: { ...author, avatar_url: resolved.get(author.avatar_url) ?? null } };
  });
}

function toReply(r: Row): AnalysisInsightCommentReply {
  return {
    id: r.id,
    text: r.content,
    created_at: r.created_at,
    author: normalizeAuthor(r.author),
    attachments: parseAttachments(r.attachments),
  };
}

/** Load insight comments (top-level newest first + nested replies oldest→newest). */
export async function loadAnalysisInsightComments(
  analysisId: string,
): Promise<AnalysisInsightComment[]> {
  const { data, error } = await supabase
    .from("analysis_comments" as never)
    .select(SELECT_WITH_AUTHOR)
    .eq("analysis_id", analysisId)
    .order("created_at", { ascending: true });

  if (error) throw error;
  const rows = await resolveRowAvatars((data ?? []) as unknown as Row[]);

  const map = new Map<string, AnalysisInsightComment>();
  const topLevel: AnalysisInsightComment[] = [];

  for (const r of rows) {
    if (!r.parent_id) {
      const entry = { ...toReply(r), replies: [] as AnalysisInsightCommentReply[] };
      map.set(r.id, entry);
      topLevel.push(entry);
    }
  }
  for (const r of rows) {
    if (r.parent_id && map.has(r.parent_id)) {
      map.get(r.parent_id)!.replies.push(toReply(r));
    }
  }
  // Newest top-level first; replies remain chronological inside each thread.
  return topLevel.reverse();
}

export async function uploadInsightCommentAttachment(
  analysisId: string,
  file: File,
): Promise<InsightCommentAttachment> {
  if (file.size > MAX_INSIGHT_COMMENT_ATTACHMENT_BYTES) {
    throw new Error("File is too large. Maximum size is 25MB.");
  }

  const data = { user: await getCachedAuthUser() };
  const user = data.user;
  if (!user) throw new Error("Not signed in");

  const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, "_") || "file";
  const path = `${user.id}/${analysisId}/${Date.now()}-${safeName}`;

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

export async function getInsightCommentAttachmentUrl(path: string): Promise<string | null> {
  const { data, error } = await supabase.storage
    .from(INSIGHT_COMMENT_ATTACHMENT_BUCKET)
    .createSignedUrl(path, 60 * 60);
  if (error) {
    console.warn("[insight-comments] signed url failed", error.message);
    return null;
  }
  return data.signedUrl;
}

async function insertComment(
  payload: Record<string, unknown>,
  mentionedProfileIds: string[] = [],
): Promise<Row> {
  const { data, error } = await supabase
    .from("analysis_comments" as never)
    .insert(payload as never)
    .select(SELECT_WITH_AUTHOR)
    .single();
  if (error) throw error;
  const [row] = await resolveRowAvatars([data as unknown as Row]);
  // Fire-and-forget: notify analysis owner + collaborators (excluding author).
  // Run on the server with the admin client so RLS on profiles / app_users /
  // resource_collaborators can never silently drop recipients (which was
  // causing collaborators to miss comment notifications).
  (async () => {
    try {
      const { fanOutCommentNotifications } = await import("@/lib/comment-notify.functions");
      await fanOutCommentNotifications({ data: { commentId: row.id, mentionedProfileIds } });
    } catch (e) {
      console.warn("[insight-comments] server fan-out failed, falling back", e);
      try {
        await fanOutCommentNotification(row);
      } catch (e2) {
        console.warn("[insight-comments] fallback fan-out failed", e2);
      }
    }
  })();
  return row;
}

async function resolveAppUsersToProfileIds(appUserIds: string[]): Promise<string[]> {
  if (!appUserIds.length) return [];
  const { data: appUsers, error: appUsersError } = await supabase
    .from("app_users")
    .select("auth_user_id")
    .in("id", appUserIds);
  if (appUsersError) throw appUsersError;
  const authUserIds = ((appUsers ?? []) as { auth_user_id: string | null }[])
    .map((u) => u.auth_user_id)
    .filter((v): v is string => !!v);
  if (!authUserIds.length) return [];

  const [
    { data: linkedProfiles, error: linkedError },
    { data: legacyProfiles, error: legacyError },
  ] = await Promise.all([
    supabase.from("profiles").select("id").in("user_id", authUserIds),
    supabase.from("profiles").select("id").in("id", authUserIds),
  ]);
  if (linkedError) throw linkedError;
  if (legacyError) throw legacyError;

  return [
    ...((linkedProfiles ?? []) as { id: string }[]),
    ...((legacyProfiles ?? []) as { id: string }[]),
  ].map((p) => p.id);
}

async function collectResourceCollaboratorProfileIds(
  resourceType: "analysis" | "project",
  resourceId: string,
): Promise<string[]> {
  const { data: collabs } = await supabase
    .from("resource_collaborators")
    .select("app_user_id")
    .eq("resource_type", resourceType)
    .eq("resource_id", resourceId);
  const appUserIds = ((collabs ?? []) as { app_user_id: string }[])
    .map((c) => c.app_user_id)
    .filter((v): v is string => !!v);
  return resolveAppUsersToProfileIds(appUserIds);
}

async function fanOutCommentNotification(row: Row): Promise<void> {
  const { notifyMany, notify } = await import("@/lib/notifications");
  const analysisId = (row as unknown as { analysis_id: string }).analysis_id;
  const authorId = (row as unknown as { author_id: string }).author_id;
  const parentId = (row as unknown as { parent_id: string | null }).parent_id ?? null;
  const content = (row as unknown as { content: string }).content ?? "";
  if (!analysisId) return;

  const { data: analysis } = await supabase
    .from("analyses")
    .select("owner_id,name,project_id")
    .eq("id", analysisId)
    .maybeSingle();
  const recipients = new Set<string>();
  if (analysis?.owner_id) recipients.add(analysis.owner_id as string);

  for (const id of await collectResourceCollaboratorProfileIds("analysis", analysisId)) {
    recipients.add(id);
  }

  const projectId = (analysis as { project_id?: string | null } | null)?.project_id ?? null;
  if (projectId) {
    const { data: project } = await supabase
      .from("projects")
      .select("owner_id")
      .eq("id", projectId)
      .maybeSingle();
    if (project?.owner_id) recipients.add(project.owner_id as string);
    for (const id of await collectResourceCollaboratorProfileIds("project", projectId)) {
      recipients.add(id);
    }
  }

  const author = (row as unknown as { author: { full_name: string | null } | null }).author;
  const authorName = author?.full_name?.trim() || "Someone";
  const preview = content.length > 120 ? content.slice(0, 117) + "…" : content;
  const analysisName = analysis?.name ?? "your analysis";
  const link = `/analysis?id=${analysisId}`;

  // If this is a reply, send a dedicated notification to the parent comment's author.
  if (parentId) {
    const { data: parent } = await supabase
      .from("analysis_comments" as never)
      .select("author_id")
      .eq("id", parentId)
      .maybeSingle();
    const parentAuthorId = (parent as { author_id?: string } | null)?.author_id ?? null;
    if (parentAuthorId && parentAuthorId !== authorId) {
      await notify({
        recipientId: parentAuthorId,
        type: "comment.reply",
        title: `${authorName} replied to your comment`,
        body: preview,
        resourceType: "analysis",
        resourceId: analysisId,
        link,
      });
      recipients.delete(parentAuthorId);
    }
  }

  recipients.delete(authorId);
  const list = [...recipients];
  if (!list.length) return;

  await notifyMany(list, {
    type: parentId ? "comment.reply" : "comment.created",
    title: parentId
      ? `${authorName} replied on "${analysisName}"`
      : `${authorName} commented on "${analysisName}"`,
    body: preview,
    resourceType: "analysis",
    resourceId: analysisId,
    link,
  });
}


/** Add a top-level insight comment. */
export async function addAnalysisInsightComment(
  analysisId: string,
  text: string,
  attachments: InsightCommentAttachment[] = [],
  mentionedProfileIds: string[] = [],
): Promise<AnalysisInsightComment> {
  const authorId = await requireCurrentProfileId();
  const row = await insertComment({
    analysis_id: analysisId,
    author_id: authorId,
    content: text,
    parent_id: null,
    attachments,
  }, mentionedProfileIds);
  return { ...toReply(row), replies: [] };
}

/** Add a reply to an existing insight comment. */
export async function addAnalysisInsightCommentReply(
  analysisId: string,
  parentId: string,
  text: string,
  attachments: InsightCommentAttachment[] = [],
  mentionedProfileIds: string[] = [],
): Promise<AnalysisInsightCommentReply> {
  const authorId = await requireCurrentProfileId();
  const row = await insertComment({
    analysis_id: analysisId,
    author_id: authorId,
    parent_id: parentId,
    content: text,
    attachments,
  }, mentionedProfileIds);
  return toReply(row);
}

export function formatInsightCommentTime(iso: string): string {
  return new Date(iso).toLocaleString([], {
    weekday: "long",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatInsightCommentRelative(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "Just now";
  if (m < 60) return `${m} min${m === 1 ? "" : "s"} ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.floor(h / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

export function formatAttachmentSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
