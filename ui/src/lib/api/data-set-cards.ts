import { supabase } from "@/integrations/supabase/client";
import { getCurrentProfileId } from "@/lib/current-profile";
import type { Provider } from "@/lib/data-sources";
import { getCachedAuthUser } from "@/lib/auth-user";

export type DataSetCard = {
  id: string;
  title: string;
  useCase: string;
  provider: string;
  providerColor: string;
  providerKey: Provider | null;
  author: string;
  authorInitials: string;
  updated: string;
  createdAt: string;
  icon: "folder" | "dollar" | "image";
  fileCount: number;
  connectionType: "live" | "synced";
  source: "cloud" | "database" | "upload";
  dataSourceId?: string;
  analysisId?: string;
};

const PROVIDER_META: Record<string, { label: string; color: string; key: Provider | null }> = {
  mysql: { label: "MySQL", color: "#00758f", key: "mysql" },
  postgres: { label: "PostgreSQL", color: "#336791", key: "postgres" },
  postgresql: { label: "PostgreSQL", color: "#336791", key: "postgres" },
  snowflake: { label: "Snowflake", color: "#29b5e8", key: "snowflake" },
  clickhouse: { label: "ClickHouse", color: "#fdc328", key: "clickhouse" },
  mariadb: { label: "MariaDB", color: "#003545", key: "mariadb" },
  bigquery: { label: "BigQuery", color: "#4285f4", key: "bigquery" },
  gcp: { label: "GCP", color: "#4285f4", key: "bigquery" },
  gcs: { label: "GCS", color: "#4285f4", key: "bigquery" },
  s3: { label: "S3", color: "#ff9900", key: null },
  azure: { label: "Azure", color: "#0078d4", key: null },
  databricks: { label: "Databricks", color: "#ff3621", key: "databricks" },
  mssql: { label: "SQL Server", color: "#cc2927", key: "mssql" },
  trino: { label: "Trino", color: "#dd00a1", key: "trino" },
  presto: { label: "Presto", color: "#dd00a1", key: "presto" },
  mongodb: { label: "MongoDB", color: "#00684a", key: "mongodb" },
  cloudsql: { label: "Cloud SQL", color: "#4285f4", key: "cloudsql" },
  csv: { label: "CSV", color: "#079455", key: null },
  xlsx: { label: "XLS", color: "#079455", key: null },
  xls: { label: "XLS", color: "#079455", key: null },
  json: { label: "JSON", color: "#1565ef", key: null },
  file: { label: "FILE", color: "#475467", key: null },
};

function providerMeta(raw: string) {
  const key = raw.toLowerCase();
  return (
    PROVIDER_META[key] ?? {
      label: raw.toUpperCase() || "Dataset",
      color: "#475467",
      key: null,
    }
  );
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function initialsFrom(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

export const dataSetCardsKey = ["data-set-cards"] as const;

export async function fetchDataSetCards(): Promise<DataSetCard[]> {
  const userRes = { user: await getCachedAuthUser() };
  const user = userRes.user;
  const userId = user?.id;
  if (!userId) return [];

  let authorName = user?.email?.split("@")[0] ?? "You";
  const { data: profile } = await supabase
    .from("profiles")
    .select("first_name,last_name")
    .eq("id", userId)
    .maybeSingle();
  if (profile) {
    const full = [profile.first_name, profile.last_name].filter(Boolean).join(" ");
    if (full) authorName = full;
  }
  const authorInitials = initialsFrom(authorName);

  const cards: DataSetCard[] = [];

  const { data: cloudRows, error: cloudErr } = await supabase
    .from("cloud_datasets")
    .select("id,name,provider,created_at,updated_at,schema,connection_status")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });
  if (cloudErr) {
    console.warn("[data-set-cards] cloud_datasets fetch failed", cloudErr);
  } else {
    for (const row of cloudRows ?? []) {
      const schema = (row.schema as Record<string, unknown>) ?? {};
      const files = (schema.selected_files as string[]) ?? [];
      const meta = providerMeta(row.provider ?? "gcp");
      const live = row.connection_status === "connected";
      const useCase =
        (typeof schema.use_cases === "string" && schema.use_cases) ||
        (typeof schema.domain === "string" && schema.domain) ||
        "Cloud data connection";
      const createdAt = row.updated_at ?? row.created_at;
      cards.push({
        id: `cloud-${row.id}`,
        title: row.name,
        useCase,
        provider: meta.label,
        providerColor: meta.color,
        providerKey: meta.key,
        author: authorName,
        authorInitials,
        updated: formatDate(createdAt),
        createdAt,
        icon: row.provider === "bigquery" || row.provider === "gcp" ? "dollar" : "folder",
        fileCount: Math.max(files.length, 1),
        connectionType: live ? "live" : "synced",
        source: "cloud",
        dataSourceId: row.id,
      });
    }
  }

  const { data: dbRows, error: dbErr } = await supabase
    .from("data_sources")
    .select("id,name,provider,use_cases,domain,status,created_at,updated_at,config")
    .eq("owner_id", userId)
    .order("created_at", { ascending: false });
  if (dbErr) {
    console.warn("[data-set-cards] data_sources fetch failed", dbErr);
  } else {
    for (const row of dbRows ?? []) {
      const meta = providerMeta(row.provider ?? "mysql");
      const config = (row.config as Record<string, unknown>) ?? {};
      const tables = (config.tables as unknown[]) ?? [];
      const live = row.status === "connected";
      const createdAt = row.updated_at ?? row.created_at;
      cards.push({
        id: `db-${row.id}`,
        title: row.name,
        useCase: row.use_cases || row.domain || "Database connection",
        provider: meta.label,
        providerColor: meta.color,
        providerKey: meta.key,
        author: authorName,
        authorInitials,
        updated: formatDate(createdAt),
        createdAt,
        icon: "folder",
        fileCount: Math.max(tables.length, 1),
        connectionType: live ? "live" : "synced",
        source: "database",
        dataSourceId: row.id,
      });
    }
  }

  const profileId = await getCurrentProfileId();
  const { data: uploads, error: uploadErr } = await supabase
    .from("analyses")
    .select("id,name,filename,created_at,updated_at")
    .eq("owner_id", profileId ?? userId)
    .not("filename", "is", null)
    .order("created_at", { ascending: false })
    .limit(50);
  if (uploadErr) {
    console.warn("[data-set-cards] analyses fetch failed", uploadErr);
  } else {
    for (const row of uploads ?? []) {
      if (!row.filename) continue;
      const ext = row.filename.includes(".") ? (row.filename.split(".").pop() ?? "file") : "file";
      const meta = providerMeta(ext);
      const createdAt = row.updated_at ?? row.created_at;
      cards.push({
        id: `upload-${row.id}`,
        title: row.name || row.filename,
        useCase: "Uploaded file analysis",
        provider: meta.label,
        providerColor: meta.color,
        providerKey: meta.key,
        author: authorName,
        authorInitials,
        updated: formatDate(createdAt),
        createdAt,
        icon: "image",
        fileCount: 1,
        connectionType: "synced",
        source: "upload",
        analysisId: row.id,
      });
    }
  }

  return cards.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
}
