import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";

export type Provider =
  | "mysql"
  | "postgres"
  | "snowflake"
  | "mssql"
  | "clickhouse"
  | "mariadb"
  | "bigquery"
  | "trino"
  | "presto"
  | "mongodb"
  | "databricks"
  | "cloudsql";
export type DataSourceStatus = "connected" | "disconnected" | "error";
export type StorageProvider = "aws" | "azure" | "gcp";
export type ConnectionType = "storage" | "database";

export const STORAGE_PROVIDER_LABELS: Record<StorageProvider, string> = {
  aws: "Amazon S3",
  azure: "Azure Blob Storage",
  gcp: "Google Cloud Storage",
};

export type DataSource = {
  id: string;
  provider: Provider;
  connectionType: ConnectionType;
  storageProvider?: StorageProvider;
  name: string;
  description: string | null;
  domain: string | null;
  use_cases: string | null;
  status: DataSourceStatus;
  config: Record<string, unknown>;
  last_tested_at: string | null;
  last_error: string | null;
  created_at: string;
};

export type DataSourceFile = {
  id: string;
  data_source_id: string;
  path: string;
  size_bytes: number | null;
  selected: boolean;
  uploaded: boolean;
  created_at: string;
};

type CloudDatasetRow = {
  id: string;
  user_id: string;
  name: string;
  provider: string;
  region: string | null;
  bucket_name: string | null;
  connection_status: string | null;
  last_tested_at: string | null;
  created_at: string;
  schema: Record<string, unknown> | null;
};

const KEY = ["data_sources"] as const;

async function userId() {
  const data = { user: await getCachedAuthUser() };
  return data.user?.id ?? null;
}

function mapProvider(raw: string): Provider {
  if (raw === "gcp") return "bigquery";
  return raw as Provider;
}

function mapStatus(raw: string | null): DataSourceStatus {
  if (raw === "connected") return "connected";
  if (raw === "error") return "error";
  return "disconnected";
}

function mapCloudDataset(row: CloudDatasetRow): DataSource {
  const schema = row.schema ?? {};
  const rawProvider = (row.provider ?? "").toLowerCase();
  const storageProvider: StorageProvider | undefined =
    rawProvider === "aws" || rawProvider === "s3"
      ? "aws"
      : rawProvider === "azure" || rawProvider === "azure_blob"
        ? "azure"
        : rawProvider === "gcp" || rawProvider === "gcs"
          ? "gcp"
          : undefined;
  return {
    id: row.id,
    provider: mapProvider(row.provider),
    connectionType: "storage",
    storageProvider,
    name: row.name,
    description: (schema.description as string) ?? null,
    domain: (schema.domain as string) ?? null,
    use_cases: (schema.use_cases as string) ?? null,
    status: mapStatus(row.connection_status),
    config: {
      connection_id: row.id,
      bucket_name: row.bucket_name,
      region: row.region,
      selected_files: schema.selected_files ?? [],
    },
    last_tested_at: row.last_tested_at,
    last_error: null,
    created_at: row.created_at,
  };
}

type McpConnectionRow = {
  id: string;
  connection_name: string | null;
  customer_name: string | null;
  database_type: string | null;
  host: string | null;
  port: number | null;
  database_name: string | null;
  status: string | null;
  created_at: string;
};

function mapDatabaseProvider(raw: string | null): Provider {
  const key = (raw ?? "").toLowerCase();
  if (key === "postgresql" || key === "postgres") return "postgres";
  if (key === "sqlserver" || key === "mssql") return "mssql";
  return (key || "postgres") as Provider;
}

function mapMcpConnection(row: McpConnectionRow): DataSource {
  return {
    id: row.id,
    provider: mapDatabaseProvider(row.database_type),
    connectionType: "database",
    name: row.connection_name || row.database_name || "Database connection",
    description: null,
    domain: row.database_name ?? null,
    use_cases: row.host ? `${row.host}${row.port ? `:${row.port}` : ""}` : null,
    status: mapStatus((row.status ?? "").toLowerCase() === "active" ? "connected" : row.status),
    config: {
      connection_id: row.id,
      database_name: row.database_name,
      host: row.host,
      port: row.port,
      selected_files: [],
    },
    last_tested_at: null,
    last_error: null,
    created_at: row.created_at,
  };
}

export function useDataSources() {
  return useQuery({
    queryKey: KEY,
    queryFn: async () => {
      const uid = await userId();
      if (!uid) return [];
      const [cloudRes, dbRes] = await Promise.all([
        supabase
          .from("cloud_datasets")
          .select("*")
          .eq("user_id", uid)
          .order("created_at", { ascending: false }),
        (supabase.from as any)("mcp_connections")
          .select("id,connection_name,customer_name,database_type,host,port,database_name,status,created_at")
          .eq("user_id", uid)
          .order("created_at", { ascending: false }),
      ]);
      if (cloudRes.error) throw cloudRes.error;
      const cloud = ((cloudRes.data ?? []) as CloudDatasetRow[]).map(mapCloudDataset);
      if (dbRes.error) {
        console.warn("[data-sources] mcp_connections fetch failed", dbRes.error);
        return cloud;
      }
      const dbs = ((dbRes.data ?? []) as McpConnectionRow[]).map(mapMcpConnection);
      return [...cloud, ...dbs].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      );
    },
  });
}

export function useDataSourceFiles(dataSourceId: string | null) {
  return useQuery({
    enabled: !!dataSourceId,
    queryKey: ["data_source_files", dataSourceId],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("cloud_datasets")
        .select("schema")
        .eq("id", dataSourceId!)
        .single();
      if (error) throw error;
      const schema = (data?.schema as Record<string, unknown>) ?? {};
      const paths = (schema.selected_files as string[]) ?? [];
      return paths.map((path, i) => ({
        id: `${dataSourceId}-${i}`,
        data_source_id: dataSourceId!,
        path,
        size_bytes: null,
        selected: true,
        uploaded: true,
        created_at: new Date().toISOString(),
      })) as DataSourceFile[];
    },
  });
}

export function useDataSourceMutations() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: KEY });

  const create = useMutation({
    mutationFn: async (input: {
      provider: Provider;
      name: string;
      description?: string | null;
      domain?: string | null;
      use_cases?: string | null;
      config: Record<string, unknown>;
      status?: DataSourceStatus;
    }) => {
      const uid = await userId();
      if (!uid) throw new Error("Not signed in");
      const connectionId = input.config.connection_id as string | undefined;
      if (!connectionId) {
        throw new Error("Missing cloud connection id. Authenticate again.");
      }
      const { data, error } = await supabase
        .from("cloud_datasets")
        .update({
          name: input.name,
          connection_status: input.status ?? "connected",
          last_tested_at: new Date().toISOString(),
          schema: {
            description: input.description ?? null,
            domain: input.domain ?? null,
            use_cases: input.use_cases ?? null,
            selected_files: (input.config.selected_files as string[]) ?? [],
          },
        })
        .eq("id", connectionId)
        .select("*")
        .single();
      if (error) throw error;
      return mapCloudDataset(data as CloudDatasetRow);
    },
    onSuccess: invalidate,
  });

  const update = useMutation({
    mutationFn: async ({
      id,
      patch,
      type = "storage",
    }: {
      id: string;
      patch: Partial<DataSource>;
      type?: ConnectionType;
    }) => {
      if (type === "database") {
        const row: Record<string, unknown> = {};
        if (patch.name) row.connection_name = patch.name;
        if (patch.status) row.status = patch.status === "connected" ? "active" : "inactive";
        const { error } = await (supabase.from as any)("mcp_connections").update(row).eq("id", id);
        if (error) throw error;
        return;
      }
      const row: Record<string, unknown> = {};
      if (patch.name) row.name = patch.name;
      if (patch.status) row.connection_status = patch.status;
      if (patch.last_tested_at) row.last_tested_at = patch.last_tested_at;
      if (
        patch.description !== undefined ||
        patch.domain !== undefined ||
        patch.use_cases !== undefined
      ) {
        const { data: existing } = await supabase
          .from("cloud_datasets")
          .select("schema")
          .eq("id", id)
          .single();
        const schema = (existing?.schema as Record<string, unknown>) ?? {};
        if (patch.description !== undefined) schema.description = patch.description;
        if (patch.domain !== undefined) schema.domain = patch.domain;
        if (patch.use_cases !== undefined) schema.use_cases = patch.use_cases;
        row.schema = schema;
      }
      const { error } = await supabase.from("cloud_datasets").update(row).eq("id", id);
      if (error) throw error;
    },
    onSuccess: invalidate,
  });

  const remove = useMutation({
    mutationFn: async (input: string | { id: string; type?: ConnectionType }) => {
      const id = typeof input === "string" ? input : input.id;
      const type = typeof input === "string" ? "storage" : (input.type ?? "storage");
      if (type === "database") {
        const { error } = await (supabase.from as any)("mcp_connections").delete().eq("id", id);
        if (error) throw error;
        return;
      }
      const { error } = await supabase.from("cloud_datasets").delete().eq("id", id);
      if (error) throw error;
    },
    onSuccess: invalidate,
  });


  const insertFiles = useMutation({
    mutationFn: async ({
      dataSourceId,
      files,
    }: {
      dataSourceId: string;
      files: { path: string; size_bytes?: number; selected?: boolean; uploaded?: boolean }[];
    }) => {
      const { data: existing, error: fetchErr } = await supabase
        .from("cloud_datasets")
        .select("schema")
        .eq("id", dataSourceId)
        .single();
      if (fetchErr) throw fetchErr;
      const schema = (existing?.schema as Record<string, unknown>) ?? {};
      schema.selected_files = files.map((f) => f.path);
      const { error } = await supabase
        .from("cloud_datasets")
        .update({ schema })
        .eq("id", dataSourceId);
      if (error) throw error;
    },
    onSuccess: (_d, vars) => {
      qc.invalidateQueries({ queryKey: ["data_source_files", vars.dataSourceId] });
      invalidate();
    },
  });

  return { create, update, remove, insertFiles };
}
