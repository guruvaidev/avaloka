import type { Provider } from "@/lib/data-sources";

export type ProviderFieldOption = { value: string; label: string };

export type ProviderField = {
  key: string;
  label: string;
  type?: "text" | "password" | "number" | "file" | "select";
  placeholder?: string;
  required?: boolean;
  tooltip?: string;
  options?: ProviderFieldOption[];
};

/** Region options for GCP cloud connect (matches legacy frontend). */
export const GCP_REGION_OPTIONS: ProviderFieldOption[] = [
  { value: "us-east-1", label: "us-east-1 (N. Virginia)" },
  { value: "us-east-2", label: "us-east-2 (Ohio)" },
  { value: "us-west-1", label: "us-west-1 (N. California)" },
  { value: "us-west-2", label: "us-west-2 (Oregon)" },
  { value: "eu-west-1", label: "eu-west-1 (Ireland)" },
  { value: "eu-central-1", label: "eu-central-1 (Frankfurt)" },
  { value: "ap-south-1", label: "ap-south-1 (Mumbai)" },
  { value: "ap-southeast-1", label: "ap-southeast-1 (Singapore)" },
  { value: "ap-northeast-1", label: "ap-northeast-1 (Tokyo)" },
];

export type ProviderMeta = {
  id: Provider;
  name: string;
  description: string;
  fields: ProviderField[];
  /** Show the Live / Synced connection-type radio above the fields */
  hasConnectionType?: boolean;
};

const hostPort = (port: string): ProviderField[] => [
  { key: "host", label: "Host", placeholder: "Add IP Address or Domain Name", required: true },
  { key: "port", label: "Port", type: "number", placeholder: port, required: true, tooltip: `Default port is ${port}` },
];

const userPass: ProviderField[] = [
  { key: "username", label: "Username", placeholder: "Enter Username", required: true },
  { key: "password", label: "Password", type: "password", placeholder: "Enter Password", required: true },
];

const dbBasic = (port: string): ProviderField[] => [...hostPort(port), ...userPass];

export const PROVIDERS: ProviderMeta[] = [
  {
    id: "mysql",
    name: "MySQL",
    description: "Connect to a MySQL database",
    hasConnectionType: true,
    fields: dbBasic("3306"),
  },
  {
    id: "snowflake",
    name: "Snowflake",
    description: "Connect to a Snowflake account",
    fields: [
      { key: "account", label: "Account Identifier", placeholder: "xy12345.ap-south-1", required: true },
      ...userPass,
    ],
  },
  {
    id: "postgres",
    name: "PostgreSQL",
    description: "Connect to a Postgres database",
    fields: [
      { key: "host", label: "Host", placeholder: "Enter Domain", required: true },
      { key: "port", label: "Port", type: "number", placeholder: "5432", required: true, tooltip: "Default port is 5432" },
      ...userPass,
    ],
  },
  {
    id: "mssql",
    name: "Microsoft SQL",
    description: "Connect to a Microsoft SQL Server",
    fields: [
      { key: "host", label: "Host", placeholder: "Enter Domain", required: true },
      { key: "port", label: "Port", type: "number", placeholder: "1433", required: true, tooltip: "Default port is 1433" },
      ...userPass,
    ],
  },
  { id: "mariadb", name: "MariaDB", description: "Connect to a MariaDB database", fields: dbBasic("3306") },
  {
    id: "clickhouse",
    name: "ClickHouse",
    description: "Connect to a ClickHouse cluster",
    fields: [
      { key: "host", label: "Host", placeholder: "Enter Domain Name", required: true },
      {
        key: "port",
        label: "Port",
        type: "number",
        placeholder: "8123",
        required: true,
        tooltip: "8123 (HTTP interface – most common), 9000 (Native TCP – advanced)",
      },
      ...userPass,
    ],
  },
  {
    id: "bigquery",
    name: "Google BigQuery",
    description: "Connect to a GCS bucket folder in your GCP project",
    fields: [
      {
        key: "project_id",
        label: "Project ID",
        placeholder: "GCP Console → Project ID",
        required: false,
        tooltip: "GCP Console → project selector → Project ID",
      },
      {
        key: "region",
        label: "Region",
        type: "select",
        required: true,
        options: GCP_REGION_OPTIONS,
        tooltip: "Bucket location / region from Google Cloud Console",
      },
      {
        key: "bucket_name",
        label: "Bucket name",
        placeholder: "Cloud Storage → Buckets → Name",
        required: true,
        tooltip: "Cloud Storage → Buckets → Name column (not Project ID)",
      },
      {
        key: "folder_path",
        label: "Folder path",
        placeholder: "Path inside bucket, e.g. data/exports",
        required: true,
        tooltip: "Folder prefix inside the bucket (no bucket name here)",
      },
      {
        key: "service_account_json",
        label: "Service Account Key (JSON)",
        type: "file",
        required: true,
        tooltip: "GCP Console → IAM & Admin → Service Accounts → Keys → Add key → JSON",
      },
    ],
  },
  { id: "trino", name: "Trino", description: "Connect to a Trino cluster", fields: dbBasic("8080") },
  { id: "presto", name: "Presto", description: "Connect to a Presto cluster", fields: dbBasic("8080") },
  {
    id: "mongodb",
    name: "MongoDB",
    description: "Connect to a MongoDB cluster",
    fields: [
      { key: "host", label: "Host", placeholder: "cluster0.mongodb.net", required: true },
      ...userPass,
    ],
  },
  {
    id: "databricks",
    name: "Databricks",
    description: "Connect to a Databricks workspace",
    fields: [
      { key: "host", label: "Workspace URL", placeholder: "adb-xxx.azuredatabricks.net", required: true },
      { key: "http_path", label: "HTTP Path", placeholder: "/sql/1.0/warehouses/abc", required: true },
      { key: "password", label: "Access Token", type: "password", placeholder: "Enter Access Token", required: true },
    ],
  },
  { id: "cloudsql", name: "Cloud SQL", description: "Connect to a Google Cloud SQL instance", fields: dbBasic("5432") },
];

export function getProvider(id: Provider): ProviderMeta {
  return PROVIDERS.find((p) => p.id === id)!;
}
