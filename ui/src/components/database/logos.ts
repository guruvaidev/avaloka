const mysql = { url: "/assets/logos/Logo_Type_MySQL.png" };
const postgres = { url: "/assets/logos/Logo_Type_PostgreSQL.png" };
const snowflake = { url: "/assets/logos/Logo_Type_Snowflake.png" };
const mssql = { url: "/assets/logos/Logo_Type_Microsoft_SQL.png" };
const clickhouse = { url: "/assets/logos/Logo_Type_ClickHouse.png" };
const mariadb = { url: "/assets/logos/Logo_Type_MariaDB.png" };
const bigquery = { url: "/assets/logos/Logo_Type_Google_Big_Query.png" };
const trino = { url: "/assets/logos/Logo_Type_Trino.png" };
const presto = { url: "/assets/logos/Logo_Type_Presto.png" };
const mongodb = { url: "/assets/logos/Logo_Type_Mongodb.png" };
const databricks = { url: "/assets/logos/Logo_Type_Databricks.png" };
const cloudsql = { url: "/assets/logos/Logo_Type_Cloud_SQL.png" };
const gcs = { url: "/assets/logos/Logo_Type_GCS.png" };
const s3 = { url: "/assets/logos/Logo_Type_S3.png" };
const azure = { url: "/assets/logos/Logo_Type_Azure.png" };
import type { Provider, StorageProvider } from "@/lib/data-sources";

export const PROVIDER_LOGOS: Record<Provider, string> = {
  mysql: mysql.url,
  postgres: postgres.url,
  snowflake: snowflake.url,
  mssql: mssql.url,
  clickhouse: clickhouse.url,
  mariadb: mariadb.url,
  bigquery: bigquery.url,
  trino: trino.url,
  presto: presto.url,
  mongodb: mongodb.url,
  databricks: databricks.url,
  cloudsql: cloudsql.url,
};

export const STORAGE_PROVIDER_LOGOS: Record<StorageProvider, string> = {
  aws: s3.url,
  azure: azure.url,
  gcp: gcs.url,
};

// Keep string-keyed lookups (e.g. "gcs"/"s3") working for legacy modal code.
(PROVIDER_LOGOS as Record<string, string>).gcs = gcs.url;
(PROVIDER_LOGOS as Record<string, string>).s3 = s3.url;
(PROVIDER_LOGOS as Record<string, string>).azure = azure.url;
