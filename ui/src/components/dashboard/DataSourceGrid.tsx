import { useState } from "react";
import { Button } from "@/components/base/buttons/button";
import { ArrowRight } from "@untitledui/icons";
const mysql = { url: "/assets/logos/Logo_Type_MySQL.png" };
const postgres = { url: "/assets/logos/Logo_Type_PostgreSQL.png" };
const sqlserver = { url: "/assets/logos/Logo_Type_Microsoft_SQL.png" };
const mariadb = { url: "/assets/logos/Logo_Type_MariaDB.png" };
const mongodb = { url: "/assets/logos/Logo_Type_Mongodb.png" };
const gcs = { url: "/assets/logos/Logo_Type_GCS.png" };
const s3 = { url: "/assets/logos/Logo_Type_S3.png" };
const azure = { url: "/assets/logos/Logo_Type_Azure.png" };
import type {
  StorageConnection,
  StorageProvider,
} from "@/components/database/CloudStorageConnectModal";
import { CloudStorageBrowserModal } from "@/components/database/CloudStorageBrowserModal";
import { CloudStorageConnectionsListModal } from "@/components/database/CloudStorageConnectionsListModal";
import { DatabaseConnectionsListModal } from "@/components/database/DatabaseConnectionsListModal";
import type { DbType } from "@/components/database/DatabaseConnectModal";
import { cn } from "@/lib/utils";

export type DataSource = {
  id: string;
  name: string;
  logoUrl: string;
  storageProvider?: StorageProvider;
  dbType?: DbType;
  comingSoon?: boolean;
  comingSoonHint?: string;
};

const initialSources: DataSource[] = [
  { id: "mysql", name: "MySQL", logoUrl: mysql.url, dbType: "mysql" },
  { id: "postgres", name: "PostgreSQL", logoUrl: postgres.url, dbType: "postgresql" },
  { id: "sqlserver", name: "SQL Server", logoUrl: sqlserver.url, dbType: "mssql" },
  { id: "mariadb", name: "MariaDB", logoUrl: mariadb.url, dbType: "mariadb" },
  { id: "mongodb", name: "MongoDB", logoUrl: mongodb.url, dbType: "mongodb" },
];

const moreSources: DataSource[] = [
  { id: "gcs", name: "Google Cloud Storage", logoUrl: gcs.url, storageProvider: "gcp" },
  { id: "s3", name: "Amazon S3", logoUrl: s3.url, storageProvider: "aws" },
  { id: "azure", name: "Azure Blob Storage", logoUrl: azure.url, storageProvider: "azure" },
];

export function DataSourceGrid({ onSelect, disabled = false }: { onSelect?: (id: string) => void; disabled?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const [connectProvider, setConnectProvider] = useState<StorageProvider | null>(null);
  const [browseConnection, setBrowseConnection] = useState<StorageConnection | null>(null);
  const [dbType, setDbType] = useState<DbType | null>(null);
  const sources = expanded ? [...initialSources, ...moreSources] : initialSources;

  const handleTileClick = (s: DataSource) => {
    if (disabled) return;
    if (s.comingSoon) return;
    if (s.storageProvider) {
      setConnectProvider(s.storageProvider);
      return;
    }
    if (s.dbType) {
      setDbType(s.dbType);
      return;
    }
    onSelect?.(s.id);
  };

  return (
    <div className={cn("grid grid-cols-2 gap-3 sm:grid-cols-4", disabled && "opacity-60")}>
      {sources.map((s) => (
        <div
          key={s.id}
          className="relative"
          title={
            disabled
              ? "Create a project first"
              : s.comingSoon
                ? s.comingSoonHint || "Coming soon"
                : undefined
          }
        >
          <Button
            color="secondary"
            size="lg"
            onPress={() => handleTileClick(s)}
            isDisabled={s.comingSoon || disabled}
            className={cn(
              "h-[72px] w-full justify-center rounded-xl",
              (s.comingSoon || disabled) && "cursor-not-allowed opacity-60",
            )}
          >
            <img src={s.logoUrl} alt={s.name} className="h-10 w-full max-w-[112px] shrink-0 object-contain object-center" />
          </Button>
          {s.comingSoon && (
            <span className="pointer-events-none absolute -right-1 -top-1 rounded-full bg-amber-100 px-1.5 py-0.5 text-[9px] font-medium text-amber-700 shadow">
              Coming soon
            </span>
          )}
        </div>
      ))}
      {!expanded && (
        <Button
          color="link-gray"
          size="sm"
          iconTrailing={ArrowRight}
          isDisabled={disabled}
          onPress={() => setExpanded(true)}
          className="col-span-full mt-1 justify-self-end"
        >
          View More
        </Button>
      )}


      {connectProvider && (
        <CloudStorageConnectionsListModal
          open={!!connectProvider}
          provider={connectProvider}
          onOpenChange={(o: boolean) => {
            if (!o) setConnectProvider(null);
          }}
          onBrowse={(conn: StorageConnection) => {
            setConnectProvider(null);
            setBrowseConnection(conn);
          }}
        />
      )}
      <CloudStorageBrowserModal
        open={!!browseConnection}
        connection={browseConnection}
        onOpenChange={(o) => {
          if (!o) setBrowseConnection(null);
        }}
      />
      {dbType && (
        <DatabaseConnectionsListModal
          open={!!dbType}
          dbType={dbType}
          onOpenChange={(o: boolean) => {
            if (!o) setDbType(null);
          }}
        />
      )}
    </div>
  );
}
