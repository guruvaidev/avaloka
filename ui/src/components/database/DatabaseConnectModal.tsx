import { useMemo, useRef, useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AlertTriangle, Database, Loader2, X } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { MCP_API_BASE } from "@/lib/api/mcp-config";
import { BACKEND_API_BASE } from "@/lib/api/backend-config";
import { mcpProxyFetch, proxyFetch } from "@/lib/api/backendApi";

type Stage = "idle" | "registering" | "saving" | "encrypting";
const STAGE_LABEL: Record<Exclude<Stage, "idle">, string> = {
  registering: "Registering database…",
  saving: "Saving connection…",
  encrypting: "Encrypting credentials…",
};

export type DbType = "postgresql" | "mysql" | "mariadb" | "mssql" | "mongodb" | "oracle" | "sqlite";

export const DB_LABEL: Record<DbType, string> = {
  postgresql: "PostgreSQL",
  mysql: "MySQL",
  mariadb: "MariaDB",
  mssql: "Microsoft SQL Server",
  mongodb: "MongoDB",
  oracle: "Oracle",
  sqlite: "SQLite",
};

const DEFAULT_PORT: Record<DbType, number> = {
  postgresql: 5432,
  mysql: 3306,
  mariadb: 3306,
  mssql: 1433,
  mongodb: 27017,
  oracle: 1521,
  sqlite: 0,
};

const CUSTOMER_ID_RE = /^[a-z0-9_]+$/;

export type DatabaseRegisterResponse = {
  customer_id: string;
  api_key: string;
  mcp_endpoint: string;
  status: string;
  created_at: string;
  database_type: string;
};

export function DatabaseConnectModal({
  open,
  dbType,
  onOpenChange,
  onConnected,
}: {
  open: boolean;
  dbType: DbType;
  onOpenChange: (o: boolean) => void;
  onConnected?: (connectionId: string) => void;
}) {
  const [form, setForm] = useState(() => ({
    connection_name: "",
    customer_id: "",
    customer_name: "",
    contact_email: "",
    environment: "production",
    host: "",
    port: String(DEFAULT_PORT[dbType] || ""),
    database_name: "",
    username: "",
    password: "",
    query_timeout: "30",
    max_rows: "1000",
    connection_string: "",
  }));
  const isMongo = dbType === "mongodb";
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState("");
  const customerIdRef = useRef<HTMLInputElement>(null);

  const setField = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const cidNormalized = useMemo(() => form.customer_id.trim().toLowerCase(), [form.customer_id]);

  const validate = (): string | null => {
    if (!form.connection_name.trim()) return "Connection Name is required";
    if (cidNormalized.length < 3 || !CUSTOMER_ID_RE.test(cidNormalized))
      return "Customer ID must be 3+ chars, lowercase letters/digits/underscore only";
    if (!form.customer_name.trim()) return "Customer Name is required";
    if (!form.contact_email.trim() || !form.contact_email.includes("@"))
      return "Valid Contact Email is required";
    const hasUri = isMongo && !!form.connection_string.trim();
    if (!hasUri) {
      if (!form.host.trim()) return "Host is required";
      if (!form.database_name.trim()) return "Database Name is required";
      if (!form.username.trim()) return "Username is required";
      if (!form.password) return "Password is required";
    }
    return null;
  };

  const handleSubmit = async () => {
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setError("");
    setBusy(true);
    setStage("registering");

    let insertedId: string | null = null;
    let reg: DatabaseRegisterResponse;

    // --- Stage 1: Register with MCP ---
    try {
      const port = parseInt(form.port, 10) || DEFAULT_PORT[dbType];
      const query_timeout = parseInt(form.query_timeout, 10) || 30;
      const max_rows = parseInt(form.max_rows, 10) || 1000;

      let res: Response;
      try {
        res = await mcpProxyFetch(MCP_API_BASE, "/customers/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            customer_id: cidNormalized,
            customer_name: form.customer_name.trim(),
            contact_email: form.contact_email.trim(),
            environment: form.environment,
            database_type: dbType,
            host: form.host.trim(),
            port,
            database_name: form.database_name.trim(),
            username: form.username,
            password: form.password,
            query_timeout,
            max_rows,
            ...(isMongo && form.connection_string.trim()
              ? { connection_string: form.connection_string.trim() }
              : {}),
          }),
        });
      } catch (netErr: any) {
        setError(
          `Registering database failed: could not reach the registration service (${netErr?.message || "network error"}).`,
        );
        setBusy(false);
        setStage("idle");
        return;
      }

      const body = await res.json().catch(() => ({} as any));
      if (!res.ok) {
        if (res.status === 409) {
          setError(
            `Customer ID "${cidNormalized}" is already registered. Choose a different Customer ID.`,
          );
          setTimeout(() => customerIdRef.current?.focus(), 0);
        } else if (res.status === 422) {
          const detail =
            typeof body?.detail === "string"
              ? body.detail
              : Array.isArray(body?.detail)
                ? body.detail.map((d: any) => d?.msg || JSON.stringify(d)).join("; ")
                : JSON.stringify(body?.detail || {});
          setError(`Registering database failed (validation): ${detail}`);
        } else {
          setError(
            `Registration failed: ${body?.detail || res.statusText || `HTTP ${res.status}`}`,
          );
        }
        setBusy(false);
        setStage("idle");
        return;
      }
      reg = body as DatabaseRegisterResponse;

      // --- Stage 2: Save connection row ---
      setStage("saving");
      const { data: userData } = await supabase.auth.getUser();
      const uid = userData.user?.id;
      if (!uid) {
        setError("Saving connection failed: not signed in.");
        setBusy(false);
        setStage("idle");
        return;
      }

      const { data: inserted, error: insertErr } = await (supabase.from as any)("mcp_connections")
        .insert({
          user_id: uid,
          connection_name: form.connection_name.trim(),
          customer_id: reg.customer_id,
          customer_name: form.customer_name.trim(),
          contact_email: form.contact_email.trim(),
          database_type: dbType,
          host: form.host.trim(),
          port,
          database_name: form.database_name.trim(),
          username: "",
          api_key: "",
          mcp_endpoint: reg.mcp_endpoint,
          status: reg.status,
          query_timeout,
          max_rows,
        })
        .select("id")
        .single();

      if (insertErr || !inserted) {
        setError(`Saving connection failed: ${insertErr?.message || "unknown error"}`);
        setBusy(false);
        setStage("idle");
        return;
      }
      insertedId = (inserted as any).id as string;

      // --- Stage 3: Encrypt credentials (via backend, no CORS preflight) ---
      setStage("encrypting");
      try {
        const { data: sess } = await supabase.auth.getSession();
        const token = sess.session?.access_token;
        if (!token) throw new Error("Your session expired. Sign in again.");
        const encRes = await proxyFetch(
          `/api/mcp-connections/${insertedId}/encrypt`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ apiKey: reg.api_key, username: form.username }),
          },
        );
        if (!encRes.ok) {
          let msg = `Encrypting credentials failed (${encRes.status})`;
          if (encRes.status === 401) msg = "Your session expired. Sign in again.";
          else if (encRes.status === 403) msg = "That connection belongs to another account.";
          else if (encRes.status === 500)
            msg = "Server could not encrypt credentials (encryption key not configured).";
          throw new Error(msg);
        }
      } catch (encErr: any) {
        // rollback saved row
        try {
          await (supabase.from as any)("mcp_connections").delete().eq("id", insertedId);
        } catch {}
        const msg = encErr?.message || "";
        if (/failed to fetch|network|cors/i.test(msg)) {
          setError(
            "Could not reach the credential encryption service. The connection was not saved.",
          );
        } else {
          setError(msg || "Encrypting credentials failed");
        }
        setBusy(false);
        setStage("idle");
        return;
      }

      toast.success("Database connected");
      onConnected?.(insertedId);
      onOpenChange(false);
    } finally {
      setBusy(false);
      setStage((s) => (s === "idle" ? s : "idle"));
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <div className="flex items-center justify-between border-b px-1 pb-3">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-[#1565EF]" />
            <div>
              <h2 className="text-base font-semibold">Connect {DB_LABEL[dbType]}</h2>
              <p className="text-xs text-muted-foreground">
                Opens a live connection to your database. Credentials are encrypted server-side.
              </p>
            </div>
          </div>
          <button
            onClick={() => onOpenChange(false)}
            className="rounded p-1 text-muted-foreground hover:bg-muted"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="grid max-h-[65vh] grid-cols-2 gap-3 overflow-y-auto p-1 pt-3">
          <Field label="Connection Name *">
            <Input value={form.connection_name} onChange={setField("connection_name")} placeholder="Production analytics" />
          </Field>
          <Field label="Database Type">
            <Input value={DB_LABEL[dbType]} readOnly className="bg-muted" />
          </Field>
          {isMongo && (
            <div className="col-span-2">
              <Field
                label="Connection String (URI) — optional"
                hint="Alternative to the individual fields below, e.g. mongodb+srv://user:pass@cluster0.mongodb.net/db"
              >
                <Input
                  value={form.connection_string}
                  onChange={setField("connection_string")}
                  placeholder="mongodb+srv://user:pass@cluster0.mongodb.net/mydb"
                  autoComplete="off"
                />
              </Field>
            </div>
          )}
          <Field label="Customer ID *" hint="Lowercase letters, digits, underscore. 3+ chars.">
            <Input
              ref={customerIdRef}
              value={form.customer_id}
              onChange={(e) =>
                setForm((f) => ({ ...f, customer_id: e.target.value.replace(/\s+/g, "") }))
              }
              onBlur={() =>
                setForm((f) => ({ ...f, customer_id: f.customer_id.trim().toLowerCase() }))
              }
              placeholder="acme_prod"
            />
            {form.customer_id && cidNormalized !== form.customer_id && (
              <p className="mt-1 text-[11px] text-muted-foreground">
                Will be sent as: <code>{cidNormalized}</code>
              </p>
            )}
          </Field>
          <Field label="Customer Name *">
            <Input value={form.customer_name} onChange={setField("customer_name")} placeholder="Acme Inc" />
          </Field>
          <Field label="Contact Email *">
            <Input type="email" value={form.contact_email} onChange={setField("contact_email")} placeholder="ops@acme.com" />
          </Field>
          <Field label="Connection Type">
            <select
              value={form.environment}
              onChange={(e) => setForm((f) => ({ ...f, environment: e.target.value }))}
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            >
              <option value="production">production</option>
              <option value="staging">staging</option>
              <option value="development">development</option>
            </select>
          </Field>
          <Field label="Host *">
            <Input value={form.host} onChange={setField("host")} placeholder="db.acme.internal" />
          </Field>
          <Field label="Port">
            <Input type="number" value={form.port} onChange={setField("port")} />
          </Field>
          <Field label="Database Name *">
            <Input value={form.database_name} onChange={setField("database_name")} />
          </Field>
          <Field label="Username *">
            <Input value={form.username} onChange={setField("username")} autoComplete="off" />
          </Field>
          <Field label="Password *">
            <Input type="password" value={form.password} onChange={setField("password")} autoComplete="new-password" />
          </Field>
          <div />
          <Field label="Query Timeout (s)">
            <Input type="number" value={form.query_timeout} onChange={setField("query_timeout")} />
          </Field>
          <Field label="Max Rows">
            <Input type="number" value={form.max_rows} onChange={setField("max_rows")} />
          </Field>
        </div>

        {error && (
          <div className="mt-2 flex items-start gap-2 rounded border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="mt-3 flex justify-end gap-2 border-t pt-3">
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={busy}
            className="gap-2 bg-[#1565EF] hover:bg-[#1257cf]"
          >
            {busy && <Loader2 className="h-4 w-4 animate-spin" />}
            {busy ? (stage !== "idle" ? STAGE_LABEL[stage] : "Connecting…") : "Test & Register"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <Label className="mb-1 block text-xs font-medium">{label}</Label>
      {children}
      {hint && <p className="mt-1 text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}
