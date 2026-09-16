import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { bootstrapPayPalSandbox } from "@/lib/paypal-bootstrap.functions";
import { diagnosePayPalConfig } from "@/lib/paypal-diagnostics.functions";

export const Route = createFileRoute("/_authenticated/paypal-bootstrap")({
  component: PayPalBootstrapPage,
});

type Result = {
  productId: string;
  planId: string;
  webhookId: string | null;
  webhookUrl: string;
  apiBase: string;
  staleDeleted?: string[];
  staleFailed?: Array<{ id: string; url: string; error: string }>;
  currentEnvPlanId?: string | null;
  currentEnvWebhookId?: string | null;
};

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        <button
          type="button"
          className="text-xs font-semibold text-[#1565EF] hover:underline"
          onClick={() => {
            navigator.clipboard.writeText(value);
            toast.success(`${label} copied`);
          }}
        >
          Copy
        </button>
      </div>
      <p className="break-all font-mono text-sm">{value}</p>
    </div>
  );
}

function PayPalBootstrapPage() {
  const bootstrap = useServerFn(bootstrapPayPalSandbox);
  const diagnose = useServerFn(diagnosePayPalConfig);
  const [loading, setLoading] = useState(false);
  const [diagLoading, setDiagLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [diag, setDiag] = useState<any>(null);

  const run = async () => {
    try {
      setLoading(true);
      const res = (await bootstrap()) as Result;
      setResult(res);
      toast.success("PayPal provisioning complete");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Bootstrap failed";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const runDiag = async () => {
    try {
      setDiagLoading(true);
      const res = await diagnose();
      setDiag(res);
      toast.success("Diagnostics complete");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Diagnostics failed");
    } finally {
      setDiagLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl px-6 py-12">
      <h1 className="text-2xl font-semibold">PayPal Sandbox Bootstrap</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Creates the PayPal Product, Billing Plan (15-day trial + monthly), and
        Webhook using your saved <code>PAYPAL_CLIENT_ID</code> /
        <code> PAYPAL_CLIENT_SECRET</code>. Admin only.
      </p>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button
          onClick={run}
          disabled={loading}
          className="h-11 bg-[#1565EF] font-semibold text-white hover:bg-[#1257d4]"
        >
          {loading ? "Provisioning…" : "Provision PayPal (sandbox)"}
        </Button>
        <Button
          onClick={runDiag}
          disabled={diagLoading}
          variant="outline"
          className="h-11 font-semibold"
        >
          {diagLoading ? "Checking…" : "Run diagnostics"}
        </Button>
      </div>

      {diag && (
        <pre className="mt-6 max-h-[500px] overflow-auto rounded-lg border border-border bg-muted p-3 text-xs">
          {JSON.stringify(diag, null, 2)}
        </pre>
      )}

      {result && (
        <div className="mt-8 space-y-3">
          <Row label="API Base" value={result.apiBase} />
          <Row label="Product ID" value={result.productId} />
          <Row label="PAYPAL_PLAN_ID (save as secret)" value={result.planId} />
          <Row label="Webhook URL" value={result.webhookUrl} />
          {result.webhookId && (
            <Row
              label="PAYPAL_WEBHOOK_ID (save as secret)"
              value={result.webhookId}
            />
          )}
          {(result.currentEnvPlanId !== result.planId ||
            result.currentEnvWebhookId !== result.webhookId) && (
            <div className="rounded-md border border-amber-400 bg-amber-50 p-3 text-sm">
              <div className="font-semibold text-amber-900">
                Secrets are out of sync
              </div>
              <div className="mt-1 text-amber-900">
                Current <code>PAYPAL_PLAN_ID</code> in env:{" "}
                <code>{result.currentEnvPlanId || "(unset)"}</code>
                <br />
                Current <code>PAYPAL_WEBHOOK_ID</code> in env:{" "}
                <code>{result.currentEnvWebhookId || "(unset)"}</code>
                <br />
                Update both secrets to the values above, then re-run diagnostics.
                Runtime secrets in Lovable Cloud take effect on the next
                server-function call — no redeploy needed.
              </div>
            </div>
          )}
          {result.staleDeleted && result.staleDeleted.length > 0 && (
            <div className="rounded-md border border-border bg-muted p-3 text-xs">
              <div className="font-semibold">Deleted stale webhooks:</div>
              <ul className="mt-1 list-disc pl-5">
                {result.staleDeleted.map((s) => (
                  <li key={s} className="break-all">{s}</li>
                ))}
              </ul>
            </div>
          )}
          {result.staleFailed && result.staleFailed.length > 0 && (
            <div className="rounded-md border border-red-300 bg-red-50 p-3 text-xs">
              <div className="font-semibold text-red-900">
                Could not delete some webhooks:
              </div>
              <ul className="mt-1 list-disc pl-5 text-red-900">
                {result.staleFailed.map((s) => (
                  <li key={s.id} className="break-all">
                    {s.id} ({s.url}) — {s.error}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="rounded-md border border-[#1565EF]/30 bg-[#eaf1ff] p-3 text-sm">
            Next: save <code>PAYPAL_PLAN_ID</code> and{" "}
            <code>PAYPAL_WEBHOOK_ID</code> as secrets. Once saved, PayPal
            checkout is live from the Billing tab.
          </div>
        </div>
      )}
    </div>
  );
}
