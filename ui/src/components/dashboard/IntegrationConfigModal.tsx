import { useEffect, useState } from "react";
import { Eye, EyeOff, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { integrationsApi, REPO_PATTERN, type IntegrationState } from "@/lib/api/integrations";

interface Props {
  /** Provider key; only "github" is wired up. */
  provider: string;
  name: string;
  state?: IntegrationState;
  onClose: () => void;
  /** Called after a successful save so the caller can refetch card state. */
  onSaved: () => void;
}

export function IntegrationConfigModal({ provider, name, state, onClose, onSaved }: Props) {
  const isGithub = provider === "github";
  const connected = !!state?.has_token;

  const [token, setToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [repo, setRepo] = useState(state?.config?.repo ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Never let the token outlive the modal.
  useEffect(() => () => setToken(""), []);

  const repoValid = REPO_PATTERN.test(repo.trim());
  const repoTouchedInvalid = repo.length > 0 && !repoValid;

  const close = () => {
    setToken("");
    onClose();
  };

  const submit = async () => {
    setError(null);
    setSuccess(null);
    const trimmedRepo = repo.trim();
    if (!repoValid) {
      setError("Repository must look like owner/repo.");
      return;
    }
    if (!connected && !token) {
      setError("A personal access token is required to connect.");
      return;
    }
    setBusy(true);
    try {
      if (token) {
        const res = await integrationsApi.connectGithub(token, trimmedRepo);
        setToken("");
        setSuccess(`Connected to ${res.repo} — push access confirmed`);
      } else {
        await integrationsApi.patchGithub({ repo: trimmedRepo });
        setSuccess(`Repository updated to ${trimmedRepo}`);
      }
      onSaved();
      setTimeout(close, 900);
    } catch (e: any) {
      setError(e?.message || "Something went wrong.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-[480px] rounded-2xl border border-border bg-background p-6 shadow-xl">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-base font-semibold">{name} integration</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              {isGithub
                ? "Connect your account with a personal access token."
                : "This integration isn't available yet."}
            </p>
          </div>
          <button onClick={close} aria-label="Close" className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        {!isGithub ? (
          <div className="mt-5 rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
            Coming soon
          </div>
        ) : (
          <>
            <label className="mt-5 block text-sm font-medium">Personal access token</label>
            <div className="relative mt-1.5">
              <input
                type={showToken ? "text" : "password"}
                value={token}
                autoComplete="off"
                onChange={(e) => setToken(e.target.value)}
                placeholder={connected ? "Token saved — leave blank to keep the current token" : "github_pat_…"}
                className="h-10 w-full rounded-lg border border-border bg-background px-3 pr-10 text-sm outline-none focus:border-[#1565EF]"
              />
              <button
                type="button"
                onClick={() => setShowToken((v) => !v)}
                aria-label={showToken ? "Hide token" : "Show token"}
                className="absolute inset-y-0 right-2 flex items-center text-muted-foreground hover:text-foreground"
              >
                {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>

            <label className="mt-4 block text-sm font-medium">Repository</label>
            <input
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              placeholder="owner/repo"
              className={cn(
                "mt-1.5 h-10 w-full rounded-lg border bg-background px-3 font-mono text-sm outline-none focus:border-[#1565EF]",
                repoTouchedInvalid ? "border-red-500" : "border-border",
              )}
            />
            {repoTouchedInvalid && (
              <p className="mt-1.5 text-xs text-red-600">Use the format owner/repo.</p>
            )}

            <p className="mt-3 text-xs text-muted-foreground">
              The token needs push (write) access to this repository.
            </p>

            {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
            {success && <p className="mt-4 text-sm text-green-600">{success}</p>}
          </>
        )}

        <div className="mt-6 flex justify-end gap-2">
          <Button variant="outline" className="h-9" onClick={close} disabled={busy}>
            Cancel
          </Button>
          {isGithub && (
            <Button
              className="h-9 gap-2 bg-[#1565EF] text-white hover:bg-[#1257cc] disabled:opacity-60"
              onClick={submit}
              disabled={busy || !repoValid || (!connected && !token)}
            >
              {busy && <Loader2 className="h-4 w-4 animate-spin" />}
              {connected ? "Save & test" : "Connect"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
