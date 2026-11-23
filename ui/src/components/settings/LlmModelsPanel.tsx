import { useEffect, useState } from "react";
import { backendApi, type LlmModelsResponse } from "@/lib/api/backendApi";

/**
 * The model screen: which language model each agent role runs, and the fallback
 * chain behind it (GET /api/llm-models). Each reply separately states which model
 * answered it when a fallback stood in.
 */
const loadFromApi = () => backendApi.getLlmModels();

export function LlmModelsPanel({
  load = loadFromApi,
}: {
  load?: () => Promise<LlmModelsResponse>;
}) {
  const [data, setData] = useState<LlmModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    load()
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : "Failed to load"));
    return () => {
      live = false;
    };
  }, [load]);

  return (
    <section
      data-testid="llm-models"
      aria-label="Language models"
      className="m-4 rounded-2xl border border-border bg-white p-5 text-sm shadow-sm dark:bg-[#111418] sm:mx-6"
    >
      <h2 className="text-base font-semibold text-foreground">Language models</h2>
      {error ? (
        <p className="mt-2 text-xs text-muted-foreground">Could not load the model setup: {error}</p>
      ) : !data ? (
        <p className="mt-2 text-xs text-muted-foreground">Loading…</p>
      ) : (
        <ul className="mt-3 flex flex-col gap-3">
          {data.roles.map((r) => (
            <li key={r.role} data-testid={`llm-role-${r.role}`}>
              <div className="font-medium text-foreground">
                {r.role}: {r.provider}/{r.model}
              </div>
              <ol className="mt-1 flex flex-col gap-0.5 text-xs text-muted-foreground">
                {r.chain.map((t, i) => (
                  <li key={`${t.provider}-${t.model}`} data-testid={`llm-tier-${r.role}-${i}`}>
                    {i === 0 ? "Configured" : `Fallback ${i}`}: {t.provider}/{t.model}
                    {t.providers_only?.length ? ` (providers: ${t.providers_only.join(", ")})` : ""}
                    {t.armed ? "" : " (not available: no key set)"}
                  </li>
                ))}
              </ol>
            </li>
          ))}
        </ul>
      )}
      {data?.note && <p className="mt-3 text-xs text-muted-foreground">{data.note}</p>}
    </section>
  );
}
