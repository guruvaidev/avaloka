import { createFileRoute, useRouter, type ErrorComponentProps } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { InferenceWorkspace } from "@/components/inference/InferenceWorkspace";

export const Route = createFileRoute("/_authenticated/inference")({
  validateSearch: (search: Record<string, unknown>): { run_id?: string } => ({ run_id: typeof search.run_id === "string" ? search.run_id : undefined }),
  head: () => ({
    meta: [
      { title: "Inference · Avaloka AI" },
      { name: "description", content: "Run real-time and batch predictions on your trained Avaloka models." },
      { property: "og:title", content: "Inference · Avaloka AI" },
      { property: "og:description", content: "Run real-time and batch predictions on your trained Avaloka models." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
    links: [{ rel: "stylesheet", href: "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" }],
  }),
  component: InferencePage,
  errorComponent: InferenceError,
});
function InferencePage() { const { run_id } = Route.useSearch(); return <InferenceWorkspace runId={run_id ?? ""} />; }
function InferenceError({ error }: ErrorComponentProps) {
  const router = useRouter();
  return (
    <main className="inference-page min-w-0 flex-1 overflow-y-auto bg-inf-canvas p-7">
      <h1 className="text-xl font-semibold">Inference</h1>
      <section className="inf-card mt-5" role="alert">
        <p className="text-sm font-medium">This section couldn't load</p>
        <Button variant="outline" className="mt-3 min-h-11" onClick={() => void router.invalidate()}>Retry</Button>
        <details className="mt-3 text-xs text-muted-foreground"><summary className="cursor-pointer">Details</summary><pre className="mt-2 whitespace-pre-wrap break-all">{error instanceof Error ? error.message : String(error)}</pre></details>
      </section>
    </main>
  );
}
