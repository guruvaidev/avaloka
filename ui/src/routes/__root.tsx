import { createServerFn } from "@tanstack/react-start";
import { getRuntimeEnv } from "@/lib/runtime-env";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";

import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";
import { Toaster } from "@/components/ui/sonner";
import { useThemeApplier } from "@/lib/theme";
import { TrainingJobsProvider } from "@/lib/training-jobs";
import { AutoInsightsBotOverlay } from "@/components/analysis/AutoInsightsBotOverlay";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Go home
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();
  useEffect(() => {
    reportLovableError(error, { boundary: "tanstack_root_error_component" });
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          This page didn't load
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Something went wrong on our end. You can try refreshing or head back home.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Try again
          </button>
          <a
            href="/"
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

const getPublicEnv = createServerFn({ method: "GET" }).handler(() => getRuntimeEnv());

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  loader: () => getPublicEnv(),
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "Log in · Avaloka AI" },
      { name: "description", content: "Sign in to Avaloka AI — the unified data workspace for SQL warehouses, dashboards, and AI-driven analytics." },
      { name: "author", content: "Lovable" },
      { property: "og:title", content: "Log in · Avaloka AI" },
      { property: "og:description", content: "Sign in to Avaloka AI — the unified data workspace for SQL warehouses, dashboards, and AI-driven analytics." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
      { name: "twitter:site", content: "@Lovable" },
      { name: "twitter:title", content: "Log in · Avaloka AI" },
      { name: "twitter:description", content: "Sign in to Avaloka AI — the unified data workspace for SQL warehouses, dashboards, and AI-driven analytics." },
      { property: "og:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/90f70da9-867a-464c-845d-9435165b579a/id-preview-9f21cf52--d096b358-5cb6-462a-800f-115f9aa8054c.lovable.app-1783161881109.png" },
      { name: "twitter:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/90f70da9-867a-464c-845d-9435165b579a/id-preview-9f21cf52--d096b358-5cb6-462a-800f-115f9aa8054c.lovable.app-1783161881109.png" },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      { rel: "icon", type: "image/svg+xml", href: "/favicon.svg" },
      { rel: "icon", type: "image/png", href: "/favicon.png" },
      { rel: "preconnect", href: "https://api.fontshare.com" },
      {
        rel: "stylesheet",
        href: "https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&display=swap",
      },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: ReactNode }) {
  const env = Route.useLoaderData();
  const serialized = JSON.stringify(env).replace(/</g, "\\u003c");
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: `window.__ENV__=${serialized}` }} />
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();
  useThemeApplier();

  return (
    <QueryClientProvider client={queryClient}>
      <TrainingJobsProvider>
        {/* Required: nested routes render here. Removing <Outlet /> breaks all child routes. */}
        <Outlet />
        <AutoInsightsBotOverlay />
      </TrainingJobsProvider>
      <Toaster position="top-center" richColors />
    </QueryClientProvider>
  );
}
