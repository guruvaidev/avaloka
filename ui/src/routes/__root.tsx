import { createServerFn } from "@tanstack/react-start";
import { getPublicRuntimeEnv, publicEnvScript } from "@/lib/runtime-env";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { useEffect, useMemo, type ReactNode } from "react";

import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";
import { Toaster } from "@/components/ui/sonner";
import { useThemeApplier } from "@/lib/theme";
import { TeamInvitationClaim } from "@/components/dashboard/TeamInvitationClaim";
import { TrainingJobsProvider } from "@/lib/training-jobs";

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

// TanStack Router 1.170.41 passes the error as `unknown` (anything can be thrown).
function ErrorComponent({ error: thrown, reset }: { error: unknown; reset: () => void }) {
  const error = useMemo(
    () => (thrown instanceof Error ? thrown : new Error(String(thrown ?? "Unknown error"))),
    [thrown],
  );
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

// Only browser-safe keys reach the page: API_BASE and SUPABASE_INCLUSTER_URL are
// in-cluster addresses the UI server proxies to, and must not be published.
const getPublicEnv = createServerFn({ method: "GET" }).handler(() => getPublicRuntimeEnv());

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
      { property: "og:image", content: "/og-image.png" },
      { name: "twitter:image", content: "/og-image.png" },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      { rel: "icon", type: "image/svg+xml", href: "/favicon.svg" },
      { rel: "icon", type: "image/png", href: "/favicon.png" },
      // Fonts are self-hosted (styles.css, public/fonts): no third-party request
      // leaves the viewer's browser for a page to render.
      {
        rel: "preload",
        href: "/fonts/figtree/figtree-latin-wght-normal.woff2",
        as: "font",
        type: "font/woff2",
        crossOrigin: "anonymous",
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
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: publicEnvScript(env) }} />
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
  const router = useRouter();

  useThemeApplier();

  useEffect(() => {
    const cookie = document.cookie
      .split("; ")
      .find((item) => item.startsWith("tab="));

    const cookietab = cookie
      ? decodeURIComponent(cookie.slice("tab=".length))
      : null;

    if (cookietab === "billing") {
      // Delete the cookie
      document.cookie =
        "tab=; path=/; max-age=0; SameSite=Lax";

      // Redirect to /settings?tab=billing
      void router.navigate({
        to: "/settings",
        search: { tab: "billing" },
        replace: true,
      });
    }
  }, [router]);

  return (
    <QueryClientProvider client={queryClient}>
      <TeamInvitationClaim />
      <TrainingJobsProvider>
        <Outlet />
      </TrainingJobsProvider>

      <Toaster position="top-center" richColors />
    </QueryClientProvider>
  );
}
