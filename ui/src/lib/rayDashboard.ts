export function rayDashboardBase(): string {
  // Same origin as the app: the UI's nginx serves /ray/ (see webui-configmap.yaml).
  // Local K8s: the UI is on localhost:3000 and the dashboard proxy on localhost:28265
  // (same site for cookies). Elsewhere (test/app): this site's own /ray/.
  const local = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  const url = import.meta.env.VITE_RAY_DASHBOARD_URL
    || (local ? "http://localhost:28265/ray/" : new URL("/ray/", window.location.origin).toString());
  return `${url.replace(/\/+$/, "")}/`;
}

export function rayJobUrl(jobId: string): string {
  return `${rayDashboardBase()}#/jobs/${encodeURIComponent(jobId)}`;
}

/** Preserve existing markdown and code; annotate ordinary job IDs for display-only links. */
export function rayJobMarkdown(content: string): string {
  return content.split(/(```[\s\S]*?```|`[^`]*`|!?\[[^\]]*\]\([^)]*\))/g)
    .map((part, i) => i % 2 ? part : part.replace(/\bdta-[0-9a-f]{16}\b/g, (id) => `[${id}](${rayJobUrl(id)})`)).join("");
}

export function rayJobIdFromHref(href: string | undefined): string | undefined {
  if (!href) return undefined;
  if (href.startsWith("/ray-jobs?")) return new URLSearchParams(href.split("?")[1]).get("job") || undefined;
  if (href.startsWith(`${rayDashboardBase()}#/jobs/`)) {
    try { return decodeURIComponent(href.slice(`${rayDashboardBase()}#/jobs/`.length)) || undefined; }
    catch { return undefined; }
  }
  return undefined;
}