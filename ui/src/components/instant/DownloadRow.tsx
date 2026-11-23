// Downloads: the interactive export, the notebook that reproduces it, and the
// Markdown when the run has one. Share is disabled until saved reports exist
// (PR-13: report-level links, private by default, 1.7-INSTANT-ANALYSIS.md §4.5).
export const SHARE_PENDING_TITLE = "Sharing arrives with saved reports";

const LINK_CLASS =
  "rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary_hover";

export function DownloadRow({
  exportUrl,
  notebookUrl,
  markdownUrl,
}: {
  exportUrl?: string;
  notebookUrl?: string;
  markdownUrl?: string;
}) {
  return (
    <nav aria-label="Downloads" className="flex flex-wrap gap-2" data-testid="instant-downloads">
      {exportUrl ? (
        <a href={exportUrl} download className={LINK_CLASS} data-testid="instant-download-html">
          Interactive report (.html)
        </a>
      ) : null}
      {notebookUrl ? (
        <a href={notebookUrl} download className={LINK_CLASS} data-testid="instant-download-ipynb">
          Notebook (.ipynb)
        </a>
      ) : null}
      {markdownUrl ? (
        <a href={markdownUrl} download className={LINK_CLASS} data-testid="instant-download-md">
          Markdown (.md)
        </a>
      ) : null}
      <button
        type="button"
        disabled
        title={SHARE_PENDING_TITLE}
        aria-label="Share"
        className="rounded-lg border border-secondary px-3 py-2 text-sm font-medium text-tertiary disabled:cursor-not-allowed disabled:opacity-60"
        data-testid="instant-share"
      >
        Share
      </button>
    </nav>
  );
}
