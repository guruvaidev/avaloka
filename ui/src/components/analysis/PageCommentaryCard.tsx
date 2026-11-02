import { PAGE_COMMENTARY_LABEL, type CommentaryLine } from "@/lib/page-commentary";

/**
 * "On this page": code-written commentary about the Auto Insights the page is
 * showing. Labelled with its source so nobody mistakes it for a model reply;
 * renders nothing when the profile has no computed figures.
 */
export function PageCommentaryCard({
  lines,
  datasetName,
}: {
  lines: CommentaryLine[];
  datasetName?: string;
}) {
  if (!lines.length) return null;
  return (
    <section
      aria-label="On this page"
      data-testid="page-commentary"
      data-source="profile"
      className="mb-5 rounded-lg border border-secondary bg-secondary/40 px-3 py-2.5"
    >
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-xs font-semibold text-primary">
          On this page{datasetName ? `: ${datasetName}` : ""}
        </h3>
        <span className="text-[11px] text-tertiary" data-testid="page-commentary-label">
          {PAGE_COMMENTARY_LABEL}
        </span>
      </div>
      <ul className="mt-1.5 flex flex-col gap-1 text-xs text-secondary">
        {lines.map((line) => (
          <li key={line.key} data-key={line.key}>
            {line.text}
          </li>
        ))}
      </ul>
    </section>
  );
}
