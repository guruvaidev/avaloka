import { useState, type ReactNode } from "react";
import { AlertTriangle, Check, ChevronDown, FileText, Pencil, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { CLEANING_DISCLOSURE, issueHasDetails, reviewIssues, type ReviewIssue } from "@/lib/upload-review";

export function ReviewTooltip({ text, children }: { text: string; children: ReactNode }) {
  return <TooltipProvider delayDuration={250}><Tooltip><TooltipTrigger asChild>{children}</TooltipTrigger><TooltipContent className="max-w-80 border border-ai-border bg-primary text-primary shadow-md">{text}</TooltipContent></Tooltip></TooltipProvider>;
}

export function FileStatusBar({ file, ready, onReplace }: { file: File; ready: boolean; onReplace: () => void }) {
  const size = file.size < 1024 * 1024 ? `${(file.size / 1024).toFixed(1)} KB` : `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  return <div className="flex h-12 min-w-0 items-center gap-2 rounded-lg border border-ai-border bg-ai-tile px-3">
    <FileText className="size-4 shrink-0 text-tertiary" />
    <ReviewTooltip text={file.name}><span tabIndex={0} className="min-w-0 flex-1 truncate text-xs font-medium text-primary">{file.name}</span></ReviewTooltip>
    <span className="shrink-0 text-[11px] text-tertiary">{size}</span>
    <span className="inline-flex shrink-0 items-center gap-1 text-xs text-success-primary"><Check className="size-3" />{ready ? "Ready" : "Pending"}</span>
    <Button variant="link" size="sm" className="shrink-0 px-0 text-xs text-brand-secondary" onClick={onReplace}>Replace file</Button>
  </div>;
}

export function EditableDatasetName({ name, onSave }: { name: string; onSave: (name: string) => void }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const save = () => { if (draft.trim()) { onSave(draft.trim().slice(0, 80)); setEditing(false); } };
  return editing ? <div className="flex items-center gap-2">
    <input autoFocus aria-label="Dataset name" maxLength={80} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => {
      if (event.key === "Enter") { event.preventDefault(); save(); }
      if (event.key === "Escape") { event.stopPropagation(); setEditing(false); setDraft(name); }
    }} className="min-w-0 flex-1 rounded-md border border-ai-border bg-primary px-2 py-1.5 text-sm text-primary outline-brand" />
    <Button variant="ghost" size="icon" aria-label="Save dataset name" onClick={save}><Check /></Button>
    <Button variant="ghost" size="icon" aria-label="Cancel name edit" onClick={() => setEditing(false)}><X /></Button>
  </div> : <div className="flex min-w-0 items-center gap-2"><h3 className="min-w-0 break-words text-base font-semibold text-primary">{name}</h3><ReviewTooltip text="Edit dataset name"><Button variant="ghost" size="icon" className="shrink-0" aria-label="Edit dataset name" onClick={() => { setDraft(name); setEditing(true); }}><Pencil /></Button></ReviewTooltip></div>;
}

export function ReviewIssueRow({ issue, index }: { issue: ReviewIssue; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const interactive = issueHasDetails(issue);
  const content = <><span className="shrink-0 text-xs text-ai-issue-text">{String(index + 1).padStart(2, "0")}</span><span className="min-w-0 flex-1 text-left"><strong className="block text-sm font-medium text-primary">{issue.title}</strong><small className="block whitespace-normal text-xs font-normal text-tertiary">{issue.summary}</small></span><AlertTriangle className="size-4 shrink-0 text-ai-warning" />{interactive && <ChevronDown className={`size-4 shrink-0 text-tertiary transition-transform motion-reduce:transition-none ${expanded ? "rotate-180" : ""}`} />}</>;
  return <div className="overflow-hidden rounded-lg border border-ai-issue-border bg-ai-issue-row">
    {interactive ? <Button variant="ghost" aria-expanded={expanded} aria-controls={`upload-issue-${issue.id}`} onClick={() => setExpanded(!expanded)} className="h-auto w-full justify-start gap-3 rounded-none px-3 py-3 whitespace-normal hover:bg-ai-issue-pill focus-visible:ring-inset">{content}</Button> : <div className="flex items-center gap-3 px-3 py-3">{content}</div>}
    {interactive && expanded && <div id={`upload-issue-${issue.id}`} className="space-y-2 border-t border-ai-issue-border px-3 py-3 text-xs text-secondary">
      {issue.details.map((detail, index) => <p key={index}>{detail}</p>)}
      {Boolean(issue.columns?.length) && <table className="w-full text-left"><thead><tr><th className="pb-2">Column</th><th className="pb-2 text-right">Empty cells</th><th className="pb-2 text-right">%</th></tr></thead><tbody>{issue.columns?.map((column, index) => <tr key={index}><td className="max-w-40 break-words py-1">{column.name}</td><td className="py-1 text-right tabular-nums">{column.count?.toLocaleString() ?? "Not supplied"}</td><td className="py-1 text-right tabular-nums">{column.percent != null ? `${column.percent}%` : "Not supplied"}</td></tr>)}</tbody></table>}
      <p className="border-t border-ai-issue-border pt-2">When you clean: no issue-specific cleaning options are sent; backend handling is not confirmed.</p>
    </div>}
  </div>;
}

export function DatasetReviewIssues({ dataset }: { dataset: unknown }) {
  const issues = reviewIssues(dataset);
  return <section className="mt-4"><h4 className="text-sm font-semibold text-primary">Dataset issues</h4><div className="mt-2 space-y-2">{issues.length ? issues.map((issue, index) => <ReviewIssueRow key={issue.id} issue={{ ...issue, id: `${(dataset as any)?.dataset_id}-${issue.id}` }} index={index} />) : <p className="text-xs text-tertiary">No issue details were supplied by the profile.</p>}</div></section>;
}

export function CleaningRules() {
  const [expanded, setExpanded] = useState(false);
  return <div className="mt-4"><Button variant="link" className="h-auto px-0 text-xs text-brand-secondary" aria-expanded={expanded} aria-controls="upload-cleaning-rules" onClick={() => setExpanded(!expanded)}>View cleaning rules<ChevronDown className={`transition-transform motion-reduce:transition-none ${expanded ? "rotate-180" : ""}`} /></Button>{expanded && <div id="upload-cleaning-rules" className="mt-2 rounded-lg border border-ai-border bg-ai-tile p-3 text-xs text-secondary"><p>{CLEANING_DISCLOSURE}</p><ul className="mt-2 list-disc space-y-1 pl-4"><li>Duplicate row handling: not confirmed.</li><li>Empty cell handling: not confirmed.</li><li>Original file retention: not confirmed.</li></ul></div>}</div>;
}
/** Multi-dataset summary: "{workbook}: {N} sheets, {rows} rows" or "{N} files", plus skipped-sheet notes once. */
export function UploadSummaryCard({ title, notes }: { title: string; notes: string[] }) {
  return <div className="mt-4 rounded-lg border border-ai-border bg-ai-tile px-3 py-2.5" data-testid="upload-summary-card">
    <p className="text-sm font-semibold text-primary">{title}</p>
    {notes.map((note, index) => <p key={index} className="mt-1 text-xs text-tertiary">{note}</p>)}
  </div>;
}

/** One 48px collapsed row per sheet / file; expanding shows that dataset's existing details. */
export function CompactDatasetRow({ name, meta, domain, issueCount, open, onToggle, checkbox, children }: {
  name: string; meta: string; domain?: string | null; issueCount: number; open: boolean; onToggle: () => void;
  checkbox?: { checked: boolean; onChange: (checked: boolean) => void; label: string }; children: ReactNode;
}) {
  return <div className="rounded-lg border border-ai-border bg-primary">
    <div className="flex h-12 min-w-0 items-center gap-2 px-3">
      {checkbox && <input type="checkbox" aria-label={checkbox.label} checked={checkbox.checked} onChange={(event) => checkbox.onChange(event.target.checked)} className="size-4 shrink-0 accent-brand-solid" />}
      <button type="button" aria-expanded={open} onClick={onToggle} className="flex min-w-0 flex-1 items-center gap-2 text-left">
        <span className="min-w-0 truncate text-sm font-medium text-primary" title={name}>{name}</span>
        <span className="shrink-0 text-[11px] text-tertiary">{meta}</span>
        {domain ? <span className="min-w-0 truncate rounded bg-secondary px-1.5 py-0.5 text-[10px] text-tertiary">{domain}</span> : null}
        <span className="ml-auto flex shrink-0 items-center gap-2">
          {issueCount > 0 ? <span className="rounded-full border border-ai-issue-border bg-ai-issue-pill px-2 py-0.5 text-[11px] font-medium text-ai-warning">{issueCount} issue{issueCount === 1 ? "" : "s"}</span> : null}
          <ChevronDown className={`size-4 text-tertiary transition-transform motion-reduce:transition-none ${open ? "rotate-180" : ""}`} />
        </span>
      </button>
    </div>
    {open ? <div className="border-t border-ai-border px-3 pb-3">{children}</div> : null}
  </div>;
}
