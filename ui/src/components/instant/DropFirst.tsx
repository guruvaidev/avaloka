// The Start state of /instant: a drop zone and the sample datasets. The accept
// list and the byte limit are the caller's (from /api/limits), never this file's.
import { useRef, useState, type DragEvent, type ChangeEvent } from "react";
import type { InstantLimits, SampleDataset } from "@/lib/api/instant";
import { cx } from "@/lib/utils/cx";

export function DropFirst({
  accept,
  limits,
  samples,
  busy,
  error,
  onFile,
  onSample,
}: {
  accept: string;
  limits: InstantLimits | null;
  samples: SampleDataset[];
  /** What the page is doing right now; disables the inputs. */
  busy: string | null;
  error: string | null;
  onFile: (file: File) => void;
  onSample: (sample: SampleDataset) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const disabled = busy != null;

  const pick = (list: FileList | null) => {
    const file = list?.[0];
    if (file) onFile(file);
  };
  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setOver(false);
    if (!disabled) pick(e.dataTransfer?.files ?? null);
  };
  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    pick(e.target.files);
    e.target.value = "";
  };

  const formats = accept
    .split(",")
    .map((s) => s.trim().replace(/^\./, "").toUpperCase())
    .join(", ");

  return (
    <section aria-label="Start" className="flex flex-col gap-4">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        data-testid="instant-dropzone"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={cx(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-10 text-center outline-none transition-colors focus-visible:ring-2 focus-visible:ring-brand",
          over
            ? "border-brand bg-brand-primary_alt"
            : "border-secondary bg-secondary hover:bg-primary_hover",
          disabled && "cursor-wait opacity-60",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          data-testid="instant-file-input"
          className="hidden"
          disabled={disabled}
          onChange={onChange}
        />
        <p className="text-md font-semibold text-primary">
          {busy ?? "Drop a file here, or tap to choose one"}
        </p>
        <p className="text-sm text-tertiary" data-testid="instant-limit-line">
          {formats}
          {limits ? ` up to ${limits.maxFileMb} MB` : ""}
        </p>
      </div>

      {error ? (
        <p role="alert" className="text-sm text-error-primary">
          {error}
        </p>
      ) : null}

      <div className="flex flex-col gap-2">
        <p className="text-sm text-secondary">Or try sample data</p>
        <div className="flex flex-wrap gap-2">
          {samples.map((s) => (
            <button
              key={s.id}
              type="button"
              disabled={disabled}
              data-testid={`instant-sample-${s.id}`}
              onClick={() => onSample(s)}
              className="rounded-full border border-secondary bg-primary px-4 py-2 text-sm font-medium text-secondary transition-colors hover:bg-primary_hover disabled:cursor-wait disabled:opacity-60"
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
