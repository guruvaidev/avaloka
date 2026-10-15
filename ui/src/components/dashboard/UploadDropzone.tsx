import { useRef, useState, type DragEvent } from "react";
import { cx } from "@/lib/utils/cx";
import { UploadModal } from "./UploadModal";
import { Button } from "@/components/base/buttons/button";
import { Upload01 } from "@untitledui/icons";
const bgPattern = { url: "/assets/dashboard/Background_pattern_decorative.png" };
const illustration = { url: "/assets/dashboard/Upload_Illustration.png" };

interface UploadDropzoneProps {
  projectId?: string | null;
  projectName?: string | null;
  redirectTo?: "analysis" | "preview";
  disabled?: boolean;
  onStandaloneAnalysisSaved?: (analysisId: string) => void;
}

export function UploadDropzone({ projectId, projectName, redirectTo, disabled = false, onStandaloneAnalysisSaved }: UploadDropzoneProps = {}) {
  const [open, setOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [droppedFiles, setDroppedFiles] = useState<File[] | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const openPicker = () => {
    if (disabled) return;
    if (inputRef.current) inputRef.current.value = "";
    inputRef.current?.click();
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (disabled) return;
    const files = Array.from(e.dataTransfer.files ?? []);
    setDroppedFiles(files.length ? files : null);
    setOpen(true);
  };

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    if (!next) setDroppedFiles(null);
  };


  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".csv,.xls,.xlsx,.json"
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (!files.length) return;
          setDroppedFiles(files);
          setOpen(true);
        }}
      />
      <div
        aria-disabled={disabled}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          openPicker();
        }}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragActive(true); }}
        onDragEnter={(e) => { e.preventDefault(); if (!disabled) setDragActive(true); }}
        onDragLeave={(e) => {
          if (e.relatedTarget instanceof Node && e.currentTarget.contains(e.relatedTarget)) return;
          setDragActive(false);
        }}
        onDrop={onDrop}
        className={cx(
           "group relative block w-full overflow-hidden rounded-2xl border-2 border-dashed px-6 py-8 text-center outline-focus-ring transition-colors",
          disabled
            ? "cursor-not-allowed border-secondary opacity-60 select-none"
             : "cursor-pointer hover:border-brand hover:bg-brand-primary_alt focus-within:border-brand",
           dragActive && !disabled ? "border-brand bg-brand-primary_alt ring-2 ring-brand" : "border-secondary bg-primary/60",
        )}
      >
        <img
          src={bgPattern.url}
          alt=""
          aria-hidden
          className="pointer-events-none absolute left-1/2 top-1/2 w-[480px] -translate-x-1/2 -translate-y-1/2 select-none opacity-30"
        />
        <div className="relative flex flex-col items-center">
          <img
            src={illustration.url}
            alt=""
            aria-hidden
            className="h-auto w-auto max-w-[120px] select-none object-contain"
          />
          <h3 className="mt-6 text-md font-semibold text-primary">
            Select or drag &amp; drop file to analyze data
          </h3>
          <Button color="primary" size="md" iconLeading={Upload01} isDisabled={disabled} onPress={openPicker} className="mt-4">
            Browse files
          </Button>
          <p className="mt-3 text-sm text-tertiary">CSV, Excel (.xls, .xlsx) or JSON · up to 100MB</p>
        </div>
      </div>

      <UploadModal
        open={open}
        onOpenChange={handleOpenChange}
        initialFiles={droppedFiles}
        projectId={projectId}
        projectName={projectName}
        redirectTo={redirectTo}
        onStandaloneAnalysisSaved={onStandaloneAnalysisSaved}
      />
    </div>
  );
}
