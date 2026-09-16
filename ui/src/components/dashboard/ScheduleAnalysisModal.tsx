import { useEffect, useRef, useState } from "react";
import { Calendar, X, HelpCircle, UploadCloud01, ChevronDown, CheckCircle } from "@untitledui/icons";
import {
  Dialog,
  DialogContent,
  DialogOverlay,
  DialogPortal,
} from "@/components/ui/dialog";
import { Button } from "@/components/base/buttons/button";
import { cx } from "@/lib/utils/cx";

interface ScheduleAnalysisModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSchedule?: (data: ScheduleFormData) => void;
}

export interface ScheduleFormData {
  name: string;
  state: string;
  date: string;
  time: string;
  file: File | null;
}

const STATES = ["Engineering", "Marketing", "Sales", "Finance", "Operations"];
const TIMES = [
  "12:00 AM", "1:00 AM", "2:00 AM", "9:00 AM", "10:00 AM",
  "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "5:00 PM",
];

function Field({
  label,
  required,
  tooltip,
  children,
}: {
  label: string;
  required?: boolean;
  tooltip?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-1 text-sm font-semibold text-primary">
        {label}
        {required && <span className="text-brand-secondary">*</span>}
        {tooltip && <HelpCircle className="size-4 text-fg-quaternary" />}
      </label>
      {children}
    </div>
  );
}

const fieldCx =
  "h-11 w-full rounded-lg border border-secondary bg-primary px-3.5 text-md text-primary placeholder:text-placeholder shadow-xs outline-none focus:border-brand focus:ring-2 focus:ring-brand/20 transition";

export function ScheduleAnalysisModal({ open, onOpenChange, onSchedule }: ScheduleAnalysisModalProps) {
  const [name, setName] = useState("");
  const [state, setState] = useState("");
  const [date, setDate] = useState("");
  const [time, setTime] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) {
      // reset after close animation
      const t = setTimeout(() => setSuccess(false), 200);
      return () => clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    if (!success) return;
    const t = setTimeout(() => onOpenChange(false), 1800);
    return () => clearTimeout(t);
  }, [success, onOpenChange]);

  const handleSubmit = () => {
    onSchedule?.({ name, state, date, time, file });
    setSuccess(true);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent className={cx("gap-0 overflow-hidden rounded-2xl border-secondary bg-primary p-0 shadow-xl", success ? "max-w-[440px]" : "max-w-[640px]")}> 
          {success ? (
            <div className="flex flex-col items-center justify-center px-8 py-12 text-center">
              <div className="flex size-14 items-center justify-center rounded-full bg-success-secondary">
                <CheckCircle className="size-7 text-success-primary" />
              </div>
              <h2 className="mt-6 text-lg font-semibold text-primary">Successfully Scheduled</h2>
              <p className="mt-1 text-md text-tertiary">{name || "Analysis I"}</p>
            </div>
          ) : (
          <>
          {/* Header */}
          <div className="flex items-start justify-between gap-4 border-b border-secondary px-6 py-5">
            <div className="flex items-start gap-4">
              <div className="flex size-12 shrink-0 items-center justify-center rounded-xl border border-secondary bg-primary shadow-xs">
                <Calendar className="size-6 text-fg-secondary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-primary">Schedule Analysis</h2>
                <p className="mt-0.5 text-sm text-tertiary">
                  Your new project has been created. Invite colleagues to collaborate on this project.
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              className="rounded-md p-1 text-fg-quaternary transition hover:bg-primary_hover hover:text-fg-quaternary_hover"
              aria-label="Close"
            >
              <X className="size-5" />
            </button>
          </div>

          {/* Body */}
          <div className="flex flex-col gap-5 px-6 py-5">
            <Field label="Analysis Name">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter Analysis Name"
                className={fieldCx}
              />
            </Field>

            <Field label="State">
              <div className="relative">
                <select
                  value={state}
                  onChange={(e) => setState(e.target.value)}
                  className={cx(fieldCx, "appearance-none pr-10", !state && "text-placeholder")}
                >
                  <option value="" disabled>Select Department</option>
                  {STATES.map((s) => (
                    <option key={s} value={s} className="text-primary">{s}</option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-3.5 top-1/2 size-5 -translate-y-1/2 text-fg-quaternary" />
              </div>
            </Field>

            <div className="grid grid-cols-2 gap-4">
              <Field label="Date">
                <div className="relative">
                  <input
                    type="date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    className={cx(fieldCx, "pl-10")}
                  />
                  <Calendar className="pointer-events-none absolute left-3.5 top-1/2 size-5 -translate-y-1/2 text-fg-quaternary" />
                </div>
              </Field>
              <Field label="Time">
                <div className="relative">
                  <select
                    value={time}
                    onChange={(e) => setTime(e.target.value)}
                    className={cx(fieldCx, "appearance-none pr-10", !time && "text-placeholder")}
                  >
                    <option value="" disabled>12:00 AM</option>
                    {TIMES.map((t) => (
                      <option key={t} value={t} className="text-primary">{t}</option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-3.5 top-1/2 size-5 -translate-y-1/2 text-fg-quaternary" />
                </div>
              </Field>
            </div>

            <Field label="Files" required tooltip>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className={cx(fieldCx, "flex items-center justify-between text-left", !file && "text-placeholder")}
                >
                  <span className="truncate">{file?.name ?? "Choose File"}</span>
                  <ChevronDown className="size-5 text-fg-quaternary" />
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </div>
            </Field>

            {/* OR separator */}
            <div className="flex items-center gap-3">
              <div className="h-px flex-1 bg-border-secondary" />
              <span className="text-sm font-medium text-tertiary">OR</span>
              <div className="h-px flex-1 bg-border-secondary" />
            </div>

            {/* Dropzone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={cx(
                "flex flex-col items-center justify-center gap-3 rounded-xl border border-secondary bg-primary px-6 py-5 transition",
                dragging && "border-brand bg-brand-primary/5",
              )}
            >
              <div className="flex size-10 items-center justify-center rounded-lg border border-secondary bg-primary shadow-xs">
                <UploadCloud01 className="size-5 text-fg-quaternary" />
              </div>
              <p className="text-sm text-tertiary">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="font-semibold text-brand-secondary hover:underline"
                >
                  Select
                </button>{" "}
                or drag and drop file
              </p>
              <p className="text-xs text-tertiary">SVG, PNG, JPG or GIF (max. 800x400px)</p>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 border-t border-secondary px-6 py-4">
            <Button color="secondary" size="md" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button color="primary" size="md" onClick={handleSubmit}>
              Schedule
            </Button>
          </div>
          </>
          )}
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}
