import { useRef, useState } from "react";
import { ChevronDown, X, HelpCircle, CheckCircle2, Check, UploadCloud, FileText, Trash2 } from "lucide-react";

import { Dialog, DialogPortal, DialogOverlay } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ModelsIcon } from "@/components/dashboard/icons/ConfigSectionIcons";
import { cn } from "@/lib/utils";

type Step = "form" | "confirm" | "success";

const TASK_TYPES = ["Forecasting", "Prediction", "Comparison", "Classification", "Regression"];
const TAG_OPTIONS = ["Sales", "Marketing", "Revenue", "Forecasting", "Champion", "Analysis"];
const MODEL_FILES = ["Revenue Model.csv", "Sales Model.csv", "Forecast.csv", "Pipeline.csv"];

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function RegisterModelModal({ open, onOpenChange }: Props) {
  const [step, setStep] = useState<Step>("form");

  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [modelFile, setModelFile] = useState("");
  const [upload, setUpload] = useState<{ name: string; size: string; progress: number } | null>(null);

  const [taskOpen, setTaskOpen] = useState(false);
  const [tagOpen, setTagOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const canSubmit = !!(name && taskType && tags.length > 0 && modelFile && upload?.progress === 100);

  const reset = () => {
    setStep("form");
    setName("");
    setTaskType("");
    setDescription("");
    setTags([]);
    setModelFile("");
    setUpload(null);
  };

  const handleOpenChange = (o: boolean) => {
    if (!o) reset();
    onOpenChange(o);
  };

  const toggleTag = (t: string) =>
    setTags((p) => (p.includes(t) ? p.filter((x) => x !== t) : [...p, t]));

  const handleFile = (file: File) => {
    setUpload({ name: file.name, size: `${Math.round(file.size / 1024)} KB`, progress: 0 });
    let p = 0;
    const id = setInterval(() => {
      p += 20;
      setUpload((u) => (u ? { ...u, progress: Math.min(100, p) } : u));
      if (p >= 100) clearInterval(id);
    }, 150);
  };

  const handleConfirm = () => {
    setStep("success");
    setTimeout(() => handleOpenChange(false), 1800);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      {step === "form" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 max-h-[92vh] w-[760px] max-w-[94vw] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-2xl bg-white shadow-xl duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
            <div className="flex items-center justify-between border-b border-border px-5 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-white">
                  <ModelsIcon className="h-4 w-4 text-foreground" />
                </div>
                <h2 className="text-base font-semibold text-foreground">Register Model</h2>
              </div>
              <DialogPrimitive.Close className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </DialogPrimitive.Close>
            </div>

            <div className="max-h-[70vh] space-y-4 overflow-y-auto px-5 py-5">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Model Name</label>
                <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Enter Model Name" className="h-10" />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Task Type</label>
                <Popover open={taskOpen} onOpenChange={setTaskOpen}>
                  <PopoverTrigger asChild>
                    <button type="button" className={cn("flex h-10 w-full items-center justify-between rounded-md border border-border bg-white px-3 text-sm", taskType ? "text-foreground" : "text-muted-foreground")}>
                      {taskType || "Select Task Type"}
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                    {TASK_TYPES.map((t) => (
                      <button key={t} onClick={() => { setTaskType(t); setTaskOpen(false); }} className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted">
                        {t}
                        {taskType === t && <Check className="h-4 w-4 text-primary" />}
                      </button>
                    ))}
                  </PopoverContent>
                </Popover>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  Description <span className="font-normal text-muted-foreground">(Optional)</span>
                </label>
                <Textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Enter Description" className="min-h-[96px] resize-none" />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Tags</label>
                <Popover open={tagOpen} onOpenChange={setTagOpen}>
                  <PopoverTrigger asChild>
                    <button type="button" className="flex h-10 w-full items-center justify-between rounded-md border border-border bg-white px-3 text-sm text-muted-foreground">
                      Select Team Members
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                    {TAG_OPTIONS.map((t) => {
                      const sel = tags.includes(t);
                      return (
                        <button key={t} onClick={() => toggleTag(t)} className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted">
                          {t}
                          {sel && <Check className="h-4 w-4 text-primary" />}
                        </button>
                      );
                    })}
                  </PopoverContent>
                </Popover>
                {tags.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5 pt-2">
                    {tags.map((t) => (
                      <span key={t} className="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-200">
                        {t}
                        <button onClick={() => toggleTag(t)} className="text-blue-700/70 hover:text-blue-700" aria-label={`Remove ${t}`}>
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">Select / Upload Model</label>
                <Popover open={modelOpen} onOpenChange={setModelOpen}>
                  <PopoverTrigger asChild>
                    <button type="button" className={cn("flex h-10 w-full items-center justify-between rounded-md border border-border bg-white px-3 text-sm", modelFile ? "text-foreground" : "text-muted-foreground")}>
                      {modelFile || "Select Department"}
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-1" align="start">
                    {MODEL_FILES.map((f) => (
                      <button key={f} onClick={() => { setModelFile(f); setModelOpen(false); }} className="flex w-full items-center justify-between rounded-md px-2.5 py-2 text-sm hover:bg-muted">
                        {f}
                        {modelFile === f && <Check className="h-4 w-4 text-primary" />}
                      </button>
                    ))}
                  </PopoverContent>
                </Popover>
              </div>

              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const f = e.dataTransfer.files?.[0];
                  if (f) handleFile(f);
                }}
                className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-[#1565EF] bg-white px-6 py-8 text-center"
              >
                <button onClick={() => fileRef.current?.click()} className="flex h-10 w-10 items-center justify-center rounded-lg border border-border">
                  <UploadCloud className="h-5 w-5 text-foreground" />
                </button>
                <p className="mt-3 text-sm text-foreground">
                  <button onClick={() => fileRef.current?.click()} className="font-semibold text-[#1565EF]">Select</button>{" "}
                  or drag and drop file
                </p>
                <p className="mt-1 text-xs text-muted-foreground">SVG, PNG, JPG or GIF (max. 800x400px)</p>
                <input ref={fileRef} type="file" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
              </div>

              {upload && (
                <div className="flex items-start gap-3 rounded-lg border border-border bg-white p-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-md border border-border">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium text-foreground">{upload.name}</div>
                      <button onClick={() => setUpload(null)} className="text-muted-foreground hover:text-foreground">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                      <span>{upload.size} of {upload.size}</span>
                      {upload.progress === 100 && (
                        <span className="inline-flex items-center gap-1 text-emerald-600">
                          <CheckCircle2 className="h-3 w-3" /> Complete
                        </span>
                      )}
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                        <div className="h-full bg-[#1565EF] transition-all" style={{ width: `${upload.progress}%` }} />
                      </div>
                      <span className="text-xs text-muted-foreground">{upload.progress}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-end gap-3 border-t border-border px-5 py-4">
              <Button
                onClick={() => setStep("confirm")}
                disabled={!canSubmit}
                className="h-11 gap-2 bg-[#1565EF] px-5 font-semibold text-white hover:bg-[#1257cf] disabled:bg-[#f4f6fb] disabled:text-muted-foreground"
              >
                <CheckCircle2 className="h-4 w-4" />
                Register Model
              </Button>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}

      {step === "confirm" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[420px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-xl">
            <DialogPrimitive.Close className="absolute right-4 top-4 text-muted-foreground hover:text-foreground">
              <X className="h-5 w-5" />
            </DialogPrimitive.Close>
            <div className="flex flex-col items-center text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50">
                <HelpCircle className="h-6 w-6 text-[#1565EF]" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-foreground">Are you sure you want to Register Model</h3>
              <p className="mt-1 text-sm text-muted-foreground">{name}?</p>
              <div className="mt-6 flex w-full gap-3">
                <Button variant="outline" onClick={() => setStep("form")} className="h-11 flex-1 gap-2 font-semibold">
                  <X className="h-4 w-4" />
                  Cancel
                </Button>
                <Button onClick={handleConfirm} className="h-11 flex-1 gap-2 bg-[#1565EF] font-semibold text-white hover:bg-[#1257cf]">
                  <CheckCircle2 className="h-4 w-4" />
                  Confirm
                </Button>
              </div>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}

      {step === "success" && (
        <DialogPortal>
          <DialogOverlay />
          <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-[420px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-8 shadow-xl">
            <div className="flex flex-col items-center text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100">
                <CheckCircle2 className="h-6 w-6 text-emerald-600" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-foreground">Successfully Registered</h3>
              <p className="mt-1 text-sm text-muted-foreground">{name}</p>
            </div>
          </DialogPrimitive.Content>
        </DialogPortal>
      )}
    </Dialog>
  );
}
