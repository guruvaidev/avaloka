import { useEffect } from "react";
import { Trash01, X } from "@untitledui/icons";
import {
  Dialog,
  DialogContent,
  DialogOverlay,
  DialogPortal,
} from "@/components/ui/dialog";
import { Button } from "@/components/base/buttons/button";

interface DeleteAnalysisModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}

export function DeleteAnalysisModal({ open, onOpenChange, onConfirm }: DeleteAnalysisModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogPortal>
        <DialogOverlay className="bg-black/40" />
        <DialogContent className="max-w-[480px] gap-0 rounded-2xl border-0 bg-primary p-6 shadow-xl [&>button]:hidden">
          <div className="flex items-start gap-4">
            <div className="flex size-11 shrink-0 items-center justify-center rounded-full bg-error-secondary">
              <Trash01 className="size-5 text-fg-error-primary" />
            </div>
            <div className="flex-1 pt-1">
              <h3 className="text-base font-semibold text-primary">Delete Analysis</h3>
              <p className="mt-1 text-sm text-tertiary">
                Are you sure you want to delete this analysis ? This action cannot be undone.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              aria-label="Close"
              className="cursor-pointer rounded-md p-1 text-fg-quaternary hover:text-fg-quaternary_hover"
            >
              <X className="size-5" />
            </button>
          </div>
          <div className="mt-6 flex justify-end gap-3">
            <Button color="secondary" size="md" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button color="primary-destructive" size="md" onClick={onConfirm}>
              Delete
            </Button>
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}

interface DeletedToastModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  label: string;
}

export function DeletedToastModal({ open, onOpenChange, label }: DeletedToastModalProps) {
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => onOpenChange(false), 1600);
    return () => clearTimeout(t);
  }, [open, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogPortal>
        <DialogOverlay className="bg-black/40" />
        <DialogContent className="max-w-[360px] gap-0 rounded-2xl border-0 bg-primary p-8 shadow-xl [&>button]:hidden">
          <div className="flex flex-col items-center text-center">
            <div className="flex size-12 items-center justify-center rounded-full bg-error-secondary">
              <Trash01 className="size-6 text-fg-error-primary" />
            </div>
            <h3 className="mt-5 text-base font-semibold text-primary">Deleted</h3>
            <p className="mt-1 text-sm text-tertiary">{label}</p>
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}
