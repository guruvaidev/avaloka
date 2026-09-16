import { Dialog, DialogContent } from "@/components/ui/dialog";
import { DataSetBrowserView } from "./DataSetBrowserView";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function DataSetBrowserModal({ open, onOpenChange }: Props) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="grid h-[min(720px,calc(100vh-64px))] w-[min(1080px,calc(100vw-48px))] max-w-none gap-0 overflow-hidden rounded-2xl border-secondary bg-primary p-0 shadow-2xl">
        <DataSetBrowserView
          onNavigateAway={() => onOpenChange(false)}
          className="h-full"
        />
      </DialogContent>
    </Dialog>
  );
}
