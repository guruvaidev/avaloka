import { Folder } from "lucide-react";
import { Button } from "@/components/ui/button";
import { folderHint, type BucketFolder } from "@/lib/bucket-picker";

export function BucketFolderName({ folder, onOpen, disabled }: { folder: BucketFolder; onOpen: (prefix: string) => void; disabled?: boolean }) {
  const hint = folderHint(folder.name);
  return (
    <Button variant="link" disabled={disabled} title={folder.name} onDoubleClick={(event) => event.stopPropagation()} onClick={() => onOpen(folder.prefix)} className="h-auto w-full min-w-0 justify-start gap-2 p-0 text-left font-medium text-foreground hover:no-underline">
      <Folder className="shrink-0 text-primary" />
      <span className="min-w-0 flex-1">
        <span className="block truncate">{folder.name}</span>
        {hint && <span className="block truncate text-[10px] font-normal leading-3 text-muted-foreground">{hint}</span>}
      </span>
    </Button>
  );
}