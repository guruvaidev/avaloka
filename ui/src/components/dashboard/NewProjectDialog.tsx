import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export const PROJECT_NAME_MAX = 80;

/** Small "New project" form; `onCreate` runs the existing create-project mutation and rejects on failure. */
export function NewProjectDialog({
  open,
  onOpenChange,
  onCreate,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreate: (name: string, description: string | null) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [pending, setPending] = useState(false);
  const canCreate = name.trim().length > 0 && !pending;

  const submit = async () => {
    if (!canCreate) return;
    setPending(true);
    try {
      await onCreate(name.trim(), description.trim() || null);
      setName("");
      setDescription("");
      onOpenChange(false);
    } catch {
      /* the caller shows the error; keep what the user typed */
    } finally {
      setPending(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => (!pending ? onOpenChange(o) : undefined)}>
      <DialogContent className="max-w-[440px] border-border bg-card text-foreground">
        <DialogHeader>
          <DialogTitle>New project</DialogTitle>
          <DialogDescription className="text-muted-foreground">Name your project to enable uploads and data sources.</DialogDescription>
        </DialogHeader>
        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <label className="block space-y-1.5">
            <span className="text-sm font-medium text-foreground">Project name</span>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={PROJECT_NAME_MAX}
              required
              readOnly={pending}
              autoFocus
              placeholder="Project name"
              className="border-input bg-background text-foreground"
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-sm font-medium text-foreground">
              Description <span className="font-normal text-muted-foreground">(Optional)</span>
            </span>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              readOnly={pending}
              placeholder="Description"
              className="min-h-[72px] resize-none border-input bg-background text-foreground"
            />
          </label>
          <DialogFooter className="gap-2">
            <Button type="button" variant="outline" disabled={pending} onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!canCreate} aria-busy={pending}>
              {pending ? <Loader2 className="size-4 animate-spin" aria-label="Creating project" /> : null}
              Create
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
