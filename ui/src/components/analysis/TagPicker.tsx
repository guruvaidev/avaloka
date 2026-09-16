import { useState, type FormEvent } from "react";
import { Tag01, X, HelpCircle, Plus } from "@untitledui/icons";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export type TagItem = { id: string; name: string; color: string };

const PRESET_COLORS = [
  "#1F2937",
  "#10B981",
  "#3B82F6",
  "#6366F1",
  "#8B5CF6",
  "#D946EF",
  "#EC4899",
  "#F97316",
];

type Size = "sm" | "md";

interface TagPickerProps {
  size?: Size;
}

export function TagPicker({ size = "md" }: TagPickerProps) {
  const [tags, setTags] = useState<TagItem[]>([]);
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [color, setColor] = useState<string>(PRESET_COLORS[2]);
  const [customColor, setCustomColor] = useState<string>("#7F56D9");

  const isSm = size === "sm";

  const reset = () => {
    setName("");
    setColor(PRESET_COLORS[2]);
    setCustomColor("#7F56D9");
  };

  const handleAdd = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setTags((prev) => [
      ...prev,
      { id: `${Date.now()}-${Math.random()}`, name: trimmed, color },
    ]);
    reset();
    setOpen(false);
  };

  const removeTag = (id: string) => {
    setTags((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      {tags.map((t) => (
        <span
          key={t.id}
          className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium"
          style={{
            color: t.color,
            borderColor: `${t.color}40`,
            backgroundColor: `${t.color}14`,
          }}
        >
          {t.name}
          <button
            type="button"
            onClick={() => removeTag(t.id)}
            className="grid size-4 place-items-center rounded-full hover:bg-black/10"
            aria-label={`Remove ${t.name}`}
          >
            <X className="size-3" />
          </button>
        </span>
      ))}

      <Popover
        open={open}
        onOpenChange={(v) => {
          setOpen(v);
          if (!v) reset();
        }}
      >
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  className={
                    isSm
                      ? "grid size-6 place-items-center rounded-full text-tertiary hover:bg-secondary hover:text-primary transition"
                      : "grid size-7 place-items-center rounded-full text-tertiary hover:bg-secondary hover:text-primary transition"
                  }
                  aria-label="Add tag"
                >
                  <Tag01 className={isSm ? "size-3.5" : "size-4"} />
                </button>
              </PopoverTrigger>
            </TooltipTrigger>
            <TooltipContent
              side="bottom"
              sideOffset={6}
              className="bg-black text-white px-3 py-1.5 text-sm rounded-lg border-0 shadow-lg"
            >
              Add Tag
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <PopoverContent
          align="start"
          sideOffset={8}
          className="w-[360px] rounded-2xl border border-secondary bg-primary p-0 shadow-xl"
        >
          <form onSubmit={handleAdd}>
            <div className="flex items-center justify-between px-5 pt-4 pb-2">
              <h4 className="text-base font-semibold text-primary">Tag</h4>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="grid size-7 place-items-center rounded-md text-tertiary hover:bg-secondary hover:text-primary"
                aria-label="Close"
              >
                <X className="size-4" />
              </button>
            </div>

            <div className="space-y-4 px-5 pb-5">
              <div>
                <label className="flex items-center gap-1 text-sm font-medium text-primary">
                  Tag Name <span className="text-red-500">*</span>
                  <HelpCircle className="size-3.5 text-tertiary" />
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Enter a Tag name"
                  className="mt-1.5 w-full rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary placeholder:text-tertiary focus:border-brand-secondary focus:outline-none"
                  autoFocus
                />
              </div>

              <div>
                <p className="text-sm font-semibold text-primary">Brand color</p>
                <p className="text-xs text-tertiary">
                  Update your dashboard to your brand color.
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  {PRESET_COLORS.map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={() => setColor(c)}
                      className={
                        "size-7 rounded-full transition " +
                        (color === c
                          ? "ring-2 ring-offset-2 ring-offset-primary"
                          : "")
                      }
                      style={{
                        backgroundColor: c,
                        boxShadow:
                          color === c ? `0 0 0 2px ${c}` : undefined,
                      }}
                      aria-label={`Color ${c}`}
                    />
                  ))}
                </div>

                <div className="mt-3 flex items-center gap-3">
                  <span className="text-sm font-semibold text-primary">
                    Custom
                  </span>
                  <label className="relative inline-block size-7">
                    <span
                      className="block size-7 rounded-full ring-2 ring-brand-secondary"
                      style={{ backgroundColor: customColor }}
                    />
                    <input
                      type="color"
                      value={customColor}
                      onChange={(e) => {
                        setCustomColor(e.target.value);
                        setColor(e.target.value);
                      }}
                      className="absolute inset-0 size-full cursor-pointer opacity-0"
                    />
                  </label>
                  <input
                    type="text"
                    value={customColor}
                    onChange={(e) => {
                      setCustomColor(e.target.value);
                      setColor(e.target.value);
                    }}
                    className="flex-1 rounded-lg border border-secondary bg-primary px-3 py-1.5 text-sm uppercase text-primary focus:border-brand-secondary focus:outline-none"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 border-t border-secondary px-5 py-3">
              <button
                type="button"
                onClick={() => {
                  reset();
                  setOpen(false);
                }}
                className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-secondary bg-primary px-4 py-2 text-sm font-semibold text-primary hover:bg-secondary"
              >
                <X className="size-4" /> Cancel
              </button>
              <button
                type="submit"
                disabled={!name.trim()}
                className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-brand-solid px-4 py-2 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-50"
              >
                <Plus className="size-4" /> Add
              </button>
            </div>
          </form>
        </PopoverContent>
      </Popover>
    </div>
  );
}
