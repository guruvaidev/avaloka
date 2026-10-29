// Empty state for the Data View "Chart" tab. Charts are created from the chat;
// there is no in-page chart builder.

import { BarChart3, MessageSquare } from "lucide-react";

export function ChartEmptyState({ onAsk }: { onAsk: () => void }) {
  return (
    <div className="flex flex-col items-center gap-3 px-4 py-14 text-center">
      <div className="grid size-12 place-items-center rounded-full bg-[#eaf1fd] text-[#1a5fe0]">
        <BarChart3 className="size-6" />
      </div>
      <p className="max-w-sm text-sm text-secondary">No chart yet. Ask in the chat to create one.</p>
      <button
        type="button"
        onClick={onAsk}
        className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
      >
        <MessageSquare className="size-4" /> Ask in chat
      </button>
    </div>
  );
}
