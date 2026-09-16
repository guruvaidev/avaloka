import { Copy01, Link01, UserPlus01, X } from "@untitledui/icons";
import { Button } from "@/components/base/buttons/button";
import type { HistoryViewPayload } from "@/lib/analysis-history";
import { copyTextToClipboard, shareLinkFullUrl } from "@/lib/resource-collaborators";
import { toast } from "sonner";

type CollaborationPayload = Extract<HistoryViewPayload, { kind: "share" } | { kind: "collaborator" }>;

type Props = {
  payload: CollaborationPayload | null;
  onClose: () => void;
  onOpenSharing?: () => void;
};

export function HistoryCollaborationDetailModal({ payload, onClose, onOpenSharing }: Props) {
  if (!payload) return null;

  const handleCopyLink = async (token: string) => {
    const copied = await copyTextToClipboard(shareLinkFullUrl(token));
    if (copied) toast.success("Link copied");
    else toast.error("Could not copy link");
  };

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-secondary bg-primary shadow-xl">
        <div className="flex items-center justify-between gap-3 border-b border-secondary px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-lg border border-secondary bg-primary shadow-xs">
              {payload.kind === "share" ? (
                <Link01 className="size-5 text-fg-quaternary" />
              ) : (
                <UserPlus01 className="size-5 text-fg-quaternary" />
              )}
            </div>
            <h3 className="text-lg font-semibold text-primary">
              {payload.kind === "share" ? "Share link" : "Collaborator invite"}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1 text-fg-quaternary hover:text-fg-quaternary_hover"
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="space-y-4 px-5 py-5">
          {payload.kind === "share" ? (
            <>
              <div>
                <p className="text-sm font-medium text-secondary">Link access</p>
                <p className="mt-1 text-sm text-primary">{payload.access}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-secondary">Share URL</p>
                <div className="mt-1.5 flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2.5">
                  <span className="min-w-0 flex-1 truncate text-sm text-tertiary">
                    {shareLinkFullUrl(payload.token)}
                  </span>
                  <button
                    type="button"
                    aria-label="Copy link"
                    onClick={() => {
                      void handleCopyLink(payload.token);
                    }}
                    className="shrink-0 text-fg-quaternary hover:text-fg-quaternary_hover"
                  >
                    <Copy01 className="size-4" />
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <div>
                <p className="text-sm font-medium text-secondary">Name</p>
                <p className="mt-1 text-sm text-primary">{payload.name}</p>
              </div>
              {payload.email ? (
                <div>
                  <p className="text-sm font-medium text-secondary">Email</p>
                  <p className="mt-1 text-sm text-primary">{payload.email}</p>
                </div>
              ) : null}
              {payload.department ? (
                <div>
                  <p className="text-sm font-medium text-secondary">Department</p>
                  <p className="mt-1 text-sm text-primary">{payload.department}</p>
                </div>
              ) : null}
              <div>
                <p className="text-sm font-medium text-secondary">Access level</p>
                <p className="mt-1 text-sm text-primary">{payload.access}</p>
              </div>
            </>
          )}
        </div>

        <div className="flex items-center justify-end gap-3 border-t border-secondary px-5 py-4">
          {onOpenSharing ? (
            <Button
              color="secondary"
              size="md"
              onClick={() => {
                onOpenSharing();
                onClose();
              }}
            >
              Open sharing
            </Button>
          ) : null}
          <Button color="primary" size="md" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}
