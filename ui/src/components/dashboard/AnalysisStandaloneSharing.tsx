import { useCallback, useEffect, useState } from "react";
import { useServerFn } from "@tanstack/react-start";
import { LoaderCircle, Share2, Trash2, Users } from "lucide-react";
import { Button } from "@/components/base/buttons/button";
import { Checkbox } from "@/components/base/checkbox/checkbox";
import {
  createAnalysisShare,
  getAnalysisShareSettings,
  removeAnalysisShare,
  updateAnalysisShare,
} from "@/lib/analysis-sharing.functions";

type ShareAccess = "view" | "comment" | "edit";
type SharingSettings = {
  viewerProfileId: string;
  isOwner: boolean;
  teams: Array<{ id: string; name: string }>;
  members: Array<{ id: string; name: string; email: string | null; avatar: string | null }>;
  shares: Array<{
    id: string;
    targetType: "team" | "profile";
    targetId: string;
    targetName: string;
    accessLevel: ShareAccess;
    canShare: boolean;
    sharedBy: string | null;
  }>;
};

interface Props {
  analysisId: string;
  resourceType?: "analysis" | "project";
  open: boolean;
  onChange?: () => void;
}

const ACCESS_LABELS: Record<ShareAccess, string> = {
  view: "View",
  comment: "Comment",
  edit: "Edit",
};

export function AnalysisStandaloneSharing({ analysisId, resourceType = "analysis", open, onChange }: Props) {
  const loadFn = useServerFn(getAnalysisShareSettings);
  const createFn = useServerFn(createAnalysisShare);
  const updateFn = useServerFn(updateAnalysisShare);
  const removeFn = useServerFn(removeAnalysisShare);
  const [settings, setSettings] = useState<SharingSettings | null>(null);
  const [target, setTarget] = useState("");
  const [accessLevel, setAccessLevel] = useState<ShareAccess>("view");
  const [canShare, setCanShare] = useState(false);
  const [loading, setLoading] = useState(false);
  const [savingTarget, setSavingTarget] = useState(false);
  const [actingOnShare, setActingOnShare] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSettings((await loadFn({ data: { analysisId, resourceType } })) as SharingSettings);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load analysis sharing.");
    } finally {
      setLoading(false);
    }
  }, [analysisId, resourceType, loadFn]);

  useEffect(() => {
    if (open) void reload();
    else {
      setSettings(null);
      setTarget("");
      setCanShare(false);
      setError(null);
    }
  }, [open, reload]);

  const grant = async () => {
    if (!target) return;
    setSavingTarget(true);
    setError(null);
    try {
      const [kind, id] = target.split(":", 2);
      await createFn({
        data: {
          analysisId,
          resourceType,
          ...(kind === "team" ? { teamId: id } : { recipientProfileId: id }),
          accessLevel,
          canShare: settings?.isOwner ? canShare : false,
        },
      });
      setTarget("");
      setCanShare(false);
      await reload();
      onChange?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not share this analysis.");
    } finally {
      setSavingTarget(false);
    }
  };

  const updateGrant = async (
    grantId: string,
    patch: { accessLevel: ShareAccess; canShare?: boolean },
  ) => {
    setActingOnShare(grantId);
    setError(null);
    try {
      await updateFn({ data: { resourceType, shareId: grantId, ...patch } });
      await reload();
      onChange?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not update this share.");
    } finally {
      setActingOnShare(null);
    }
  };

  const revokeGrant = async (grantId: string) => {
    setActingOnShare(grantId);
    setError(null);
    try {
      await removeFn({ data: { resourceType, shareId: grantId } });
      await reload();
      onChange?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not remove this share.");
    } finally {
      setActingOnShare(null);
    }
  };

  if (!open) return null;
  return (
    <section
      className="mt-5 border-t border-secondary px-5 pt-4 pb-4"
      aria-label={`${resourceType === "project" ? "Project" : "Standalone analysis"} access`}
    >
      <div className="flex items-center gap-2">
        <Users className="size-4 text-fg-quaternary" />
        <h3 className="text-sm font-semibold text-primary">Team collaboration</h3>
      </div>
      <p className="mt-1 text-xs text-tertiary">
        Share this {resourceType === "project" ? "project" : "standalone analysis"} with a teammate or a whole team. Team access follows current
        membership.
      </p>

      {loading && !settings ? (
        <div className="mt-3 flex items-center gap-2 text-xs text-tertiary">
          <LoaderCircle className="size-4 animate-spin" /> Loading access…
        </div>
      ) : settings ? (
        <>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <select
              aria-label="Choose a teammate or team"
              value={target}
              onChange={(event) => setTarget(event.target.value)}
              className="col-span-2 min-w-0 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-brand"
            >
              <option value="">Choose a teammate or team…</option>
              <optgroup label="Team members">
                {settings.members.map((member) => (
                  <option key={member.id} value={`profile:${member.id}`}>
                    {member.name}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Teams">
                {settings.teams.map((team) => (
                  <option key={team.id} value={`team:${team.id}`}>
                    {team.name}
                  </option>
                ))}
              </optgroup>
            </select>
            <select
              aria-label="Access level"
              value={accessLevel}
              onChange={(event) => setAccessLevel(event.target.value as ShareAccess)}
              className="min-w-0 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-brand"
            >
              {Object.entries(ACCESS_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
            </select>
            <Button
              color="primary"
              size="sm"
              isDisabled={!target || savingTarget}
              isLoading={savingTarget}
              iconLeading={Share2}
              onClick={() => void grant()}
            >
              Share
            </Button>
          </div>
          {settings.isOwner && (
            <Checkbox
              className="mt-3"
              size="sm"
              isSelected={canShare}
              onChange={setCanShare}
              label="Allow this recipient to share with other teammates"
            />
          )}

          <div className="mt-4 space-y-2">
            <p className="text-sm font-medium text-secondary">People and teams with access</p>
            {settings.shares.length ? (
              settings.shares.map((share) => {
                const canEditShare =
                  settings.isOwner || share.sharedBy === settings.viewerProfileId;
                return (
                  <div
                    key={share.id}
                    className="flex min-w-0 items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2.5 shadow-xs"
                  >
                    <span className="min-w-0 flex-1 truncate text-sm text-primary">
                      {share.targetName}
                      {share.targetType === "team" && (
                        <span className="ml-1 text-[10px] uppercase text-tertiary">team</span>
                      )}
                    </span>
                    {canEditShare ? (
                      <>
                        <select
                          aria-label={`Access for ${share.targetName}`}
                          value={share.accessLevel as ShareAccess}
                          disabled={actingOnShare === share.id}
                          onChange={(event) =>
                            void updateGrant(share.id, {
                              accessLevel: event.target.value as ShareAccess,
                            })
                          }
                          className="rounded-lg border border-secondary bg-primary px-2 py-2 text-sm text-primary shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-brand disabled:opacity-50"
                        >
                          {Object.entries(ACCESS_LABELS).map(([value, label]) => (
                              <option key={value} value={value}>
                                {label}
                              </option>
                            ))}
                        </select>
                        {settings.isOwner && (
                          <Checkbox
                            size="sm"
                            aria-label={`Allow ${share.targetName} to reshare`}
                            isSelected={share.canShare}
                            isDisabled={actingOnShare === share.id}
                            onChange={(selected) =>
                              void updateGrant(share.id, {
                                accessLevel: share.accessLevel as ShareAccess,
                                canShare: selected,
                              })
                            }
                          />
                        )}
                        <button
                          type="button"
                          aria-label={`Remove ${share.targetName}`}
                          disabled={actingOnShare === share.id}
                          onClick={() => void revokeGrant(share.id)}
                          className="flex size-9 shrink-0 items-center justify-center rounded-lg text-fg-quaternary outline-none hover:bg-primary_hover hover:text-error-primary focus-visible:ring-2 focus-visible:ring-brand disabled:opacity-50"
                        >
                          {actingOnShare === share.id ? (
                            <LoaderCircle className="size-4 animate-spin" />
                          ) : (
                            <Trash2 className="size-4" />
                          )}
                        </button>
                      </>
                    ) : (
                      <span className="text-xs text-tertiary">
                        {ACCESS_LABELS[share.accessLevel as ShareAccess] ?? "View"}
                      </span>
                    )}
                  </div>
                );
              })
            ) : (
              <p className="text-xs text-tertiary">No one else has access yet.</p>
            )}
          </div>
        </>
      ) : null}
      {error && (
        <p role="alert" className="mt-2 text-xs text-error-primary">
          {error}
        </p>
      )}
    </section>
  );
}
