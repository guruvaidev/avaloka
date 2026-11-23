// The /instant page body: Start → Preview → Ask, on one page. The route wraps
// it in the app shell; tests render it alone with a mocked fetch. `api` is the
// instant API module, injectable so a test can stub one call. When the
// question comes back `ready`, `onReady` takes the run to the report page
// (/instant/:runId); the route wires it to the router.
import { useEffect, useRef, useState } from "react";
import {
  EXAMPLE_QUESTIONS,
  INSTANT_ACCEPT,
  SAMPLE_DATASETS,
  instantApi,
  issuesFromProfile,
  profileStatusOf,
  rejectFile,
  type InstantApi,
  type InstantAskReady,
  type InstantAskResponse,
  type InstantLimits,
  type InstantUpload,
  type SampleDataset,
} from "@/lib/api/instant";
import { DropFirst } from "./DropFirst";
import { ProfilePreview } from "./ProfilePreview";
import { AskBox } from "./AskBox";

export const INSTANT_TAGLINE = "One question → verified report";

export function InstantFlow({
  api = instantApi,
  onReady,
}: {
  api?: InstantApi;
  onReady?: (ready: InstantAskReady) => void;
}) {
  const [limits, setLimits] = useState<InstantLimits | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<InstantUpload | null>(null);
  const [profileIssues, setProfileIssues] = useState<string[]>([]);
  const [profileStatus, setProfileStatus] = useState<InstantUpload["profileStatus"]>("skipped");
  const [profileError, setProfileError] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<InstantAskResponse | null>(null);
  const pollRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let alive = true;
    api.fetchInstantLimits().then((l) => alive && setLimits(l));
    return () => {
      alive = false;
    };
  }, [api]);

  useEffect(() => () => pollRef.current?.abort(), []);

  const start = async (file: File) => {
    const lim = limits ?? (await api.fetchInstantLimits());
    const why = rejectFile(file, lim, INSTANT_ACCEPT);
    if (why) {
      setError(why);
      return;
    }
    setError(null);
    setBusy(`Uploading ${file.name}…`);
    try {
      const up = await api.uploadForInstant(file);
      setUpload(up);
      setProfileStatus(up.profileStatus);
      setProfileError(up.profileError);
      setProfileIssues([]);
      setAnswer(null);
      setAskError(null);
      if (up.profileStatus === "pending") {
        pollRef.current?.abort();
        const ctrl = new AbortController();
        pollRef.current = ctrl;
        api.waitForInstantProfile(up.datasetId, { signal: ctrl.signal }).then((res) => {
          if (ctrl.signal.aborted || !res) return;
          setProfileStatus(profileStatusOf(res.profiling_status));
          setProfileError(typeof res.profiling_error === "string" ? res.profiling_error : null);
          setProfileIssues(issuesFromProfile(res));
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(null);
    }
  };

  const startSample = async (sample: SampleDataset) => {
    setError(null);
    setBusy(`Loading ${sample.label}…`);
    try {
      const file = await api.loadSampleFile(sample);
      await start(file);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sample unavailable");
      setBusy(null);
    }
  };

  const ask = async (question: string) => {
    if (!upload) return;
    setAsking(true);
    setAskError(null);
    try {
      const res = await api.askInstant({ dataset_id: upload.datasetId, question });
      setAnswer(res);
      if (res.status === "ready") onReady?.(res);
    } catch (e) {
      setAskError(e instanceof Error ? e.message : "Could not ask");
    } finally {
      setAsking(false);
    }
  };

  const reset = () => {
    pollRef.current?.abort();
    setUpload(null);
    setAnswer(null);
    setAskError(null);
    setError(null);
  };

  const issues = [...(upload?.preview.issues ?? []), ...profileIssues];

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-8 sm:px-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold text-primary">{INSTANT_TAGLINE}</h1>
        <p className="text-sm text-tertiary">
          Drop a file or pick a sample, see its profile, then ask.
        </p>
      </header>

      {!upload ? (
        <DropFirst
          accept={INSTANT_ACCEPT}
          limits={limits}
          samples={SAMPLE_DATASETS}
          busy={busy}
          error={error}
          onFile={start}
          onSample={startSample}
        />
      ) : (
        <>
          <ProfilePreview
            fileName={upload.fileName}
            preview={upload.preview}
            profileStatus={profileStatus}
            profileError={profileError}
            issues={issues}
          />
          <AskBox
            examples={EXAMPLE_QUESTIONS}
            busy={asking}
            error={askError}
            result={answer}
            onAsk={ask}
          />
          <button
            type="button"
            onClick={reset}
            className="self-start text-sm text-tertiary underline-offset-2 hover:underline"
          >
            Start over with another file
          </button>
        </>
      )}
    </main>
  );
}
