// Start a swarm mission and follow it live.
//
// Polling, not SSE: every call carries the Bearer header (EventSource cannot),
// records are append-only with a dense `seq`, and a two-clone laptop mission is
// tens of records. The hook asks for `records?after=<cursor>` every `pollMs`
// until the page says `done`, folding each page through `reduceSwarm`.

import { useCallback, useEffect, useRef, useState } from "react";
import { backendApi } from "@/lib/api/backendApi";
import {
  emptySwarm,
  reduceSwarm,
  type SwarmMissionRequest,
  type SwarmMissionView,
  type SwarmStartResponse,
} from "./swarm";

export type SwarmApi = Pick<typeof backendApi, "startSwarmMission" | "getSwarmRecords" | "getSwarmMission">;

export interface UseSwarmMission {
  view: SwarmMissionView;
  /** The POST response: the grant as the server stated it. */
  started: SwarmStartResponse | null;
  starting: boolean;
  polling: boolean;
  error: string | null;
  start: (body: SwarmMissionRequest) => Promise<void>;
  /** Reopen a mission by id: its records are read from the start and, if it is still running, followed. */
  open: (missionId: string) => Promise<void>;
  reset: () => void;
}

export const POLL_MS = 1000;

export function useSwarmMission({ api = backendApi, pollMs = POLL_MS }: { api?: SwarmApi; pollMs?: number } = {}): UseSwarmMission {
  const [view, setView] = useState<SwarmMissionView>(emptySwarm);
  const [started, setStarted] = useState<SwarmStartResponse | null>(null);
  const [starting, setStarting] = useState(false);
  const [missionId, setMissionId] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The cursor lives in a ref so the poll reads the latest without a re-render race.
  const cursor = useRef(0);

  const reset = useCallback(() => {
    setMissionId(null);
    setPolling(false);
    setStarted(null);
    setError(null);
    cursor.current = 0;
    setView(emptySwarm());
  }, []);

  const start = useCallback(
    async (body: SwarmMissionRequest) => {
      reset();
      setStarting(true);
      try {
        const res = await api.startSwarmMission(body);
        setStarted(res);
        setMissionId(res.mission_id);
        setPolling(true);
      } catch (e) {
        setError((e as Error)?.message || "Could not start the mission.");
      } finally {
        setStarting(false);
      }
    },
    [api, reset],
  );

  const open = useCallback(
    async (id: string) => {
      reset();
      try {
        await api.getSwarmMission(id); // 404 for a mission that is not the caller's
      } catch (e) {
        setError((e as Error)?.message || "Could not load the mission.");
        return;
      }
      setMissionId(id);
      setPolling(true);
    },
    [api, reset],
  );

  useEffect(() => {
    if (!missionId || !polling) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const page = await api.getSwarmRecords(missionId, cursor.current);
        if (!live) return;
        if (page.records.length) {
          // Advance the cursor now, from the page itself. Moved inside the
          // setView updater, it waited for React to run the updater, and under
          // load the next poll fired first and asked again from the old cursor
          // (swarm-demo: polls 0, 6, 12, 18, 18, 24, 30). The reducer drops
          // records at or below its own cursor, so the view never doubled.
          const seqs = page.records.map((r) => r.seq).filter((n): n is number => typeof n === "number");
          if (seqs.length) cursor.current = Math.max(cursor.current, ...seqs);
          setView((prev) => reduceSwarm(page.records, prev));
        }
        if (page.done) {
          setPolling(false);
          return;
        }
        // Not done but nothing new past the cursor and the server says it is finished:
        // the mission has no records (a request refused before anything ran).
        if (page.status && page.status !== "running" && !page.records.length) {
          setView((prev) => (prev.status === "idle" ? { ...prev, missionId, status: page.status === "failed" ? "failed" : "done" } : prev));
          setPolling(false);
          return;
        }
      } catch (e) {
        if (!live) return;
        setError((e as Error)?.message || "Lost the mission.");
        setPolling(false);
        return;
      }
      timer = setTimeout(tick, pollMs);
    };
    void tick();
    return () => {
      live = false;
      if (timer) clearTimeout(timer);
    };
  }, [api, missionId, polling, pollMs]);

  return { view, started, starting, polling, error, start, open, reset };
}
