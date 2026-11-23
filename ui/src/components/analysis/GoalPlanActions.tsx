import { useCallback, useEffect, useRef, useState } from "react";
import {
  decideOutgoing,
  isGoAhead,
  planMessage,
  readStoredPlan,
  writeStoredPlan,
  type GoalPlan,
  type PlanDataset,
} from "@/lib/goal-plan";

export type GoalChatLine = { role: "user" | "ai"; content: string };

/**
 * Goal-first gate in front of the chat's turn runner. The first message of a
 * new chat gets a code-written plan and runs nothing; the goal is sent to the
 * backend only when the user approves (the Go button, or a "go" reply). Every
 * later message goes straight to runTurn.
 *
 * The waiting plan is kept in this tab's sessionStorage under storageKey (the
 * thread id), so a page reload restores it and its Go action. It is not stored
 * on the server: nothing has been sent there yet.
 */
export function useGoalFirstGate({
  runTurn,
  appendLines,
  datasets,
  isNewChat,
  storageKey = null,
}: {
  /** Sends one turn to the backend. showUser=false: the message is already on screen. */
  runTurn: (content: string, opts?: { showUser?: boolean }) => Promise<boolean>;
  appendLines: (lines: GoalChatLine[]) => void;
  datasets: PlanDataset[];
  /** No turn has run in this thread yet. */
  isNewChat: boolean;
  /** The thread the plan belongs to; null keeps it in memory only. */
  storageKey?: string | null;
}) {
  const [pending, setPending] = useState<GoalPlan | null>(() =>
    storageKey ? readStoredPlan(storageKey) : null,
  );
  const pendingRef = useRef<GoalPlan | null>(pending);
  pendingRef.current = pending;
  const keyRef = useRef<string | null>(storageKey);

  const setPlan = useCallback((plan: GoalPlan | null) => {
    pendingRef.current = plan;
    setPending(plan);
    if (keyRef.current) writeStoredPlan(keyRef.current, plan);
  }, []);

  // A thread change shows that thread's waiting plan, or none. The first key
  // after mount (the thread id arriving after a reload) adopts a plan made
  // before the id was known instead of dropping it.
  useEffect(() => {
    const previous = keyRef.current;
    if (previous === storageKey) return;
    keyRef.current = storageKey;
    if (!storageKey) {
      setPlan(null);
      return;
    }
    const stored = readStoredPlan(storageKey);
    if (stored) setPlan(stored);
    else if (previous === null && pendingRef.current) setPlan(pendingRef.current);
    else setPlan(null);
  }, [storageKey, setPlan]);

  const approve = useCallback(async () => {
    const plan = pendingRef.current;
    if (!plan) return false;
    setPlan(null);
    return runTurn(plan.goal, { showUser: false });
  }, [runTurn, setPlan]);

  const send = useCallback(
    async (content: string) => {
      const text = content.trim();
      if (!text) return false;
      const current = pendingRef.current;
      const decision = decideOutgoing(text, { goalFirst: isNewChat, pending: current, datasets });
      if (decision.action === "say") {
        appendLines([
          { role: "user", content: text },
          { role: "ai", content: decision.content },
        ]);
        return true;
      }
      if (decision.action === "plan") {
        setPlan(decision.plan);
        appendLines([
          { role: "user", content: text },
          { role: "ai", content: planMessage(decision.plan) },
        ]);
        return true;
      }
      if (current && isGoAhead(text)) {
        appendLines([{ role: "user", content: text }]);
        return approve();
      }
      return runTurn(decision.content);
    },
    [approve, appendLines, datasets, isNewChat, runTurn, setPlan],
  );

  /** Drops a waiting plan. */
  const reset = useCallback(() => setPlan(null), [setPlan]);

  return { pending, send, approve, reset };
}

/** The plan's approval control, shown under the plan message while it waits. */
export function GoalPlanActions({ onGo, disabled }: { onGo: () => void; disabled?: boolean }) {
  return (
    <div className="mb-5 ml-9 flex items-center gap-2" data-testid="goal-plan-actions">
      <button
        type="button"
        onClick={onGo}
        disabled={disabled}
        className="cursor-pointer rounded-md bg-[#1565ef] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[#1a5fe0] disabled:cursor-not-allowed disabled:opacity-50"
      >
        Go
      </button>
      <span className="text-xs text-tertiary">Nothing runs until you say go.</span>
    </div>
  );
}
