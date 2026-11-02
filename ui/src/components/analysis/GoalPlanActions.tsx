import { useCallback, useRef, useState } from "react";
import {
  decideOutgoing,
  isGoAhead,
  planMessage,
  type GoalPlan,
  type PlanDataset,
} from "@/lib/goal-plan";

export type GoalChatLine = { role: "user" | "ai"; content: string };

/**
 * Goal-first gate in front of the chat's turn runner. The first message of a
 * new chat gets a code-written plan and runs nothing; the goal is sent to the
 * backend only when the user approves (the Go button, or a "go" reply). Every
 * later message goes straight to runTurn.
 */
export function useGoalFirstGate({
  runTurn,
  appendLines,
  datasets,
  isNewChat,
}: {
  /** Sends one turn to the backend. showUser=false: the message is already on screen. */
  runTurn: (content: string, opts?: { showUser?: boolean }) => Promise<boolean>;
  appendLines: (lines: GoalChatLine[]) => void;
  datasets: PlanDataset[];
  /** No turn has run in this thread yet. */
  isNewChat: boolean;
}) {
  const [pending, setPending] = useState<GoalPlan | null>(null);
  const pendingRef = useRef<GoalPlan | null>(null);
  pendingRef.current = pending;

  const setPlan = useCallback((plan: GoalPlan | null) => {
    pendingRef.current = plan;
    setPending(plan);
  }, []);

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

  /** Drops a waiting plan, e.g. when the user switches to another thread. */
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
