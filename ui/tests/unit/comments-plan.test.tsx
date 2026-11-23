// Owner ruling 2026-11-01: commenting on analyses and reports, and notifying
// other people, are Professional and Enterprise. Free can read what is shared
// with it and still receives notifications, but cannot write comments.
//   * usePlanFeatures().canComment is the UI flag, failing closed while loading.
//   * The comment boxes (insight card and report panel) show the plan note
//     instead of a composer, and no Reply, without canComment.
//   * The server functions that write with the admin client refuse a caller
//     below Professional before touching it.
import { render, renderHook, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const plan = vi.hoisted(() => ({ value: "free" as string | undefined, loading: false }));
const admin = vi.hoisted(() => ({ from: [] as string[] }));

vi.mock("@/lib/use-active-plan", () => ({
  useActivePlan: () => ({
    data: plan.value === undefined ? undefined : { plan: plan.value },
    isLoading: plan.loading,
  }),
}));

const COMMENT = {
  id: "c1",
  text: "First comment",
  created_at: "2026-11-01T10:00:00Z",
  attachments: [],
  replies: [],
  author: { id: "p-other", full_name: "Someone Else", avatar_url: null },
};

vi.mock("@/lib/current-profile", () => ({ getCurrentProfileId: async () => "p-me" }));
vi.mock("@/lib/analysis-insight-comments", () => ({
  addAnalysisInsightComment: vi.fn(),
  addAnalysisInsightCommentReply: vi.fn(),
  formatAttachmentSize: () => "",
  formatInsightCommentTime: () => "now",
  formatInsightCommentRelative: () => "now",
  initialsFromName: () => "SE",
  loadAnalysisInsightComments: async () => [COMMENT],
  MAX_INSIGHT_COMMENT_ATTACHMENTS: 5,
  MAX_INSIGHT_COMMENT_ATTACHMENT_BYTES: 25 * 1024 * 1024,
  uploadInsightCommentAttachment: vi.fn(),
}));
vi.mock("@/lib/report-comments", () => ({
  addReportComment: vi.fn(),
  addReportCommentReply: vi.fn(),
  loadReportComments: async () => [COMMENT],
  uploadReportCommentAttachment: vi.fn(),
}));
vi.mock("@/components/dashboard/MentionTextInput", () => ({
  MentionTextInput: (p: { placeholder?: string }) => <textarea placeholder={p.placeholder} />,
  useMentionCandidates: () => [],
  extractMentionedProfileIds: () => [],
}));
vi.mock("@/components/dashboard/InsightCommentAttachments", () => ({
  InsightCommentAttachmentList: () => null,
  insightCommentActionText: (text: string) => text,
}));

// Server functions: handlers are called directly; the admin client records use.
vi.mock("@tanstack/react-start", () => {
  const builder = () => {
    const b: any = { middleware: () => b, inputValidator: () => b, handler: (fn: any) => fn };
    return b;
  };
  return { createServerFn: builder, createMiddleware: builder };
});
vi.mock("@/integrations/supabase/auth-middleware", () => ({ requireSupabaseAuth: {} }));
vi.mock("@/integrations/supabase/primary-auth-middleware", () => ({ requireSupabaseAuth: {} }));
function chain(): any {
  const c: any = new Proxy(
    {},
    {
      get: (_t, prop) => {
        if (prop === "then") return undefined;
        if (prop === "data") return null;
        if (prop === "error") return null;
        if (prop === "single" || prop === "maybeSingle") return async () => ({ data: null, error: null });
        return () => c;
      },
    },
  );
  return c;
}
const adminClient = {
  from: (t: string) => {
    admin.from.push(t);
    return chain();
  },
};
vi.mock("@/integrations/supabase/client.server", () => ({ supabaseAdmin: adminClient }));
vi.mock("@/integrations/supabase/primary-client.server", () => ({ supabaseAdmin: adminClient }));

import { supabase } from "@/integrations/supabase/client";
import { usePlanFeatures } from "@/lib/use-plan-features";
import { InsightCommentCard } from "@/components/dashboard/AddCommentPopover";
import { ReportCommentsPanel } from "@/components/dashboard/ReportCommentsPanel";
import { createReportComment } from "@/lib/report-comment-create.functions";
import { fanOutCommentNotifications } from "@/lib/comment-notify.functions";
import { fanOutReportCommentNotifications } from "@/lib/report-comment-notify.functions";

// The note text, pinned literally so a missing constant fails on an assertion.
const COMMENTS_NEED_PRO = "Commenting is included in the Professional and Enterprise plans.";
const REFUSED = "Comments, and notifying other people, need a Professional plan or higher.";

function setPlan(tier: string | undefined, loading = false) {
  plan.value = tier;
  plan.loading = loading;
}

beforeEach(() => {
  setPlan("free");
  admin.from.length = 0;
});

describe("usePlanFeatures().canComment", () => {
  it.each([
    ["free", false],
    ["professional", true],
    ["enterprise", true],
    [undefined, false],
  ])("plan %s -> %s", (tier, expected) => {
    setPlan(tier as string | undefined);
    expect(renderHook(() => usePlanFeatures()).result.current.canComment).toBe(expected);
  });

  it("fails closed while the plan loads", () => {
    setPlan("enterprise", true);
    expect(renderHook(() => usePlanFeatures()).result.current.canComment).toBe(false);
  });
});

describe("comment boxes by plan", () => {
  it.each([["insight card"], ["report panel"]])("%s: Free reads comments but gets the plan note, no composer, no Reply", async (which) => {
    setPlan("free");
    if (which === "insight card") render(<InsightCommentCard analysisId="a1" />);
    else render(<ReportCommentsPanel onClose={() => {}} reportId="r1" />);
    await waitFor(() => expect(screen.queryByText("First comment")).not.toBeNull());
    expect(screen.queryByText(COMMENTS_NEED_PRO)).not.toBeNull();
    expect(screen.queryByPlaceholderText(/Write a comment/)).toBeNull();
    expect(screen.queryByText("Reply")).toBeNull();
  });

  it.each([
    ["insight card", "professional"],
    ["insight card", "enterprise"],
    ["report panel", "professional"],
    ["report panel", "enterprise"],
  ])("%s: %s gets the composer and no plan note", async (which, tier) => {
    setPlan(tier);
    // Since 4140dd96 a Professional insight card also needs Comment or Edit
    // access to the analysis (useResourceAccess -> analysis_share_access_level,
    // the same check the comment insert policy makes). The owner has level 3.
    const rpc = vi.spyOn(supabase as unknown as { rpc: (...a: unknown[]) => unknown }, "rpc").mockResolvedValue({ data: 3, error: null } as never);
    if (which === "insight card") render(<InsightCommentCard analysisId="a1" />);
    else render(<ReportCommentsPanel onClose={() => {}} reportId="r1" />);
    await waitFor(() => expect(screen.queryByText("First comment")).not.toBeNull());
    await waitFor(() => expect(screen.queryByPlaceholderText(/Write a comment/)).not.toBeNull());
    expect(screen.queryByText(COMMENTS_NEED_PRO)).toBeNull();
    if (which === "insight card") expect(screen.queryByText("Reply")).not.toBeNull();
    if (which === "insight card" && tier === "professional") {
      expect(rpc).toHaveBeenCalledWith("analysis_share_access_level", { p_analysis_id: "a1" });
    }
  });

  it("insight card: professional with view-only access gets no composer and the access note", async () => {
    setPlan("professional");
    vi.spyOn(supabase as unknown as { rpc: (...a: unknown[]) => unknown }, "rpc").mockResolvedValue({ data: 1, error: null } as never);
    render(<InsightCommentCard analysisId="a1" />);
    await waitFor(() => expect(screen.queryByText("First comment")).not.toBeNull());
    await waitFor(() => expect(screen.queryByText("Comment or Edit access is required to add comments.")).not.toBeNull());
    expect(screen.queryByText(COMMENTS_NEED_PRO)).toBeNull();
    expect(screen.queryByPlaceholderText(/Write a comment/)).toBeNull();
  });
});

/** The caller's own client, answering the plan RPCs like the database would. */
function userClient(collab: boolean) {
  const calls: string[] = [];
  return {
    calls,
    rpc: vi.fn(async (name: string, args?: any) => {
      calls.push(args?.p_action ? `${name}:${args.p_action}:${args.p_required}` : name);
      if (name === "collab_features_enabled") return { data: collab, error: null };
      if (name === "log_org_feature_bypass") return { data: null, error: null };
      return { data: false, error: null };
    }),
    from: () => ({
      select: () => ({ eq: () => ({ maybeSingle: async () => ({ data: { id: "r1" }, error: null }) }) }),
    }),
  };
}

const SERVER_FNS: [string, (ctx: any) => Promise<unknown>][] = [
  [
    "createReportComment",
    (context) =>
      (createReportComment as any)({
        data: { reportId: "r1", parentId: null, content: "hi", attachments: [] },
        context,
      }),
  ],
  ["fanOutCommentNotifications", (context) => (fanOutCommentNotifications as any)({ data: { commentId: "c1" }, context })],
  [
    "fanOutReportCommentNotifications",
    (context) => (fanOutReportCommentNotifications as any)({ data: { commentId: "c1" }, context }),
  ],
];

describe("server functions refuse below Professional before using the admin client", () => {
  it.each(SERVER_FNS)("%s: Free is refused and nothing is read or written", async (_name, call) => {
    const supabase = userClient(false);
    await expect(call({ supabase, userId: "u1" })).rejects.toThrow(REFUSED);
    expect(admin.from).toEqual([]);
    expect(supabase.calls).toEqual(["collab_features_enabled"]);
  });

  it.each(SERVER_FNS)("%s: Professional passes the check, which is recorded at the Professional level", async (name, call) => {
    const supabase = userClient(true);
    await call({ supabase, userId: "u1" }).catch((e: Error) => {
      expect(e.message).not.toBe(REFUSED);
    });
    expect(supabase.calls.slice(0, 2)).toEqual([
      "collab_features_enabled",
      `log_org_feature_bypass:${name}:professional`,
    ]);
    expect(admin.from.length).toBeGreaterThan(0);
  });
});
