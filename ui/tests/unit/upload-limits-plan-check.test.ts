// GET /api/limits -> backendApi.getUploadLimits keeps plan_check_unavailable and
// feature_states, which the upload dialogs read to tell "your plan could not be
// checked" apart from "your plan does not include this".
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/integrations/supabase/external-client", () => ({
  supabase: { auth: { getSession: async () => ({ data: { session: { access_token: "token" } } }) } },
}));

import { backendApi } from "@/lib/api/backendApi";

afterEach(() => vi.unstubAllGlobals());

function answer(body: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status: 200 })));
}

describe("getUploadLimits", () => {
  it("passes plan_check_unavailable and feature_states through", async () => {
    answer({
      plan: "free",
      limits: { max_file_bytes: 8 },
      features: { model_inference: false },
      feature_states: { model_inference: "unavailable" },
      plan_check_unavailable: true,
    });
    const limits = await backendApi.getUploadLimits();
    expect(limits.plan).toBe("free");
    expect(limits.max_file_bytes).toBe(8);
    expect(limits.plan_check_unavailable).toBe(true);
    expect(limits.feature_states).toEqual({ model_inference: "unavailable" });
  });

  it("an older server without the field reads as checked", async () => {
    answer({ plan: "free", limits: { max_file_bytes: 8 } });
    expect((await backendApi.getUploadLimits()).plan_check_unavailable).toBe(false);
  });
});
