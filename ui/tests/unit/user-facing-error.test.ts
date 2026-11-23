import { describe, expect, it } from "vitest";
import { looksLikeRawError, RUN_LOG_ERROR_LINE, scrubLogLines, userFacingError } from "@/lib/user-facing-error";

describe("userFacingError", () => {
  it("passes plain words through", () => {
    expect(userFacingError("The scheduled task failed.", "fallback")).toBe("The scheduled task failed.");
  });

  it("replaces a traceback, a repr or an exception line with the fallback", () => {
    for (const raw of [
      'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\nKeyError: \'a\'',
      "ValueError('bad value')",
      "KeyError: 'revenue_2026'",
    ]) {
      expect(userFacingError(raw, "fallback")).toBe("fallback");
    }
  });

  it("falls back on empty or non-string input", () => {
    expect(userFacingError("", "fb")).toBe("fb");
    expect(userFacingError(null, "fb")).toBe("fb");
    expect(userFacingError({ a: 1 }, "fb")).toBe("fb");
  });
});

describe("scrubLogLines", () => {
  it("replaces each raw entry once and keeps the rest", () => {
    const out = scrubLogLines(["[start] t0", "KeyError: 'a'", 'File "x.py", line 2', "[assistant] done"]);
    expect(out).toEqual(["[start] t0", RUN_LOG_ERROR_LINE, "[assistant] done"]);
    expect(out.some(looksLikeRawError)).toBe(false);
  });
});
