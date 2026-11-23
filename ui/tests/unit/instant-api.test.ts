// The /instant API module: the preview is read from the responses (never
// measured), limits come from /api/limits with the upload modal's fallback,
// the upload, the profile poll, the run and its export go through the
// same-origin proxy, and the question stub answers "coming soon" without a
// network call.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/integrations/supabase/external-client", () => ({
  supabase: {
    auth: {
      getSession: vi.fn(async () => ({ data: { session: { access_token: "tok-instant" } } })),
    },
  },
}));

import { DEV_TOKEN_KEY } from "@/lib/api/backendApi";
import {
  FALLBACK_MAX_FILE_BYTES,
  INSTANT_ACCEPT,
  INSTANT_RUN_PATH,
  SAMPLE_DATASETS,
  askInstant,
  fetchInstantExport,
  fetchInstantLimits,
  fetchInstantRun,
  issuesFromProfile,
  loadSampleFile,
  previewFromUpload,
  profileStatusOf,
  rejectFile,
  uploadForInstant,
  waitForInstantProfile,
} from "@/lib/api/instant";

function ok(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** A synthetic upload response: the shapes app/agents/sampling_async.py writes. */
const uploadFixture = {
  dataset_id: "ds-unit-1",
  session_id: "s1",
  thread_id: "t1",
  schema: ["zq_alpha", "zq_beta", "zq_gamma"],
  samples: [],
  ddl_schema: "",
  rows_sampled: 3,
  profiling_status: "pending",
  sample_statistics: {
    data_shape: { rows: 1234, columns: 3 },
    column_statistics: {
      zq_alpha: {
        type: "numeric",
        missing_ratio: 0,
        n_unique: 1234,
        total_count: 1234,
        null_count: 0,
      },
      zq_beta: {
        type: "categorical",
        missing_ratio: 0.4167,
        n_unique: 7,
        total_count: 1234,
        null_count: 514,
      },
      zq_gamma: { type: "categorical", missing_pct: 2.5, missing_ratio: 0.025, n_unique: 1 },
    },
    data_quality: { columns_with_high_missing: ["zq_beta"], constant_columns: ["zq_gamma"] },
  },
};

describe("previewFromUpload", () => {
  it("reads rows, columns and the per-column numbers from the response", () => {
    const p = previewFromUpload(uploadFixture);
    expect(p.rows).toBe(1234);
    expect(p.columns).toBe(3);
    expect(p.columnsTable.map((c) => c.name)).toEqual(["zq_alpha", "zq_beta", "zq_gamma"]);
    const beta = p.columnsTable[1]!;
    expect(beta.type).toBe("categorical");
    expect(beta.missingPct).toBe(41.7); // missing_ratio 0.4167 shown as a percent
    expect(beta.distinct).toBe(7);
    // missing_pct, when the API sends it, wins over the ratio.
    expect(p.columnsTable[2]!.missingPct).toBe(2.5);
    expect(p.issues).toEqual(["High missing share: zq_beta", "Constant columns: zq_gamma"]);
  });

  it("falls back to the quick profiling_result and reports nothing it was not given", () => {
    const p = previewFromUpload({
      dataset_id: "x",
      profiling_result: {
        data_shape: { rows: 10, columns: 1 },
        column_statistics: { zq_only: { type: "numeric" } },
      },
    });
    expect(p.rows).toBe(10);
    expect(p.columns).toBe(1);
    expect(p.columnsTable).toEqual([
      { name: "zq_only", type: "numeric", missingPct: null, distinct: null },
    ]);
    expect(p.issues).toEqual([]);
    expect(previewFromUpload(null)).toEqual({
      rows: null,
      columns: null,
      columnsTable: [],
      issues: [],
    });
  });
});

describe("issuesFromProfile / profileStatusOf", () => {
  it("takes the profiler's concerns as written and drops anything that is not a sentence", () => {
    expect(
      issuesFromProfile({
        profiling_status: "full_profile",
        profiling_result: {
          data_quality_insights: { concerns: ["zq_beta is 41.7% missing", "", 3, null] },
        },
      }),
    ).toEqual(["zq_beta is 41.7% missing"]);
    expect(issuesFromProfile({ profiling_status: "full_profile", profiling_result: {} })).toEqual(
      [],
    );
    expect(issuesFromProfile(null)).toEqual([]);
  });

  it("maps the backend's status words", () => {
    expect(profileStatusOf("pending")).toBe("pending");
    expect(profileStatusOf("full_profile")).toBe("ready");
    expect(profileStatusOf("failed")).toBe("failed");
    expect(profileStatusOf("error")).toBe("failed");
    expect(profileStatusOf("skipped")).toBe("skipped");
    expect(profileStatusOf(null)).toBe("skipped");
  });
});

describe("rejectFile", () => {
  const limits = { plan: "free", maxFileBytes: 1024, maxFileMb: 0.001 };
  it("applies the accept list and the byte limit", () => {
    expect(rejectFile(new File(["a"], "ok.csv"), limits)).toBeNull();
    expect(rejectFile(new File(["a"], "ok.XLSX"), limits)).toBeNull();
    expect(rejectFile(new File(["a"], "notes.txt"), limits)).toMatch(/not a CSV, XLSX, XLS file/);
    expect(rejectFile(new File([new Uint8Array(1025)], "big.csv"), limits)).toMatch(
      /0\.001 MB per-file limit of the free plan/,
    );
    expect(INSTANT_ACCEPT).toBe(".csv,.xlsx,.xls");
  });
});

describe("the network calls", () => {
  const fetchSpy = vi.fn<typeof fetch>();
  beforeEach(() => {
    fetchSpy.mockReset();
    vi.stubGlobal("fetch", fetchSpy);
    localStorage.removeItem(DEV_TOKEN_KEY);
  });
  afterEach(() => vi.unstubAllGlobals());

  const lastCall = () => {
    const [url, init] = fetchSpy.mock.calls.at(-1)!;
    const headers = new Headers((init as RequestInit).headers);
    return { url: String(url), init: init as RequestInit, headers };
  };

  it("fetchInstantLimits unwraps { plan, limits } and keeps the API's MB figure", async () => {
    fetchSpy.mockResolvedValueOnce(
      ok({ plan: "professional", limits: { max_file_bytes: 5 * 1024 * 1024, max_file_mb: 5 } }),
    );
    expect(await fetchInstantLimits()).toEqual({
      plan: "professional",
      maxFileBytes: 5 * 1024 * 1024,
      maxFileMb: 5,
    });
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/limits");
    expect(lastCall().headers.get("authorization")).toBe("Bearer tok-instant");
  });

  it("fetchInstantLimits falls back to the upload modal's 100 MB when /api/limits fails", async () => {
    fetchSpy.mockResolvedValueOnce(new Response("nope", { status: 500 }));
    expect(await fetchInstantLimits()).toEqual({
      plan: "free",
      maxFileBytes: FALLBACK_MAX_FILE_BYTES,
      maxFileMb: 100,
    });
  });

  it("uploadForInstant posts the file to /api/upload and builds the preview from the answer", async () => {
    fetchSpy.mockResolvedValueOnce(ok(uploadFixture));
    const up = await uploadForInstant(new File(["a,b\n1,2\n"], "mine.csv", { type: "text/csv" }));
    const { url, init, headers } = lastCall();
    expect(url).toMatch(/\/api\/public\/backend-proxy$/);
    expect(init.method).toBe("POST");
    expect(headers.get("x-backend-path")).toBe("/api/upload");
    expect(headers.get("authorization")).toBe("Bearer tok-instant");
    const body = init.body as FormData;
    expect(body).toBeInstanceOf(FormData);
    expect((body.get("file") as File).name).toBe("mine.csv");
    expect(up.datasetId).toBe("ds-unit-1");
    expect(up.fileName).toBe("mine.csv");
    expect(up.profileStatus).toBe("pending");
    expect(up.preview.rows).toBe(1234);
    expect(up.preview.columnsTable[1]!.distinct).toBe(7);
  });

  it("uploadForInstant surfaces the server's reason", async () => {
    fetchSpy.mockResolvedValueOnce(ok({ detail: "File exceeds the plan limit" }, 413));
    await expect(uploadForInstant(new File(["x"], "x.csv"))).rejects.toThrow(
      /File exceeds the plan limit/,
    );
  });

  it("waitForInstantProfile polls the profile until it lands and returns that response", async () => {
    fetchSpy
      .mockResolvedValueOnce(ok({ dataset_id: "ds-unit-1", profiling_status: "pending" }))
      .mockResolvedValueOnce(ok({ dataset_id: "ds-unit-1", profiling_status: "pending" }))
      .mockResolvedValueOnce(
        ok({
          dataset_id: "ds-unit-1",
          profiling_status: "full_profile",
          profiling_result: { data_quality_insights: { concerns: ["zq_beta is 41.7% missing"] } },
        }),
      );
    const res = await waitForInstantProfile("ds-unit-1", { intervalMs: 1 });
    expect(fetchSpy).toHaveBeenCalledTimes(3);
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/datasets/ds-unit-1/profile");
    expect(issuesFromProfile(res)).toEqual(["zq_beta is 41.7% missing"]);
  });

  it("waitForInstantProfile gives up quietly on abort and on a dead endpoint", async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    expect(await waitForInstantProfile("ds-unit-1", { signal: ctrl.signal })).toBeNull();
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRejectedValue(new Error("down"));
    expect(await waitForInstantProfile("ds-unit-1", { intervalMs: 1 })).toBeNull();
    expect(fetchSpy).toHaveBeenCalledTimes(5);
  });

  it("loadSampleFile turns a public asset into a File named like the sample", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("a,b\n1,2\n", { status: 200, headers: { "Content-Type": "text/csv" } }),
    );
    const football = SAMPLE_DATASETS.find((s) => s.id === "football")!;
    const file = await loadSampleFile(football);
    expect(fetchSpy).toHaveBeenCalledWith("/samples/2021-2022%20Football%20Team%20Stats.csv");
    expect(file.name).toBe("2021-2022 Football Team Stats.csv");
    expect(file.size).toBe(8);
    expect(rejectFile(file, { plan: "free", maxFileBytes: 1024, maxFileMb: 0.001 })).toBeNull();
  });

  it("loadSampleFile reports a missing asset", async () => {
    fetchSpy.mockResolvedValueOnce(new Response("", { status: 404 }));
    await expect(loadSampleFile(SAMPLE_DATASETS[0]!)).rejects.toThrow(
      /Salaries is unavailable \(404\)/,
    );
  });

  it("askInstant is the PR-6 placeholder: coming soon, the question echoed, no request sent", async () => {
    const res = await askInstant({
      dataset_id: "ds-unit-1",
      question: "  Which columns move together?  ",
    });
    expect(res).toEqual({
      status: "coming_soon",
      dataset_id: "ds-unit-1",
      question: "Which columns move together?",
    });
    expect(fetchSpy).not.toHaveBeenCalled();
    await expect(askInstant({ dataset_id: "ds-unit-1", question: "   " })).rejects.toThrow(
      /Ask one question first/,
    );
    // a question-back answer rides on the same call, typed
    const again = await askInstant({
      dataset_id: "ds-unit-1",
      question: "Which groups matter?",
      clarify: { parameter: "dimension", option: "region" },
    });
    expect(again.status).toBe("coming_soon");
  });

  /** A run row as PR-6 stores it, with a report that mixes good and malformed entries. */
  const runFixture = {
    run_id: "run-unit-1",
    dataset_id: "ds-unit-1",
    recipe: { id: "trend", version: "1.0" },
    verdict: "qualified",
    question: "How did zq_alpha move?",
    export_url: "/api/instant/run-unit-1/export.html",
    notebook_url: "/api/instant/run-unit-1/notebook.ipynb",
    locked: true,
    report: {
      schema_version: "1.1",
      figures: [{ id: "f1", value: 22 }],
      data_checks: [
        { gate: "date_parse", status: "passed" },
        { gate: "period_gaps", status: "qualified", caveat_kind: "data_quality", figure: "f6" },
        { gate: "bogus", status: "maybe" },
        "not a check",
      ],
      caveats: [{ kind: "data_quality", text: "The period sequence has gaps." }, { kind: "x" }],
      completion: { state: "partial", missing: ["The period sequence has gaps.", 7] },
      method: { summary: "Sum per month in DuckDB.", steps: ["Recipe trend, version 1.0."] },
      provenance: { recipe: { id: "trend", version: "1.0" }, llm_used: false },
    },
  };

  it("fetchInstantRun GETs the run through the proxy and reads the report's typed fields", async () => {
    fetchSpy.mockResolvedValueOnce(ok(runFixture));
    const run = await fetchInstantRun("run-unit-1");
    expect(lastCall().headers.get("x-backend-path")).toBe(`${INSTANT_RUN_PATH}/run-unit-1`);
    expect(lastCall().init.method).toBe("GET");
    expect(lastCall().headers.get("authorization")).toBe("Bearer tok-instant");
    expect(run.run_id).toBe("run-unit-1");
    expect(run.verdict).toBe("qualified");
    expect(run.recipe).toEqual({ id: "trend", version: "1.0" });
    expect(run.question).toBe("How did zq_alpha move?");
    expect(run.locked).toBe(true);
    expect(run.export_url).toBe("/api/instant/run-unit-1/export.html");
    // malformed entries are dropped; the schema's shapes are kept
    expect(run.report!.data_checks).toEqual([
      { gate: "date_parse", status: "passed" },
      { gate: "period_gaps", status: "qualified", caveat_kind: "data_quality", figure: "f6" },
    ]);
    expect(run.report!.caveats).toEqual([
      { kind: "data_quality", text: "The period sequence has gaps." },
    ]);
    expect(run.report!.completion).toEqual({
      state: "partial",
      missing: ["The period sequence has gaps."],
      open_questions: [],
    });
    expect(run.report!.method!.summary).toBe("Sum per month in DuckDB.");
    // the rest rides along untyped: the page never formats a figure
    expect(run.report!.figures).toEqual([{ id: "f1", value: 22 }]);
  });

  it("fetchInstantRun refuses a run it cannot read, and surfaces the server's status", async () => {
    fetchSpy.mockResolvedValueOnce(ok({ ...runFixture, verdict: "maybe" }));
    await expect(fetchInstantRun("run-unit-1")).rejects.toThrow(/could not be read/);
    fetchSpy.mockResolvedValueOnce(ok({ run_id: "x" }));
    await expect(fetchInstantRun("x")).rejects.toThrow(/could not be read/);
    fetchSpy.mockResolvedValueOnce(new Response("", { status: 404, statusText: "Not Found" }));
    await expect(fetchInstantRun("gone")).rejects.toThrow(/Run fetch failed: Not Found/);
  });

  it("fetchInstantExport returns the HTML text: a backend path via the proxy, an absolute URL as is", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("<!doctype html><p>x</p>", {
        status: 200,
        headers: { "Content-Type": "text/html" },
      }),
    );
    expect(await fetchInstantExport("/api/instant/run-unit-1/export.html")).toBe(
      "<!doctype html><p>x</p>",
    );
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/instant/run-unit-1/export.html");
    expect(lastCall().headers.get("authorization")).toBe("Bearer tok-instant");

    fetchSpy.mockResolvedValueOnce(new Response("<p>signed</p>", { status: 200 }));
    expect(await fetchInstantExport("https://store.example/exports/r.html?sig=1")).toBe(
      "<p>signed</p>",
    );
    expect(lastCall().url).toBe("https://store.example/exports/r.html?sig=1");
    expect(lastCall().headers.get("authorization")).toBeNull();

    fetchSpy.mockResolvedValueOnce(new Response("", { status: 403, statusText: "Forbidden" }));
    await expect(fetchInstantExport("/api/instant/run-unit-1/export.html")).rejects.toThrow(
      /Export fetch failed: Forbidden/,
    );
  });
});

describe("SAMPLE_DATASETS", () => {
  it("names the three app/sample_data files served from /samples", () => {
    expect(SAMPLE_DATASETS.map((s) => s.file)).toEqual([
      "salaries.csv",
      "2021-2022 Football Team Stats.csv",
      "Global_YouTube_Statistics.csv",
    ]);
    for (const s of SAMPLE_DATASETS) {
      expect(s.path).toBe(`/samples/${encodeURIComponent(s.file)}`);
    }
  });
});
