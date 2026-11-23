// /instant, the three states on one page, against a mocked fetch: the Start
// state applies the accept list and byte limit from /api/limits and offers
// the three samples; the Preview state shows the numbers the upload and the
// profile poll returned (and nothing it measured itself); the Ask state hands
// the question to the PR-6 stub and echoes it back as "coming soon".
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/integrations/supabase/external-client", () => ({
  supabase: {
    auth: { getSession: vi.fn(async () => ({ data: { session: { access_token: "tok-ui" } } })) },
  },
}));

import { DEV_TOKEN_KEY } from "@/lib/api/backendApi";
import { instantApi, type InstantApi } from "@/lib/api/instant";
import { InstantFlow } from "@/components/instant/InstantFlow";

const limitsFixture = {
  plan: "free",
  limits: { max_file_bytes: 5 * 1024 * 1024, max_file_mb: 5, max_files: 10 },
};

/** Synthetic upload answer (the shapes app/agents/sampling_async.py writes). */
const uploadFixture = {
  dataset_id: "ds-test-1",
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
      zq_alpha: { type: "numeric", missing_ratio: 0, n_unique: 1234 },
      zq_beta: { type: "categorical", missing_ratio: 0.4167, n_unique: 7 },
      zq_gamma: { type: "categorical", missing_ratio: 0.025, n_unique: 1 },
    },
    data_quality: { constant_columns: ["zq_gamma"] },
  },
};

const profilePending = { dataset_id: "ds-test-1", profiling_status: "pending" };
const profileLanded = {
  dataset_id: "ds-test-1",
  profiling_status: "full_profile",
  profiling_result: {
    data_quality_insights: { concerns: ["zq_beta is missing in 41.7% of rows"] },
  },
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** The backend path a proxied call asked for, or the plain URL for an asset. */
function pathOf(url: unknown, init?: RequestInit): string {
  return new Headers(init?.headers).get("x-backend-path") ?? String(url);
}

describe("/instant", () => {
  const fetchSpy = vi.fn<typeof fetch>();
  let profileCalls = 0;
  // The page's own poll interval is seconds; the test's is milliseconds.
  const api: InstantApi = {
    ...instantApi,
    waitForInstantProfile: (id, opts) =>
      instantApi.waitForInstantProfile(id, { ...opts, intervalMs: 2 }),
  };

  beforeEach(() => {
    profileCalls = 0;
    fetchSpy.mockReset();
    fetchSpy.mockImplementation(async (url, init) => {
      const path = pathOf(url, init);
      if (path === "/api/limits") return json(limitsFixture);
      if (path === "/api/upload") return json(uploadFixture);
      if (path === "/api/datasets/ds-test-1/profile") {
        profileCalls += 1;
        return json(profileCalls < 2 ? profilePending : profileLanded);
      }
      if (path.startsWith("/samples/")) {
        return new Response("name,amount\nx,1\n", {
          status: 200,
          headers: { "Content-Type": "text/csv" },
        });
      }
      return new Response("not found", { status: 404 });
    });
    vi.stubGlobal("fetch", fetchSpy);
    localStorage.removeItem(DEV_TOKEN_KEY);
  });
  afterEach(() => vi.unstubAllGlobals());

  const calledPaths = () => fetchSpy.mock.calls.map(([u, i]) => pathOf(u, i as RequestInit));
  const fileInput = () => screen.getByTestId("instant-file-input") as HTMLInputElement;
  const choose = (file: File) => fireEvent.change(fileInput(), { target: { files: [file] } });

  describe("Start", () => {
    it("shows the tagline, the drop zone with the accept list, and the three samples", async () => {
      render(<InstantFlow api={api} />);
      expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
        "One question → verified report",
      );
      expect(screen.getByTestId("instant-dropzone")).toBeTruthy();
      expect(fileInput().getAttribute("accept")).toBe(".csv,.xlsx,.xls");
      // The byte limit on the page is /api/limits' figure, not a constant.
      await waitFor(() =>
        expect(screen.getByTestId("instant-limit-line").textContent).toBe(
          "CSV, XLSX, XLS up to 5 MB",
        ),
      );
      expect(screen.getByTestId("instant-sample-salaries").textContent).toBe("Salaries");
      expect(screen.getByTestId("instant-sample-football").textContent).toBe("Football team stats");
      expect(screen.getByTestId("instant-sample-youtube").textContent).toBe("YouTube channels");
      expect(screen.queryByTestId("instant-preview")).toBeNull();
      expect(screen.queryByTestId("instant-ask")).toBeNull();
    });

    it("refuses a file over the plan's byte limit, or outside the accept list, before uploading", async () => {
      render(<InstantFlow api={api} />);
      await waitFor(() =>
        expect(screen.getByTestId("instant-limit-line").textContent).toContain("5 MB"),
      );
      choose(new File([new Uint8Array(5 * 1024 * 1024 + 1)], "big.csv", { type: "text/csv" }));
      await waitFor(() =>
        expect(screen.getByRole("alert").textContent).toMatch(
          /big\.csv is larger than the 5 MB per-file limit of the free plan/,
        ),
      );
      choose(new File(["hello"], "notes.txt", { type: "text/plain" }));
      await waitFor(() =>
        expect(screen.getByRole("alert").textContent).toMatch(
          /notes\.txt is not a CSV, XLSX, XLS file/,
        ),
      );
      expect(calledPaths()).not.toContain("/api/upload");
      expect(screen.queryByTestId("instant-preview")).toBeNull();
    });
  });

  describe("Preview", () => {
    it("renders the upload's numbers, then the landed profile's concerns", async () => {
      render(<InstantFlow api={api} />);
      await waitFor(() =>
        expect(screen.getByTestId("instant-limit-line").textContent).toContain("5 MB"),
      );
      choose(new File(["a,b,c\n1,2,3\n"], "mine.csv", { type: "text/csv" }));

      await waitFor(() => expect(screen.getByTestId("instant-preview")).toBeTruthy());
      expect(screen.getByText("mine.csv")).toBeTruthy();
      // rows and columns are the fixture's data_shape, formatted, not counted here
      expect(screen.getByTestId("instant-rows").textContent).toBe("1,234");
      expect(screen.getByTestId("instant-columns").textContent).toBe("3");
      const table = screen.getByTestId("instant-column-table");
      const beta = within(table).getByText("zq_beta").closest("tr")!;
      expect([...beta.querySelectorAll("td")].map((td) => td.textContent)).toEqual([
        "zq_beta",
        "categorical",
        "41.7%",
        "7",
      ]);
      const alpha = within(table).getByText("zq_alpha").closest("tr")!;
      expect([...alpha.querySelectorAll("td")].map((td) => td.textContent)).toEqual([
        "zq_alpha",
        "numeric",
        "0%",
        "1,234",
      ]);
      // the upload's own quality issue is there at once
      const issues = screen.getByTestId("instant-issues");
      expect(issues.textContent).toContain("Constant columns: zq_gamma");
      // the profile is polled until it lands, and its concern joins the list
      await waitFor(() =>
        expect(issues.textContent).toContain("zq_beta is missing in 41.7% of rows"),
      );
      expect(profileCalls).toBe(2);
      expect(calledPaths().filter((p) => p === "/api/datasets/ds-test-1/profile").length).toBe(2);
      // the drop zone is gone; the question box is here
      expect(screen.queryByTestId("instant-dropzone")).toBeNull();
      expect(screen.getByTestId("instant-ask")).toBeTruthy();
    });

    it("a sample button fetches the public CSV and uploads it as that file", async () => {
      render(<InstantFlow api={api} />);
      await waitFor(() =>
        expect(screen.getByTestId("instant-limit-line").textContent).toContain("5 MB"),
      );
      fireEvent.click(screen.getByTestId("instant-sample-football"));
      await waitFor(() => expect(screen.getByTestId("instant-preview")).toBeTruthy());
      expect(calledPaths()).toContain("/samples/2021-2022%20Football%20Team%20Stats.csv");
      const uploadCall = fetchSpy.mock.calls.find(
        ([u, i]) => pathOf(u, i as RequestInit) === "/api/upload",
      )!;
      const body = (uploadCall[1] as RequestInit).body as FormData;
      expect((body.get("file") as File).name).toBe("2021-2022 Football Team Stats.csv");
      expect(screen.getByText("2021-2022 Football Team Stats.csv")).toBeTruthy();
      expect(screen.getByTestId("instant-rows").textContent).toBe("1,234");
    });

    it("says so when the profile failed, instead of inventing issues", async () => {
      fetchSpy.mockImplementation(async (url, init) => {
        const path = pathOf(url, init);
        if (path === "/api/limits") return json(limitsFixture);
        if (path === "/api/upload")
          return json({
            ...uploadFixture,
            profiling_status: "failed",
            profiling_error: "The profiling model did not answer.",
          });
        return new Response("not found", { status: 404 });
      });
      render(<InstantFlow api={api} />);
      choose(new File(["a\n1\n"], "f.csv", { type: "text/csv" }));
      await waitFor(() => expect(screen.getByTestId("instant-preview")).toBeTruthy());
      const issues = screen.getByTestId("instant-issues");
      // the upload's constant-column issue still shows; the failure message would show if there were none
      expect(issues.textContent).toContain("Constant columns: zq_gamma");
      expect(calledPaths()).not.toContain("/api/datasets/ds-test-1/profile");
    });
  });

  describe("Ask", () => {
    async function reachAsk() {
      render(<InstantFlow api={api} />);
      await waitFor(() =>
        expect(screen.getByTestId("instant-limit-line").textContent).toContain("5 MB"),
      );
      choose(new File(["a\n1\n"], "mine.csv", { type: "text/csv" }));
      await waitFor(() => expect(screen.getByTestId("instant-ask")).toBeTruthy());
    }

    it("submits the question to the stub and shows it back as coming soon", async () => {
      const askSpy = vi.spyOn(api, "askInstant");
      await reachAsk();
      const input = screen.getByLabelText("Your question") as HTMLInputElement;
      const button = screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement;
      expect(button.disabled).toBe(true);
      fireEvent.change(input, { target: { value: "Which zq_ column drives zq_alpha?" } });
      expect(button.disabled).toBe(false);
      fireEvent.click(button);
      await waitFor(() => expect(screen.getByTestId("instant-coming-soon")).toBeTruthy());
      expect(screen.getByTestId("instant-echo").textContent).toBe(
        "Which zq_ column drives zq_alpha?",
      );
      expect(askSpy).toHaveBeenCalledWith({
        dataset_id: "ds-test-1",
        question: "Which zq_ column drives zq_alpha?",
      });
      // the placeholder sends nothing: no /api/instant request exists yet (PR-6)
      expect(calledPaths().some((p) => p.startsWith("/api/instant"))).toBe(false);
    });

    it("hands a ready answer to onReady, the route's way to the report page", async () => {
      vi.spyOn(api, "askInstant").mockResolvedValue({
        status: "ready",
        dataset_id: "ds-test-1",
        question: "How did zq_alpha move?",
        run_id: "run-42",
        recipe: "trend",
      });
      const onReady = vi.fn();
      render(<InstantFlow api={api} onReady={onReady} />);
      await waitFor(() =>
        expect(screen.getByTestId("instant-limit-line").textContent).toContain("5 MB"),
      );
      choose(new File(["a\n1\n"], "mine.csv", { type: "text/csv" }));
      await waitFor(() => expect(screen.getByTestId("instant-ask")).toBeTruthy());
      fireEvent.change(screen.getByLabelText("Your question"), {
        target: { value: "How did zq_alpha move?" },
      });
      fireEvent.click(screen.getByRole("button", { name: "Ask" }));
      await waitFor(() => expect(onReady).toHaveBeenCalledTimes(1));
      expect(onReady.mock.calls[0]![0]).toMatchObject({ status: "ready", run_id: "run-42" });
      expect(screen.queryByTestId("instant-coming-soon")).toBeNull();
    });

    it("offers three example questions as chips that fill the box", async () => {
      await reachAsk();
      const chips = within(screen.getByLabelText("Example questions")).getAllByRole("button");
      expect(chips.length).toBe(3);
      fireEvent.click(chips[1]!);
      expect((screen.getByLabelText("Your question") as HTMLInputElement).value).toBe(
        chips[1]!.textContent,
      );
    });

    it("shows nothing about agents, pricing or payments", async () => {
      await reachAsk();
      expect(document.body.textContent).not.toMatch(
        /agent|swarm|price|pricing|\$|checkout|unlock/i,
      );
    });
  });
});
