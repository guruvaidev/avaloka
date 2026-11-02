// @vitest-environment happy-dom
// 1.6 release suite: Bugs-1.6 welcome screen and navigation rows
// (BG13, BG14, BG15, BG16, BG19). See docs/test-reports/1.6-traceability.csv.
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Folder, Database01, Settings01 } from "@untitledui/icons";

// Router, query and server-function hooks: the components under test only need
// them to exist. Every stub returns inert data; nothing reaches a network.
vi.mock("@tanstack/react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@tanstack/react-router")>()),
  createFileRoute: () => (options: unknown) => ({ options }),
  useNavigate: () => vi.fn(),
  useLocation: () => ({ state: {} }),
  useRouterState: () => ({ location: { pathname: "/" } }),
}));
vi.mock("@tanstack/react-query", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@tanstack/react-query")>()),
  // The welcome page reads the signed-in user's first name exactly as the
  // server function returns it: lower-case here, as stored in the profile.
  useQuery: ({ queryKey }: { queryKey: unknown[] }) => ({ data: queryKey[0] === "my-first-name" ? { firstName: "jane" } : undefined }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));
vi.mock("@tanstack/react-start", () => ({ useServerFn: (fn: unknown) => fn }));
vi.mock("@/lib/me.functions", () => ({ getMyFirstName: vi.fn() }));
vi.mock("@/components/dashboard/DashboardShell", () => ({ DashboardShell: ({ children }: { children: ReactNode }) => <>{children}</> }));
vi.mock("@/components/dashboard/UploadDropzone", () => ({ UploadDropzone: () => null }));
vi.mock("@/components/dashboard/SecurityFooter", () => ({ SecurityFooter: () => null }));
vi.mock("@/components/dashboard/UploadModal", () => ({ UploadModal: () => null }));
vi.mock("@/components/dashboard/ScheduleAnalysisModal", () => ({ ScheduleAnalysisModal: () => null }));
vi.mock("@/components/dashboard/DeleteAnalysisModal", () => ({ DeleteAnalysisModal: () => null, DeletedToastModal: () => null }));
// Hook results must be referentially stable, as real react-query data is:
// a fresh [] per render re-fires the sidebar's memo/effect chain forever.
const stable = vi.hoisted(() => ({
  analyses: { data: [] as unknown[], refetch: () => {}, isFetching: false },
  mutations: { rename: { mutate: () => {} }, softDelete: { mutate: () => {} }, setFlag: { mutate: () => {} } },
  datasets: { byAnalysis: {}, isLoading: false },
}));
vi.mock("@/lib/analyses", () => ({
  analysesKey: () => ["analyses"],
  useStandaloneAnalyses: () => stable.analyses,
  useAnalysisMutations: () => stable.mutations,
}));
vi.mock("@/lib/analysis-datasets", () => ({
  useAnalysesDatasets: () => stable.datasets,
  analysisDatasetLabel: () => "",
  analysisGroupName: (name: string) => name,
}));
// The connection modals need a router/query-client context; the tile grid does not.
vi.mock("@/components/database/CloudStorageBrowserModal", () => ({ CloudStorageBrowserModal: () => null }));
vi.mock("@/components/database/CloudStorageConnectionsListModal", () => ({ CloudStorageConnectionsListModal: () => null }));
vi.mock("@/components/database/DatabaseConnectionsListModal", () => ({ DatabaseConnectionsListModal: () => null }));
// IconRail module-level imports that would otherwise open a Supabase client.
vi.mock("@/integrations/supabase/client", () => ({
  supabase: {
    auth: {
      onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
      getSession: async () => ({ data: { session: null } }),
      getUser: async () => ({ data: { user: null } }),
    },
  },
}));
vi.mock("@/lib/auth-user", () => ({ getCachedAuthUser: () => null }));
vi.mock("@/lib/use-user-mgmt-perms", () => ({ useProjectsPerms: () => ({}), useDashboardPerms: () => ({}) }));
vi.mock("@/lib/use-active-plan", () => ({ useActivePlan: () => ({}) }));
vi.mock("@/components/dashboard/NotificationsPopover", () => ({ NotificationsPopover: () => null }));

import { Route as ProAnalysisIndexRoute } from "../../src/routes/proanalysis.index";
import { AnalysisSidebar } from "../../src/components/dashboard/AnalysisSidebar";
import { DataSourceGrid } from "../../src/components/dashboard/DataSourceGrid";
import { RailGroup } from "../../src/components/dashboard/IconRail";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let host: HTMLElement | null = null;

function mount(node: ReactNode) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root!.render(node));
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const expandGrid = (el: HTMLElement) => {
  const viewMore = [...el.querySelectorAll("button")].find((b) => b.textContent?.trim() === "View More");
  expect(viewMore, "View More control").toBeDefined();
  act(() => {
    viewMore!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
};

describe("Bugs-1.6 welcome screen", () => {
  // BG13 is an open defect: proanalysis.index.tsx renders `Welcome ${firstName}`
  // with the name exactly as stored ("Welcome jane"). The /analysis empty state
  // (routes/analysis.tsx, "Welcome, ${userName}") has the same raw-name problem;
  // it lives inside the route file, so only the /proanalysis heading is pinned here.
  // it.fails: remove when the heading reads "Welcome, Jane".
  it.fails("BG13 [open defect, strict]: welcome heading capitalises the first name", () => {
    const Page = (ProAnalysisIndexRoute as unknown as { options: { component: () => ReactNode } }).options.component;
    const el = mount(<Page />);
    const heading = el.querySelector("h1");
    expect(heading).not.toBeNull();
    expect(heading!.textContent?.trim()).toBe("Welcome, Jane");
  });

  // Guards the strict xfail above: the page must really render the greeting,
  // so BG13 can only fail on the capitalisation, not on a broken harness.
  it("BG13: welcome heading renders the signed-in user's first name", () => {
    const Page = (ProAnalysisIndexRoute as unknown as { options: { component: () => ReactNode } }).options.component;
    const el = mount(<Page />);
    const heading = el.querySelector("h1");
    expect(heading).not.toBeNull();
    expect(heading!.textContent).toMatch(/^Welcome,? jane$/i);
  });

  it("BG14: sidebar actions are Title Case (New Analysis, Schedule Analysis)", () => {
    const el = mount(<AnalysisSidebar />);
    const labels = [...el.querySelectorAll("button")].map((b) => b.textContent?.trim() ?? "");
    for (const action of ["New Analysis", "Schedule Analysis"]) expect(labels).toContain(action);
    expect(labels).not.toContain("Schedule analysis");
    // Every multi-word action that starts with a verb noun pair is Title Case.
    for (const label of labels.filter((text) => /^(New|Schedule) /.test(text))) {
      for (const word of label.split(/\s+/)) expect(word[0], `"${label}"`).toBe(word[0].toUpperCase());
    }
  });

  it("BG15: every connector logo shares one height and is centred in an equal-height slot", () => {
    const el = mount(<DataSourceGrid />);
    expandGrid(el);
    const logos = [...el.querySelectorAll("button img")] as HTMLImageElement[];
    expect(logos.length).toBe(8);
    for (const img of logos) {
      const classes = img.className.split(/\s+/);
      expect(classes, img.alt).toEqual(expect.arrayContaining(["h-7", "w-auto", "object-contain", "object-center"]));
      // The slot around the logo has a fixed height and centres on both axes.
      const slot = img.parentElement!.parentElement!;
      expect(slot.className.split(/\s+/), img.alt).toEqual(expect.arrayContaining(["h-10", "items-center", "justify-center"]));
    }
  });

  it("BG16: 'View More' is a link-style control below the grid, not a data-source tile", () => {
    const el = mount(<DataSourceGrid />);
    const viewMore = [...el.querySelectorAll("button")].find((b) => b.textContent?.trim() === "View More")!;
    expect(viewMore).toBeDefined();
    const tile = el.querySelector('img[alt="MySQL"]')!.closest("button")!;
    // A tile carries a logo and the fixed tile box; the control carries neither.
    expect(viewMore.querySelector("img")).toBeNull();
    for (const tileClass of ["h-[96px]", "bg-card", "border-border"]) {
      expect(tile.className).toContain(tileClass);
      expect(viewMore.className).not.toContain(tileClass);
    }
    // It spans the whole grid row, so it sits below the tiles rather than among them.
    expect(viewMore.className.split(/\s+/)).toContain("col-span-full");
  });

  it("BG19: only the active rail item is marked current and highlighted", () => {
    const items = [
      { id: "projects", icon: Folder, label: "Projects" },
      { id: "data", icon: Database01, label: "Data" },
      { id: "settings", icon: Settings01, label: "Settings" },
    ];
    const el = mount(<RailGroup items={items} active="data" />);
    for (const item of items) {
      const wrapper = el.querySelector(`[data-rail-item="${item.id}"]`) as HTMLElement;
      expect(wrapper, item.id).not.toBeNull();
      const button = wrapper.querySelector("button")!;
      const accentBar = wrapper.querySelector("span.bg-brand-solid");
      if (item.id === "data") {
        expect(button.getAttribute("aria-current")).toBe("page");
        expect(button.className).toContain("bg-brand-primary_alt");
        expect(accentBar).not.toBeNull();
      } else {
        expect(button.getAttribute("aria-current")).toBeNull();
        expect(button.className).not.toContain("bg-brand-primary_alt");
        expect(accentBar).toBeNull();
      }
    }
  });
});
