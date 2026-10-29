// @vitest-environment happy-dom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnalysisResultsView } from "../../src/components/dashboard/data-view/AnalysisResultsView";
import { DataGrid } from "../../src/components/dashboard/data-view/DataGrid";
import { DatasetChipName } from "../../src/components/dashboard/data-view/DatasetChipName";
import type { TableRun } from "../../src/lib/analysis-results";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | null = null;
let host: HTMLDivElement | null = null;
const rows = [{ Country: "India", category: "Music", avg_video_views: 120 }];
const runs: TableRun[] = [1, 2].map((i) => ({ id: String(i), sectionId: "s", title: "Result", rows, prompt: `For each Country and category, calculate the average video views and compare all categories ${i}.` }));

function mount(content: React.ReactNode) {
  host = document.createElement("div");
  document.body.appendChild(host);
  const mountedRoot = createRoot(host);
  root = mountedRoot;
  act(() => mountedRoot.render(content));
  return host;
}
function results(count: number, index = 0, onIndexChange = vi.fn()) {
  return mount(<AnalysisResultsView runs={runs.slice(0, count)} index={index} tableIndex={0} onIndexChange={onIndexChange} onTableChange={() => {}} onOpenInChat={() => {}} />);
}
afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  vi.restoreAllMocks();
});

describe("Analysis results presentation", () => {
  it("uses a themed card and hides navigation for one analysis", () => {
    const el = results(1);
    const header = el.querySelector(".analysis-result-header");
    expect(header?.classList.contains("bg-card")).toBe(true);
    expect(header?.className).not.toMatch(/bg-white|bg-\[#/);
    expect(el.textContent).toContain("Analysis 1 of 1");
    expect(el.querySelector('[aria-label="Previous analysis"]')).toBeNull();
    expect(el.querySelector('[aria-label="Next analysis"]')).toBeNull();
    expect(el.querySelector('[role="combobox"]')).toBeNull();
  });
  it("shows expansion only for measured overflow and separates the chat action", () => {
    vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockReturnValue(60);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(40);
    const el = results(1);
    const more = Array.from(el.querySelectorAll("button")).find((button) => button.textContent === "Show more");
    const chat = Array.from(el.querySelectorAll("button")).find((button) => button.textContent?.includes("Open in chat"));
    expect(more).toBeDefined();
    expect(chat).toBeDefined();
    expect(more).not.toBe(chat);
    expect(more?.parentElement).toBe(chat?.parentElement);
    expect(more?.parentElement?.classList.contains("gap-4")).toBe(true);
    act(() => more?.click());
    expect(more?.textContent).toBe("Show less");
  });
  it("does not offer expansion for a prompt fitting two lines", () => {
    const el = results(1);
    expect(el.textContent).not.toContain("Show more");
  });
  it.each([0, 1])("shows navigation with two runs and disables the boundary at %i", (index) => {
    const changed = vi.fn();
    const el = results(2, index, changed);
    const previous = el.querySelector<HTMLButtonElement>('[aria-label="Previous analysis"]');
    const next = el.querySelector<HTMLButtonElement>('[aria-label="Next analysis"]');
    expect(previous?.disabled).toBe(index === 0);
    expect(next?.disabled).toBe(index === 1);
    expect(el.querySelector('[role="combobox"]')?.textContent).toContain(`Analysis ${index + 1}`);
    act(() => (index ? previous : next)?.click());
    expect(changed).toHaveBeenCalledWith(index ? 0 : 1);
  });
  it("keeps each filter in its own header next to its sortable name", () => {
    const el = mount(<DataGrid rows={rows} columns={Object.keys(rows[0])} />);
    for (const column of Object.keys(rows[0])) {
      const filter = el.querySelector(`[aria-label="Column options for ${column}"]`);
      const header = filter?.closest("th");
      expect(header?.textContent).toContain(column);
      expect(filter?.parentElement?.classList.contains("gap-1")).toBe(true);
      expect(filter?.parentElement?.classList.contains("flex-row-reverse")).toBe(column === "avg_video_views");
      const sort = filter?.parentElement?.querySelector<HTMLButtonElement>("button");
      act(() => sort?.click());
      expect(header?.getAttribute("aria-sort")).toBe("ascending");
    }
  });
  it("shows the upload-review readable chip name with the original filename tooltip", () => {
    const el = mount(<DatasetChipName name="Global_YouTube_Statistics__1.csv" />);
    expect(el.textContent).toBe("Global YouTube Statistics");
    expect(el.querySelector("span")?.title).toBe("Global_YouTube_Statistics__1.csv");
  });
});