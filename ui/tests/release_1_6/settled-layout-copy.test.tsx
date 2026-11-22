// @vitest-environment happy-dom
// 1.6 release suite: rows settled from the shipped code and its documented
// rules (Bugs-1.6 BG25, BG27, BG46; UI Bugs UI-B19). Each test drives the real
// product function or component. See docs/test-reports/1.6-traceability.csv.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@/components/dashboard/InviteCollaboratorsTrigger", () => ({ InviteCollaboratorsTrigger: () => null }));
vi.mock("@/components/dashboard/UploadDropzone", () => ({ UploadDropzone: () => null }));
vi.mock("@/components/dashboard/DataSourceGrid", () => ({ DataSourceGrid: () => null }));

import { WORKSPACE_CHROME, DATA_AREA_MIN, defaultChatWidth, maxChatWidth, clampChatWidth } from "@/lib/chat-width";
import { measureLabel, binLabel } from "@/lib/chart-display";
import { CLEANING_TOOLTIP, CLEANING_DISCLOSURE } from "@/lib/upload-review";
import { CleaningRules } from "@/components/dashboard/upload/UploadReviewPanels";
import { ProjectDetailsView } from "@/components/dashboard/ProjectDetailsView";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | null = null;
let host: HTMLElement | null = null;
function mount(node: JSX.Element) {
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

/** WCAG 2.x relative luminance / contrast ratio for #rrggbb colours. */
function contrast(a: string, b: string): number {
  const lum = (hex: string) => {
    const [r, g, bl] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
  };
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe("BG46 workspace width", () => {
  it("BG46: at common desktop widths the default layout gives the data area the majority of the window", () => {
    for (const viewport of [1280, 1366, 1440, 1920]) {
      const data = viewport - WORKSPACE_CHROME - defaultChatWidth(viewport);
      expect(data / viewport, `viewport ${viewport}`).toBeGreaterThan(0.5);
    }
  });

  it("BG46: dragging the chat wider never squeezes the data area below its 480px floor", () => {
    for (const viewport of [1024, 1280, 1366, 1920]) {
      const widest = clampChatWidth(10_000, viewport);
      expect(widest).toBe(maxChatWidth(viewport));
      expect(viewport - WORKSPACE_CHROME - widest, `viewport ${viewport}`).toBeGreaterThanOrEqual(DATA_AREA_MIN);
    }
  });
});

describe("BG27 axis number formatting", () => {
  it("BG27: large axis values are abbreviated (75K, 487K, 1.1M), not printed raw", () => {
    expect(measureLabel(75000)).toBe("75K");
    expect(measureLabel(487000)).toBe("487K");
    expect(measureLabel(1100000)).toBe("1.1M");
    expect(measureLabel(2500000000)).toBe("2.5B");
    expect(measureLabel(950)).toBe("950");
    expect(binLabel("75000-100000")).toBe("75K-100K");
  });
});

describe("BG25 Clean & Start disclosure", () => {
  it("BG25: 'View cleaning rules' discloses exactly what Clean & Start does (no cleaning options are sent)", () => {
    const el = mount(<CleaningRules />);
    const toggle = [...el.querySelectorAll("button")].find((b) => /view cleaning rules/i.test(b.textContent ?? ""));
    expect(toggle).toBeDefined();
    expect(el.textContent).not.toContain(CLEANING_DISCLOSURE);
    act(() => { toggle!.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
    expect(el.textContent).toContain("No cleaning options are sent by this action.");
    for (const rule of ["Duplicate row handling", "Empty cell handling", "Original file retention"]) {
      expect(el.textContent).toContain(rule);
    }
    expect(CLEANING_TOOLTIP).toMatch(/no cleaning options are sent/i);
  });
});

describe("UI-B19 dashboard New Analysis button", () => {
  it("UI-B19: the dashboard 'New Analysis' CTA uses explicit brand colours that stay legible in dark mode (AA contrast)", () => {
    const el = mount(<ProjectDetailsView projectId={null} projectName="Demo project" rows={[]} onRenameAnalysis={() => {}} onDeleteAnalysis={() => {}} />);
    const cta = [...el.querySelectorAll("button")].find((b) => b.textContent?.trim() === "New Analysis");
    expect(cta, "New Analysis button").toBeDefined();
    const classes = cta!.className.split(/\s+/);
    expect(classes).toContain("bg-[#1565ef]");
    expect(classes).toContain("text-white");
    // No theme token that would flip with the dark theme.
    expect(classes.some((c) => /^(bg|text)-(primary|secondary|background|foreground)$/.test(c))).toBe(false);
    expect(contrast("#ffffff", "#1565ef")).toBeGreaterThanOrEqual(4.5);
  });
});
