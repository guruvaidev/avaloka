// 1.6 release suite: Bugs-1.6 BG17 (secondary text contrast).
// Reads the colour tokens the UI actually ships (src/styles.css) and computes
// the WCAG 2.x contrast ratio, rather than trusting a class name.
// See docs/test-reports/1.6-traceability.csv.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// Vitest runs from ui/ (vitest.config.ts); accept the repo root as well.
const stylesPath = [join(process.cwd(), "src/styles.css"), join(process.cwd(), "ui/src/styles.css")].find((p) => existsSync(p));
if (!stylesPath) throw new Error("ui/src/styles.css not found from " + process.cwd());
const css = readFileSync(stylesPath, "utf8");

/** Custom properties declared directly in the first `<selector> {` block. */
function block(selector: string): Record<string, string> {
  const start = css.search(new RegExp(`(^|\\n)${selector.replace(".", "\\.")}\\s*\\{`));
  if (start < 0) throw new Error(`no ${selector} block in styles.css`);
  const open = css.indexOf("{", start);
  const close = css.indexOf("\n}", open);
  const body = css.slice(open + 1, close).replace(/\/\*[\s\S]*?\*\//g, "");
  const vars: Record<string, string> = {};
  for (const m of body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) vars[m[1]] = m[2].trim();
  return vars;
}

function resolve(vars: Record<string, string>, value: string, depth = 0): string {
  const ref = value.match(/^var\((--[\w-]+)\)$/);
  if (!ref) return value;
  if (depth > 5 || !(ref[1] in vars)) throw new Error(`unresolved ${value}`);
  return resolve(vars, vars[ref[1]], depth + 1);
}

type Linear = [number, number, number];

function hexToLinear(hex: string): Linear {
  const v = hex.replace("#", "");
  const channels = [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16) / 255);
  return channels.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)) as Linear;
}

/** OKLCH -> linear sRGB (Björn Ottosson's OKLab matrices), clamped to gamut. */
function oklchToLinear(l: number, c: number, h: number): Linear {
  const a = c * Math.cos((h * Math.PI) / 180);
  const b = c * Math.sin((h * Math.PI) / 180);
  const l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s_ = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const rgb = [
    4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_,
  ];
  return rgb.map((x) => Math.min(1, Math.max(0, x))) as Linear;
}

function toLinear(color: string): Linear {
  if (/^#[0-9a-f]{6}$/i.test(color)) return hexToLinear(color);
  const m = color.match(/^oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\)$/);
  if (m) return oklchToLinear(Number(m[1]), Number(m[2]), Number(m[3]));
  throw new Error(`unsupported colour ${color}`);
}

function contrast(fg: string, bg: string): number {
  const lum = ([r, g, b]: Linear) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const [hi, lo] = [lum(toLinear(fg)), lum(toLinear(bg))].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe("Bugs-1.6 BG17 secondary text contrast", () => {
  it("BG17: the contrast helper reproduces the bug's own example (#A0A0A0 on white fails AA)", () => {
    expect(contrast("#a0a0a0", "#ffffff")).toBeLessThan(4.5);
    expect(contrast("#666666", "#ffffff")).toBeGreaterThanOrEqual(4.5);
  });

  it("BG17: light-theme subheader, empty-state and muted text meet WCAG AA (4.5:1) on the page background", () => {
    const root = block(":root");
    const background = resolve(root, root["--background"]);
    for (const token of ["--color-text-tertiary", "--color-text-quaternary", "--text-secondary", "--muted-foreground"]) {
      expect(root[token], `${token} declared in :root`).toBeDefined();
      const ratio = contrast(resolve(root, root[token]), background);
      expect(ratio, `${token} contrast ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("BG17: dark-theme tertiary and muted text also meet WCAG AA on the dark background", () => {
    const dark = block(".dark");
    const background = resolve(dark, dark["--background"]);
    for (const token of ["--text-tertiary", "--text-secondary", "--muted-foreground"]) {
      const ratio = contrast(resolve(dark, dark[token]), background);
      expect(ratio, `${token} contrast ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(4.5);
    }
  });
});
