// avaloka/outcomes/assets/outcome-runtime.js: the formatter and the what-if
// scenario model the HTML export inlines. It is held to the same golden
// vectors as avaloka/outcomes/{format,scenario}.py (tests/avaloka/outcomes/
// test_spec.py runs them too), so Python and JS print the same strings and
// compute the same floats. docs/design/2.0-OUTCOMES.md §2.2, §4.4, §7 S2.
//
// The runtime is loaded the way the export uses it: as a plain script, in a
// fresh context where string code generation is off, which is what a CSP
// without 'unsafe-eval' enforces. Any eval or Function() would throw here.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import ts from "typescript";
import { describe, expect, it } from "vitest";

const OUTCOMES = resolve(dirname(fileURLToPath(import.meta.url)), "../../../avaloka/outcomes");
const outcomes = (p: string) => resolve(OUTCOMES, p);
const SOURCE = readFileSync(outcomes("assets/outcome-runtime.js"), "utf8");
const golden = (name: string) => JSON.parse(readFileSync(outcomes(`golden/${name}`), "utf8"));
const FORMAT = golden("format.json");
const SCENARIO = golden("scenario.json");

type Figure = {
  id?: string; value: unknown; unit: string; decimals: unknown; currency?: string; compact?: unknown;
};
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type GoldenCase = { name: string; [key: string]: any };
type Runtime = {
  SPEC_VERSION: string;
  UNITS: string[];
  SURE_TEXT: Record<string, string>;
  roundHalfEven(value: number, decimals: number, shift?: number): string;
  formatNumber(value: unknown, unit: string, decimals: unknown, currency?: string, signed?: boolean,
               compact?: unknown): string;
  formatFigure(figure: Figure, opts?: { signed?: boolean }): string;
  renderText(text: string, figures: Figure[] | Record<string, Figure>): string;
  evaluate(whatif: unknown, values?: Record<string, unknown>, opts?: { conformalQ?: number }): unknown;
};

function load(): Runtime {
  const context = vm.createContext({}, { codeGeneration: { strings: false, wasm: false } });
  vm.runInContext(SOURCE, context, { filename: "outcome-runtime.js" });
  return (context as { AvalokaOutcome: Runtime }).AvalokaOutcome;
}

// Loaded once; a runtime that cannot load under the CSP fails the first test
// by name rather than only the collection.
let loadError: unknown;
let R = {} as Runtime;
try {
  R = load();
} catch (err) {
  loadError = err;
}

// Objects made in the script's context have that context's prototypes; compare
// plain data.
const plain = (value: unknown) => JSON.parse(JSON.stringify(value));

function codeOf(fn: () => unknown): string | undefined {
  try {
    fn();
  } catch (err) {
    return (err as { code?: string }).code;
  }
  return undefined;
}

describe("outcome-runtime.js is a CSP-safe ES2019 script", () => {
  it("defines one global and runs with string code generation disabled", () => {
    expect(loadError).toBeUndefined();
    expect(R.SPEC_VERSION).toBe("1.0");
    const context = vm.createContext({}, { codeGeneration: { strings: false, wasm: false } });
    vm.runInContext(SOURCE, context);
    expect(Object.keys(context)).toEqual(["AvalokaOutcome"]);
    // The guard is real: the same context refuses eval.
    expect(() => vm.runInContext("eval('1')", context)).toThrow(/Code generation from strings/);
  });

  it("uses no syntax after ES2019, no module syntax and no code-from-string calls", () => {
    const after2019 = new Set([
      ts.SyntaxKind.QuestionDotToken,
      ts.SyntaxKind.QuestionQuestionToken,
      ts.SyntaxKind.QuestionQuestionEqualsToken,
      ts.SyntaxKind.BarBarEqualsToken,
      ts.SyntaxKind.AmpersandAmpersandEqualsToken,
      ts.SyntaxKind.PrivateIdentifier,
      ts.SyntaxKind.BigIntLiteral,
    ]);
    const banned = new Set(["eval", "Function", "import", "export", "setTimeout", "setInterval"]);
    const scanner = ts.createScanner(ts.ScriptTarget.Latest, true, ts.LanguageVariant.Standard, SOURCE);
    const found: string[] = [];
    for (let kind = scanner.scan(); kind !== ts.SyntaxKind.EndOfFileToken; kind = scanner.scan()) {
      const text = scanner.getTokenText();
      if (after2019.has(kind)) found.push(text);
      if (kind === ts.SyntaxKind.NumericLiteral && text.includes("_")) found.push(text);
      if ((kind === ts.SyntaxKind.Identifier ||
          (kind >= ts.SyntaxKind.FirstKeyword && kind <= ts.SyntaxKind.LastKeyword)) && banned.has(text)) found.push(text);
    }
    expect(found).toEqual([]);
  });
});

describe("formatter parity with avaloka/outcomes/format.py", () => {
  it("has the golden cases §7 S2 asks for", () => {
    expect(FORMAT.format.length).toBeGreaterThanOrEqual(60);
  });

  it.each(FORMAT.format as GoldenCase[])("$name", (c) => {
    const f = c.figure as Figure;
    const call = () =>
      R.formatNumber(f.value, f.unit, f.decimals, f.currency, Boolean(c.signed), f.compact);
    if (c.error) {
      expect(codeOf(call)).toBe(c.error);
    } else {
      expect(call()).toBe(c.expected);
    }
  });

  it("rounds half-even on the printed decimal, with an exact shift", () => {
    for (const c of FORMAT.round) {
      expect(R.roundHalfEven(c.value, c.decimals, c.shift)).toBe(c.expected);
    }
  });

  it("substitutes placeholders exactly as render_text does", () => {
    for (const c of FORMAT.substitute.cases) {
      const call = () => R.renderText(c.text, FORMAT.substitute.figures);
      if (c.error) {
        expect(codeOf(call)).toBe(c.error);
      } else {
        expect(call()).toBe(c.expected);
      }
    }
  });

  it("renders every text slot of both §4.1 examples", () => {
    const retail = golden("example_retail.json");
    expect(R.renderText(retail.headline.text, retail.figures)).toBe(
      "Net sales grew 8.4% year on year; price did most of the work.",
    );
    expect(R.renderText(retail.headline.bullets[0].so_what, retail.figures)).toBe(
      "Price added $1.9M of the $2.6M rise; volume added $700K.",
    );
    const fintech = golden("example_fintech.json");
    expect(R.renderText(fintech.headline.bullets[1].so_what, fintech.figures)).toBe(
      "The top 10 merchants hold 37% of exposure.",
    );
  });
});

describe("scenario parity with avaloka/outcomes/scenario.py", () => {
  it("has the same how-sure sentences", () => {
    expect(plain(R.SURE_TEXT)).toEqual(SCENARIO.sure_text);
  });

  it.each(SCENARIO.cases as GoldenCase[])("$name", (c) => {
    const opts = c.conformal_q === undefined ? undefined : { conformalQ: c.conformal_q };
    const call = () => R.evaluate(c.whatif, c.values, opts);
    if (c.error) {
      expect(codeOf(call)).toBe(c.error);
    } else {
      // Exact float equality: same operations in the same order on both sides.
      expect(plain(call())).toEqual(c.expected);
    }
  });

  it("requested above P95 is clamped", () => {
    const whatif = {
      id: "w1", kpi: "net_sales", y0: 100, z: 1.96, evidence_kind: "associational",
      evidence: { turn_id: "t1" },
      drivers: [{ id: "discount", label: "Discount", unit: "percent", baseline: 12, lo: 5, hi: 25,
                  form: "linear", beta: 0.8, se: 0.1 }],
    };
    const out = plain(R.evaluate(whatif, { discount: 40 }));
    expect(out.applied).toEqual({ discount: 25 });
    expect(out.clamped).toEqual([{ driver: "discount", requested: 40, applied: 25, bound: "hi" }]);
    expect(out.result.y).toBeCloseTo(110.4, 12);
    expect(out.result.sure).toBe("direction_only");
    expect(plain(R.evaluate(whatif, { discount: 25 })).result.sure).toBe("fairly_sure");
  });

  it("recomputes a five-driver panel within the 2 ms slider budget (§6.3)", () => {
    const drivers = [0, 1, 2, 3, 4].map((i) => ({
      id: `d${i}`, label: `D${i}`, unit: "unitless", baseline: 10, lo: 0, hi: 20,
      form: "linear", beta: 0.5 + i, se: 0.1 * (i + 1),
    }));
    const whatif = { id: "w1", kpi: "k", y0: 50, drivers, evidence_kind: "associational",
                     evidence: { turn_id: "t" } };
    const runs = 2000;
    const start = performance.now();
    for (let n = 0; n < runs; n++) {
      R.evaluate(whatif, { d0: n % 20, d1: 3, d2: 17, d3: 25, d4: -1 });
    }
    expect((performance.now() - start) / runs).toBeLessThan(2);
  });
});
