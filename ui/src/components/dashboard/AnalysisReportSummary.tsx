import { useMemo } from "react";
import { AlertTriangle, Check, ChevronDown } from "@untitledui/icons";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cx } from "@/lib/utils/cx";

type Row = Record<string, unknown>;
type Source = { key: string; values: Row };
const stats = ["count", "mean", "std", "min", "25%", "50%", "median", "75%", "max", "sum"] as const;
const visibleStats = ["count", "mean", "std", "min", "25%", "median", "75%", "max", "sum"] as const;
const issuesPattern = /^(issues|data_quality|warnings|notes)$/i;
const isRecord = (value: unknown): value is Row =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const isScalar = (value: unknown) =>
  value === null || ["string", "number", "boolean"].includes(typeof value);

export function isReportResult(rows?: Row[] | null): boolean {
  if (!rows || rows.length !== 1 || !isRecord(rows[0])) return false;
  const values = Object.values(rows[0]);
  return (
    values.length > 6 ||
    values.some(
      (value) =>
        (value !== null && typeof value === "object") ||
        (typeof value === "string" && /^(\{[\s\S]*\}|\[[\s\S]*\])$/.test(value.trim())),
    )
  );
}

/** Parse JSON and common Python repr values without evaluating the input. */
export function parseStructured(value: unknown): Row | unknown[] | null {
  if (value !== null && typeof value === "object") return value as Row | unknown[];
  if (typeof value !== "string") return null;
  const text = value.trim();
  if (
    !((text.startsWith("{") && text.endsWith("}")) || (text.startsWith("[") && text.endsWith("]")))
  )
    return null;
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed !== null && typeof parsed === "object" ? (parsed as Row | unknown[]) : null;
  } catch {
    /* Try Python repr below. */
  }
  try {
    let converted = "";
    let outside = "";
    const flush = () => {
      converted += outside.replace(/\b(True|False|None|nan|NaN|inf)\b/g, (word) =>
        word === "True" ? "true" : word === "False" ? "false" : "null",
      );
      outside = "";
    };
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char !== "'" && char !== '"') {
        outside += char;
        continue;
      }
      flush();
      const quote = char;
      let content = "";
      let closed = false;
      while (++i < text.length) {
        const next = text[i];
        if (next === quote) {
          closed = true;
          break;
        }
        if (next === "\\" && i + 1 < text.length) {
          const escaped = text[++i];
          if (escaped === "n") content += "\n";
          else if (escaped === "t") content += "\t";
          else if (escaped === "r") content += "\r";
          else if (escaped === "u" && /^[0-9a-fA-F]{4}$/.test(text.slice(i + 1, i + 5))) {
            content += String.fromCharCode(parseInt(text.slice(i + 1, i + 5), 16));
            i += 4;
          } else content += escaped;
        } else content += next;
      }
      if (!closed) return null;
      converted += JSON.stringify(content);
    }
    flush();
    const parsed: unknown = JSON.parse(converted);
    return parsed !== null && typeof parsed === "object" ? (parsed as Row | unknown[]) : null;
  } catch {
    return null;
  }
}

const humanize = (key: string) =>
  key
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (c) => c.toUpperCase());
const sourceLabel = (key: string) =>
  /dtype|type/i.test(key)
    ? "Type"
    : /missing/i.test(key)
      ? "Missing"
      : humanize(key.replace(/^column_/, ""));
function display(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number")
    return Number.isFinite(value)
      ? value.toLocaleString("en-US", {
          minimumFractionDigits: Number.isInteger(value) ? 0 : 2,
          maximumFractionDigits: 2,
        })
      : "—";
  if (typeof value === "boolean") return String(value);
  if (typeof value === "string")
    return /^(undefined|nan|\[object Object\])$/i.test(value.trim()) ? "—" : value.trim() || "—";
  try {
    return JSON.stringify(value) ?? "—";
  } catch {
    return "—";
  }
}
const numberValue = (value: unknown) =>
  typeof value === "number"
    ? value
    : typeof value === "string" && value.trim()
      ? Number(value)
      : NaN;
const statKey = (key: string) => {
  const match = key.match(/^(.*?)_(count|mean|std|min|25%|50%|75%|max|median|sum)$/i);
  return match && match[1] ? { prefix: match[1], stat: match[2].toLowerCase() } : null;
};
const overlap = (a: Source, b: Source) => {
  const ka = Object.keys(a.values),
    kb = Object.keys(b.values);
  return (
    ka.length > 0 &&
    kb.length > 0 &&
    ka.filter((key) => key in b.values).length / Math.max(ka.length, kb.length) >= 0.8
  );
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="min-w-0 rounded-lg border border-secondary bg-primary p-4">
      <h3 className="mb-3 text-sm font-semibold text-primary">{title}</h3>
      {children}
    </section>
  );
}

export function AnalysisReportSummary({ row }: { row: Row }) {
  const report = useMemo(() => {
    const entries = Object.entries(row);
    const grouped = new Map<string, Row>();
    for (const [key, value] of entries) {
      const match = statKey(key);
      if (match) grouped.set(match.prefix, { ...grouped.get(match.prefix), [match.stat]: value });
    }
    const active = [...grouped].filter(([, values]) => Object.keys(values).length >= 3);
    const statKeys = new Set(
      entries
        .filter(([key]) => active.some(([prefix]) => statKey(key)?.prefix === prefix))
        .map(([key]) => key),
    );
    const sources: Source[] = [];
    const used = new Set(statKeys);
    for (const [key, value] of entries) {
      const parsed = parseStructured(value);
      if (isRecord(parsed) && Object.keys(parsed).length && Object.values(parsed).every(isScalar)) {
        sources.push({ key, values: parsed });
        used.add(key);
      }
    }
    const merged: Source[][] = [];
    const remaining = [...sources];
    while (remaining.length) {
      const first = remaining.shift();
      if (!first) break;
      const group = [first];
      for (let i = remaining.length - 1; i >= 0; i--) {
        if (group.every((s) => overlap(s, remaining[i]))) group.push(...remaining.splice(i, 1));
      }
      merged.push(
        group.sort(
          (a, b) =>
            entries.findIndex(([key]) => key === a.key) -
            entries.findIndex(([key]) => key === b.key),
        ),
      );
    }
    const issues = entries.filter(
      ([key, value]) => issuesPattern.test(key) && typeof value === "string",
    ) as [string, string][];
    issues.forEach(([key]) => used.add(key));
    const headlines = entries.filter(
      ([key, value]) =>
        !used.has(key) && isScalar(value) && !(typeof value === "string" && value.length > 160),
    );
    headlines.forEach(([key]) => used.add(key));
    return { active, merged, issues, headlines, extras: entries.filter(([key]) => !used.has(key)) };
  }, [row]);
  const totalRows = numberValue(row.total_rows);
  const statsShown = visibleStats.filter((stat) =>
    report.active.some(([, values]) =>
      stat === "median"
        ? values.median !== undefined || values["50%"] !== undefined
        : values[stat] !== undefined,
    ),
  );

  const cell = (value: unknown, kind: string) => {
    if (/missing/i.test(kind)) {
      const count = numberValue(value);
      if (Number.isFinite(count) && count > 0)
        return (
          <span className="text-warning-primary">
            {display(value)}
            {Number.isFinite(totalRows) && totalRows > 0
              ? ` (${display((count / totalRows) * 100)}%)`
              : ""}
          </span>
        );
    }
    if (/dtype|type/i.test(kind))
      return (
        <code className="rounded border border-secondary bg-secondary px-1.5 py-0.5 text-xs text-secondary">
          {display(value)}
        </code>
      );
    return display(value);
  };
  return (
    <div className="mt-4 min-w-0 space-y-4" aria-label="Summary">
      {report.headlines.length > 0 && (
        <Section title="Headline metrics">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {report.headlines.map(([key, value]) => (
              <div
                key={key}
                className="min-w-0 rounded-md border border-secondary bg-secondary/40 p-3"
              >
                <p className="text-xs text-tertiary">{humanize(key)}</p>
                <p className="mt-1 break-words text-lg font-semibold text-primary">
                  {display(value)}
                </p>
              </div>
            ))}
          </div>
        </Section>
      )}
      {report.active.length > 0 && (
        <Section title="Summary statistics">
          <div className="max-w-full overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-secondary text-tertiary">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Metric</th>
                  {statsShown.map((stat) => (
                    <th key={stat} className="whitespace-nowrap px-3 py-2 text-right font-medium">
                      {stat === "median"
                        ? "Median (50%)"
                        : stat === "std"
                          ? "Std"
                          : stat === "25%" || stat === "75%"
                            ? stat
                            : humanize(stat)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {report.active.map(([prefix, values]) => (
                  <tr key={prefix} className="border-t border-secondary">
                    <th className="px-3 py-2 text-left font-medium text-primary">
                      {humanize(prefix)}
                    </th>
                    {statsShown.map((stat) => {
                      const value =
                        stat === "median" ? (values.median ?? values["50%"]) : values[stat];
                      const negativeMin = stat === "min" && numberValue(value) < 0;
                      return (
                        <td
                          key={stat}
                          className={cx(
                            "whitespace-nowrap px-3 py-2 text-right",
                            negativeMin ? "text-warning-primary" : "text-secondary",
                          )}
                        >
                          {negativeMin ? (
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span
                                    tabIndex={0}
                                    className="cursor-help underline decoration-dotted"
                                  >
                                    {display(value)}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent>Negative value</TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          ) : (
                            display(value)
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
      {report.merged.map((group) => {
        const names = [...new Set(group.flatMap((source) => Object.keys(source.values)))];
        const allMissingZero =
          group.some((source) => /missing/i.test(source.key)) &&
          group
            .filter((source) => /missing/i.test(source.key))
            .every((source) =>
              Object.values(source.values).every((value) => numberValue(value) === 0),
            );
        return (
          <Section
            key={group.map((s) => s.key).join("|")}
            title={group.length > 1 ? "Per-column details" : humanize(group[0].key)}
          >
            {allMissingZero && (
              <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-success-primary">
                <Check className="size-4" />
                No missing values
              </p>
            )}
            <div
              className={cx(
                "max-w-full overflow-x-auto",
                names.length > 15 && "max-h-96 overflow-y-auto",
              )}
            >
              <table className="w-full text-sm">
                <thead
                  className={cx(
                    "bg-secondary text-tertiary",
                    names.length > 15 && "sticky top-0 z-10",
                  )}
                >
                  <tr>
                    <th className="whitespace-nowrap px-3 py-2 text-left font-medium">
                      {group.length > 1 ? "Column" : "Key"}
                    </th>
                    {group.map((source) => (
                      <th
                        key={source.key}
                        className="whitespace-nowrap px-3 py-2 text-left font-medium"
                      >
                        {group.length > 1 ? sourceLabel(source.key) : "Value"}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {names.map((name) => (
                    <tr key={name} className="border-t border-secondary">
                      <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-primary">
                        {name}
                      </th>
                      {group.map((source) => (
                        <td key={source.key} className="px-3 py-2 text-secondary">
                          {cell(source.values[name], source.key)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        );
      })}
      {report.issues.length > 0 && (
        <Section title="Data-quality issues">
          <ul className="space-y-2">
            {report.issues.flatMap(([key, text]) =>
              text
                .split(/\s*\|\s*|[;\n]/)
                .map((item) => item.trim())
                .filter(Boolean)
                .map((item, i) => (
                  <li
                    key={`${key}-${i}`}
                    className={cx(
                      "flex items-start gap-2 text-sm",
                      item.toLowerCase() === "no issues detected"
                        ? "text-success-primary"
                        : "text-secondary",
                    )}
                  >
                    {item.toLowerCase() === "no issues detected" ? (
                      <Check className="mt-0.5 size-4 shrink-0" />
                    ) : (
                      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning-primary" />
                    )}
                    <span className="break-words">{item}</span>
                  </li>
                )),
            )}
          </ul>
        </Section>
      )}
      {report.extras.length > 0 && (
        <Section title="Other details">
          <div className="space-y-2">
            {report.extras.map(([key, value]) => {
              const structured = parseStructured(value);
              return structured ? (
                <details key={key} className="rounded-md border border-secondary p-3">
                  <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium text-primary">
                    {humanize(key)} <ChevronDown className="size-4" />
                  </summary>
                  <div className="mt-3 space-y-2">
                    {(Array.isArray(structured)
                      ? structured.map((item, i) => [String(i + 1), item])
                      : Object.entries(structured)
                    ).map(([name, item]) => (
                      <div
                        key={String(name)}
                        className="grid gap-1 border-t border-secondary pt-2 text-sm sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]"
                      >
                        <span className="break-words font-medium text-primary">{String(name)}</span>
                        <span className="break-words text-secondary">{display(item)}</span>
                      </div>
                    ))}
                  </div>
                </details>
              ) : (
                <div key={key} className="text-sm">
                  <p className="font-medium text-primary">{humanize(key)}</p>
                  <p className="mt-1 whitespace-pre-wrap break-words text-secondary">
                    {display(value)}
                  </p>
                </div>
              );
            })}
          </div>
        </Section>
      )}
    </div>
  );
}
