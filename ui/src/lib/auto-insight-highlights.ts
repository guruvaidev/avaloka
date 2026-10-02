import type { InsightGroup } from "./insight-groups";
import type { Slide } from "@/components/dashboard/dynamicChart";

export interface AutoInsightHighlight {
  datasetId: string;
  datasetName: string;
  text: string;
}

/** Chart selection reasons are descriptions, not evidence-backed findings. */
export function collectChartFindings(slides: Pick<Slide, "insights">[]): string[] {
  const findings: string[] = [];
  const seen = new Set<string>();
  for (const slide of slides) {
    for (const item of slide.insights) {
      const text = item.trim();
      if (!text || seen.has(text)) continue;
      seen.add(text);
      findings.push(text);
    }
  }
  return findings;
}

function usefulColumnName(name: string): boolean {
  const label = name.trim();
  return (
    !!label &&
    !/^unnamed(?:\s*[:.]\s*\d+)?$/i.test(label) &&
    !/^(?:row[ _-]?)?(?:id|index|uuid|guid)$/i.test(label)
  );
}

function sampleFinding(samples: unknown[] | undefined): string | null {
  const rows = (samples ?? []).filter(
    (row): row is Record<string, unknown> =>
      !!row && typeof row === "object" && !Array.isArray(row),
  );
  if (rows.length < 3) return null;

  let bestCategory: { score: number; text: string } | null = null;
  let bestNumber: { score: number; text: string } | null = null;
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];

  for (const column of columns) {
    if (!usefulColumnName(column)) continue;
    const values = rows
      .map((row) => row[column])
      .filter(
        (value): value is string | number | boolean =>
          (typeof value === "string" || typeof value === "number" || typeof value === "boolean") &&
          String(value).trim() !== "",
      );
    if (values.length < 3 || values.length / rows.length < 0.5) continue;

    const numbers = values.map((value) => (typeof value === "boolean" ? NaN : Number(value)));
    const numericCount = numbers.filter(Number.isFinite).length;
    if (numericCount / values.length >= 0.8) {
      const finite = numbers.filter(Number.isFinite).sort((a, b) => a - b);
      const unique = [...new Set(finite)];
      if (unique.length < 2) continue;
      // Sequential row numbers are identifiers, even when their header is vague.
      if (
        unique.length >= finite.length * 0.8 &&
        unique.every((value, i) => value === unique[0] + i)
      )
        continue;
      const median =
        finite.length % 2
          ? finite[Math.floor(finite.length / 2)]
          : (finite[finite.length / 2 - 1] + finite[finite.length / 2]) / 2;
      const format = (value: number) =>
        new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
      const text = `In ${finite.length} sampled rows, the median ${column.trim()} is ${format(median)} (range ${format(finite[0])}–${format(finite[finite.length - 1])}).`;
      const score = (finite.length / rows.length) * Math.log1p(unique.length);
      if (!bestNumber || score > bestNumber.score) bestNumber = { score, text };
      continue;
    }

    const labels = values.map((value) => String(value).trim());
    if (labels.filter((value) => value.length <= 64).length / labels.length < 0.9) continue;
    const counts = new Map<string, number>();
    labels.forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1));
    if (counts.size < 2 || counts.size > 20 || counts.size > labels.length / 2) continue;
    const [topValue, topCount] = [...counts].sort((a, b) => b[1] - a[1])[0];
    const percent = Math.round((100 * topCount) / labels.length);
    const text = `In ${labels.length} sampled rows with ${column.trim()}, “${topValue}” is most common (${topCount}, ${percent}%).`;
    const score = (labels.length / rows.length) * (topCount / labels.length);
    if (!bestCategory || score > bestCategory.score) bestCategory = { score, text };
  }

  return bestCategory?.text ?? bestNumber?.text ?? null;
}

function chartFinding(slides: Slide[]): string | null {
  for (const slide of slides) {
    if (slide.chartKind === "histogram" || !["bar", "pie", "donut"].includes(slide.type)) continue;
    const field = (slide.xField || slide.xKey).trim();
    if (!usefulColumnName(field)) continue;
    const measure = slide.series[0]?.dataKey;
    if (!measure || !Array.isArray(slide.data)) continue;
    const values = slide.data
      .map((row) => ({ category: row?.[slide.xKey], value: Number(row?.[measure]) }))
      .filter(
        (row) => row.category != null && String(row.category).trim() && Number.isFinite(row.value),
      );
    if (values.length < 2) continue;
    const top = values.reduce((best, row) => (row.value > best.value ? row : best));
    const format = (value: number) =>
      new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
    const category = String(top.category).trim();
    const isCount = slide.aggregate === "count" || /^count$/i.test(slide.series[0]?.name ?? "");
    const total = values.reduce((sum, row) => sum + row.value, 0);
    if (isCount && total > 0 && values.every((row) => row.value >= 0)) {
      return `In the plotted sample, “${category}” is the most common ${field} (${format(top.value)} of ${format(total)}, ${Math.round((100 * top.value) / total)}%).`;
    }
    return `In the plotted sample, “${category}” has the highest ${slide.series[0]?.name || measure} across ${field} (${format(top.value)}).`;
  }
  return null;
}

/** One calculated finding per dataset (including each workbook sheet). */
export function collectAutoInsightHighlights(
  groups: Pick<InsightGroup, "id" | "name" | "slides" | "samples">[],
): AutoInsightHighlight[] {
  return groups.flatMap((group) => {
    const text =
      sampleFinding(group.samples) ??
      chartFinding(group.slides) ??
      collectChartFindings(group.slides)[0];
    return text ? [{ datasetId: group.id, datasetName: group.name, text }] : [];
  });
}
