import { scalePolicy } from "./chart-semantics";

export function scatterScale(type: string, values: unknown[], choice?: "linear" | "log" | null) {
  const policy = scalePolicy(values);
  return type === "scatter" && policy.automatic && choice !== "linear";
}

export function niceAxis(values: unknown[], startAtZero = false) {
  const numbers = values.filter((value) => value != null && value !== "").map(Number).filter(Number.isFinite);
  let min = numbers.length ? Math.min(...numbers) : 0;
  let max = numbers.length ? Math.max(...numbers) : 1;
  if (startAtZero) { min = Math.min(0, min); max = Math.max(0, max); }
  if (min === max) { min = Math.min(0, min); max = max > 0 ? max * 1.2 : 1; }
  const rawStep = Math.max((max - min) / 5, 0.01);
  const power = 10 ** Math.floor(Math.log10(rawStep));
  const steps = [1, 2, 2.5, 5, 10, 20];
  for (const multiplier of steps) {
    const step = multiplier * power;
    if (step < rawStep || step < 0.01) continue;
    const low = Math.floor(min / step) * step;
    const high = Math.ceil(max / step) * step;
    const count = Math.round((high - low) / step) + 1;
    if (count > 6) continue;
    const ticks = Array.from({ length: count }, (_, i) => Number((low + i * step).toFixed(10)));
    return { ticks, domain: [ticks[0], ticks[ticks.length - 1]] as [number, number], step };
  }
  return { ticks: [min, max], domain: [min, max] as [number, number], step: max - min };
}

export function measureLabel(value: unknown): string {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value ?? "");
  const magnitude = Math.abs(number);
  const unit = magnitude >= 1e9 ? [1e9, "B"] : magnitude >= 1e6 ? [1e6, "M"] : magnitude >= 1e3 ? [1e3, "K"] : [1, ""];
  return `${Number((number / Number(unit[0])).toFixed(2))}${unit[1]}`;
}

export function logarithmicAxis(values: unknown[]) {
  const policy = scalePolicy(values);
  if (!policy.positive) return null;
  const low = Math.floor(Math.log10(policy.min));
  const high = Math.ceil(Math.log10(policy.max));
  const ticks = subsetLabels(Array.from({ length: high - low + 1 }, (_, i) => 10 ** (low + i)));
  return { ticks, domain: [10 ** low, 10 ** high] as [number, number] };
}

export function binLabel(value: unknown): string {
  const text = String(value ?? "");
  if (text.endsWith("+")) return text;
  if (/^-?\d+(?:\.\d+)?$/.test(text)) return measureLabel(text);
  return text.replace(/-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g, (match) => measureLabel(match));
}

export function subsetLabels<T>(values: T[]): T[] {
  return values.length <= 6 ? values : Array.from(new Set(Array.from({ length: 6 }, (_, i) => values[Math.round(i * (values.length - 1) / 5)])));
}

export function measureTitle(field: string, aggregate?: string | null, rateValue?: string | null) {
  if (aggregate === "rate") return rateValue ? `% ${rateValue}` : `% ${field}`;
  const name = ({ mean: "Average", avg: "Average", average: "Average", sum: "Sum", median: "Median", min: "Minimum", max: "Maximum" } as Record<string, string>)[aggregate ?? ""];
  return name ? `${name} ${field}` : aggregate === "count" ? "Count" : field;
}