// Small client-side cache mapping backend task_id -> the prompt text the user
// typed when they created it. GET /tasks does NOT return a name/prompt, so we
// remember it here to render a friendly card title.

const KEY = "avaloka:scheduled-prompt-cache";

function readAll(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeAll(map: Record<string, string>) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(map));
  } catch {}
}

export const scheduledPromptCache = {
  get(taskId: string): string | undefined {
    return readAll()[taskId];
  },
  set(taskId: string, prompt: string) {
    const all = readAll();
    all[taskId] = prompt;
    writeAll(all);
  },
  remove(taskId: string) {
    const all = readAll();
    delete all[taskId];
    writeAll(all);
  },
  all(): Record<string, string> {
    return readAll();
  },
};

export function summarizePrompt(prompt: string, max = 60): string {
  const first = prompt.split(/[.!?\n]/)[0]?.trim() ?? prompt.trim();
  if (first.length <= max) return first;
  return first.slice(0, max - 3) + "…";
}

/** Turn either an interval-schedule ({ second: 60 }) or a crontab-schedule
 *  ({ minute, hour, day_of_week, ... }) into a readable string. */
export function formatSchedule(schedule: any): string {
  if (!schedule || typeof schedule !== "object") return "Scheduled";

  // Interval form
  const intervalKeys = ["second", "seconds", "minute", "minutes", "hour", "hours", "day", "days"];
  const only = Object.keys(schedule).filter((k) => schedule[k] != null);
  if (only.length === 1 && intervalKeys.includes(only[0])) {
    const k = only[0];
    const v = Number(schedule[k]);
    if (k.startsWith("second")) {
      if (v === 60) return "Every minute";
      if (v % 3600 === 0) return `Every ${v / 3600} hour${v / 3600 === 1 ? "" : "s"}`;
      if (v % 60 === 0) return `Every ${v / 60} minute${v / 60 === 1 ? "" : "s"}`;
      return `Every ${v}s`;
    }
    if (k.startsWith("minute")) return v === 1 ? "Every minute" : `Every ${v} minutes`;
    if (k.startsWith("hour")) return v === 1 ? "Every hour" : `Every ${v} hours`;
    if (k.startsWith("day")) return v === 1 ? "Every day" : `Every ${v} days`;
  }

  // Crontab form
  const cronKeys = ["minute", "hour", "day_of_month", "month_of_year", "day_of_week"];
  if (cronKeys.some((k) => k in schedule)) {
    const min = schedule.minute;
    const hr = schedule.hour;
    const dom = schedule.day_of_month;
    const mon = schedule.month_of_year;
    const dow = schedule.day_of_week;
    const isAny = (v: any) => v == null || v === "*";
    const asNum = (v: any) => (typeof v === "number" ? v : /^\d+$/.test(String(v)) ? Number(v) : null);

    if (isAny(min) && isAny(hr) && isAny(dom) && isAny(mon) && isAny(dow)) return "Every minute";
    const mn = asNum(min);
    if (mn != null && isAny(hr) && isAny(dom) && isAny(mon) && isAny(dow)) {
      return mn === 0 ? "Every hour" : `Every hour at :${String(mn).padStart(2, "0")}`;
    }
    const hn = asNum(hr);
    if (mn != null && hn != null && isAny(dom) && isAny(mon) && isAny(dow)) {
      return `Every day at ${String(hn).padStart(2, "0")}:${String(mn).padStart(2, "0")}`;
    }
    const stepMin = typeof min === "string" ? /^\*\/(\d+)$/.exec(min) : null;
    if (stepMin && isAny(hr) && isAny(dom) && isAny(mon) && isAny(dow)) {
      return `Every ${stepMin[1]} minutes`;
    }
    const stepHr = typeof hr === "string" ? /^\*\/(\d+)$/.exec(hr) : null;
    if (stepHr && isAny(min) && isAny(dom) && isAny(mon) && isAny(dow)) {
      return `Every ${stepHr[1]} hours`;
    }

    const fmt = (v: any) => (isAny(v) ? "*" : String(v));
    const parts = [`min=${fmt(min)}`, `hr=${fmt(hr)}`];
    if (!isAny(dom)) parts.push(`dom=${dom}`);
    if (!isAny(mon)) parts.push(`mon=${mon}`);
    if (!isAny(dow)) parts.push(`dow=${dow}`);
    return `Cron ${parts.join(" ")}`;
  }

  try {
    return JSON.stringify(schedule);
  } catch {
    return "Scheduled";
  }
}

export function relativeTime(iso?: string | null): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return String(iso);
  const diff = Date.now() - t;
  const abs = Math.abs(diff);
  const s = Math.round(abs / 1000);
  const m = Math.round(s / 60);
  const h = Math.round(m / 60);
  const d = Math.round(h / 24);
  const rel =
    s < 45 ? `${s}s` :
    m < 60 ? `${m}m` :
    h < 24 ? `${h}h` :
    `${d}d`;
  return diff >= 0 ? `${rel} ago` : `in ${rel}`;
}
